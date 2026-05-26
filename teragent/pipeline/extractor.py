"""teragent.pipeline.extractor — File extraction from LLM responses

3-level degradation strategy:
    1a. Strict XML: <file path='...'>...</file> (requires quotes)
    1b. Lenient XML: <file path/name=...>...</file> (flexible attributes)
    1c. No-quote XML: <file path=foo.py>...</file>
    1d. Triple-backtick with path: ```python:src/main.py
    2.  Markdown code blocks with filename hints
    3.  Fallback: infer filenames from code content
"""
import re
import logging

logger = logging.getLogger(__name__)

# v1.0.4: Extended XML patterns — supports path/name attributes, quoted/unquoted, triple-backtick+path
FILE_BLOCK_PATTERN_STRICT = r'<file\s+path=["\'](.*?)["\']\s*>(.*?)</file>'
FILE_BLOCK_PATTERN_LENIENT = r'<file\s+(?:path|name)=["\']?(.*?)["\']?\s*>(.*?)</file>'
FILE_BLOCK_PATTERN_NOQUOTE = r'<file\s+(?:path|name)=([^"\'>\s]+)\s*>(.*?)</file>'
FENCE_PATH_PATTERN = r'```[\w]*:([\w/.\-]+\.(?:py|txt|toml|yaml|yml|json|md|cfg|ini|sh|bat|html|css|js|ts))\n(.*?)\n```'
MD_CODE_PATTERN = r'```[\w]*\n(.*?)\n```'

# Filename extraction patterns (by priority)
# 1. markdown heading + file path: ### game/entities/snake.py or **game/types.py**
MD_HEADING_PATTERN = r'(?:^|\n)(?:#{1,4}\s+|(?:\*\*|__))([\w/.\-]+\.(?:py|txt|toml|yaml|yml|json|md|cfg|ini|sh|bat|html|css|js|ts))[:\s]*(?:\*\*|__)?'
# 2. inline comment: # file: game/types.py or # path: game/config.py
COMMENT_FILE_PATTERN = r'#\s*(?:file|path|filename|File|Path)\s*[:=]\s*([\w/.\-]+\.(?:py|txt|toml|yaml|yml|json|md|cfg|ini|sh|bat|html|css|js|ts))'
# 3. code block language tag + filename: ```python:game/snake.py or ```python game/snake.py
FENCE_FILENAME_PATTERN = r'```[\w]*[:\s]+([\w/.\-]+\.(?:py|txt|toml|yaml|yml|json|md|cfg|ini|sh|bat|html|css|js|ts))'
# 4. "Create/Write file path" format: 创建 game/entities/snake.py or 写入 src/config.py
CREATE_FILE_PATTERN = r'(?:创建|写入|修改|编辑|新建|Create|Write|Modify|Edit)\s+(?:文件\s*)?[:：]?\s*`?([\w/.\-]+\.(?:py|txt|toml|yaml|yml|json|md|cfg|ini|sh|bat|html|css|js|ts))`?'


def _extract_filename_hints(content: str) -> list[str]:
    """Extract all filename hints from content, ordered by position."""
    hints_with_pos: list[tuple[int, str]] = []

    for pattern in [MD_HEADING_PATTERN, COMMENT_FILE_PATTERN, FENCE_FILENAME_PATTERN, CREATE_FILE_PATTERN]:
        for match in re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE):
            filename = match.group(1).strip()
            # Dedup: don't add same filename at nearly same position
            pos = match.start()
            if not any(abs(pos - existing_pos) < 5 and filename == existing_name
                      for existing_pos, existing_name in hints_with_pos):
                hints_with_pos.append((pos, filename))

    # Sort by appearance position
    hints_with_pos.sort(key=lambda x: x[0])
    return [name for _, name in hints_with_pos]


def _infer_filename_from_code(code: str, index: int, task_id: str = "unknown") -> str:
    """Infer filename from code content when no filename hint is available.

    Strategy:
    1. If code contains if __name__ == '__main__', infer entry_{task_id}.py
    2. If code starts with from/src/import, it's a module, use module_{task_id}_{index}.py
    3. Otherwise use code_{task_id}_{index}.py
    """
    safe_task_id = task_id.replace(".", "_")

    # Check if it's an entry file
    if "__name__" in code and "__main__" in code:
        return f"entry_{safe_task_id}.py"

    # Check if it's an __init__.py (very short and only has imports or is empty)
    stripped = code.strip()
    if not stripped or (len(stripped) < 50 and not any(kw in stripped for kw in ['def ', 'class '])):
        return f"__init__{'_' + safe_task_id if index > 0 else ''}.py"

    # Generic module file — use task_id to distinguish
    return f"module_{safe_task_id}_{index + 1}.py"


def _clean_markdown_artifacts(code: str) -> str:
    """Clean residual markdown markers from code blocks."""
    clean_code = re.sub(r'^```[\w]*\n?', '', code.strip(), flags=re.IGNORECASE)
    clean_code = re.sub(r'\n?```\s*$', '', clean_code)
    return clean_code.strip()


def extract_files_from_response(content: str | None, task_id: str = "unknown") -> dict[str, str]:
    """Extract files from LLM response using 3-level degradation strategy.

    Level 1a-d: XML/fence pattern extraction (strict → lenient → no-quote → fence+path)
    Level 2:   Markdown code blocks with filename hints
    Level 3:   Fallback filename inference from code content

    Args:
        content: Raw LLM response text
        task_id: Current sub-task ID for generating unique fallback filenames

    Returns:
        Mapping of relative file paths to file contents
    """
    if not content:
        return {}
    files = {}

    # 1a-d: Try XML/fence patterns in order; each pattern is a superset
    # of the previous one. We try STRICT first (most precise), but only
    # return early if it captured ALL <file> tags in the content.
    # If a stricter pattern misses some tags, we fall through to the
    # more lenient pattern to catch them.
    all_file_tags = re.findall(r'<file\s', content, re.IGNORECASE)
    total_file_tags = len(all_file_tags)

    # 1a. Strict XML: <file path='...'>...</file> (requires quotes)
    matches = re.findall(FILE_BLOCK_PATTERN_STRICT, content, re.DOTALL | re.IGNORECASE)
    if matches and len(matches) == total_file_tags:
        for path, code in matches:
            files[path.strip()] = _clean_markdown_artifacts(code)
        return files

    # 1b. Lenient XML: supports name attribute, optional quotes
    matches = re.findall(FILE_BLOCK_PATTERN_LENIENT, content, re.DOTALL | re.IGNORECASE)
    if matches and len(matches) == total_file_tags:
        for path, code in matches:
            files[path.strip()] = _clean_markdown_artifacts(code)
        return files

    # 1c. No-quote XML: <file path=foo.py>
    matches = re.findall(FILE_BLOCK_PATTERN_NOQUOTE, content, re.DOTALL | re.IGNORECASE)
    if matches and len(matches) == total_file_tags:
        for path, code in matches:
            files[path.strip()] = _clean_markdown_artifacts(code)
        return files

    # 1d. Triple-backtick with path: ```python:src/main.py
    matches = re.findall(FENCE_PATH_PATTERN, content, re.DOTALL | re.IGNORECASE)
    if matches:
        for path, code in matches:
            files[path.strip()] = code.strip()
        return files

    # If any XML pattern matched but didn't capture all tags, use the
    # best partial result (lenient pattern captures the most variants)
    if total_file_tags > 0:
        matches = re.findall(FILE_BLOCK_PATTERN_LENIENT, content, re.DOTALL | re.IGNORECASE)
        if not matches:
            matches = re.findall(FILE_BLOCK_PATTERN_NOQUOTE, content, re.DOTALL | re.IGNORECASE)
        if matches:
            for path, code in matches:
                files[path.strip()] = _clean_markdown_artifacts(code)
            if len(files) == total_file_tags:
                return files
            # Partial match — fall through to markdown fallback for remaining
            logger.warning(
                f"XML extraction partially succeeded for task {task_id}: "
                f"{len(files)}/{total_file_tags} files extracted, falling back to Markdown"
            )

    # 2. Fallback: Markdown code blocks
    logger.warning(
        f"XML extraction failed for task {task_id}, falling back to Markdown parsing"
    )
    md_matches = re.findall(MD_CODE_PATTERN, content, re.DOTALL)
    if not md_matches:
        return {}

    # Collect all filename hints
    filename_hints = _extract_filename_hints(content)

    if filename_hints and len(filename_hints) >= len(md_matches):
        # Filename hints >= code blocks, map 1:1 by order
        for i, code in enumerate(md_matches):
            if i < len(filename_hints):
                files[filename_hints[i]] = code.strip()
            else:
                files[_infer_filename_from_code(code, i, task_id)] = code.strip()
        return files

    elif filename_hints:
        # Fewer hints than code blocks, smart matching
        hint_idx = 0
        for i, code in enumerate(md_matches):
            matched = False
            while hint_idx < len(filename_hints):
                files[filename_hints[hint_idx]] = code.strip()
                hint_idx += 1
                matched = True
                break

            if not matched:
                files[_infer_filename_from_code(code, i, task_id)] = code.strip()
        return files

    # 3. No filename hints at all — infer from code content (use task_id to distinguish)
    safe_task_id = task_id.replace(".", "_")
    for i, code in enumerate(md_matches):
        filename = _infer_filename_from_code(code, i, task_id)
        # Guard: if inferred filename already exists (extremely rare), append task_id suffix
        if filename in files:
            filename = f"fallback_{safe_task_id}_{i + 1}.py"
            logger.warning(
                f"Duplicate filename detected, using fallback: {filename} for task {task_id}"
            )
        files[filename] = code.strip()

    # Log fallback warning
    logger.warning(
        f"Using fallback filenames for task {task_id}: {list(files.keys())}. "
        f"LLM did not output proper <file path='...'> tags."
    )

    return files
