# teragent/context/memory.py
import logging
import os
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def load_agent_md(project_root: str) -> str:
    """读取项目根目录下的 AGENT.md 持久提示词"""
    file_path = os.path.join(project_root, "AGENT.md")
    if not os.path.exists(file_path):
        return ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"Loaded AGENT.md from {file_path}")
        return content
    except Exception as e:
        logger.error(f"Failed to read AGENT.md: {e}")
        return ""


def save_agent_md(project_root: str, content: str) -> bool:
    """Write *content* to the AGENT.md file, replacing any existing content.

    Args:
        project_root: The project root directory containing AGENT.md.
        content: The full text to write.

    Returns:
        True on success, False on failure.
    """
    file_path = os.path.join(project_root, "AGENT.md")
    try:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved AGENT.md to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save AGENT.md: {e}")
        return False


def merge_agent_md(project_root: str, section_title: str, content: str) -> bool:
    """Append a new section to AGENT.md. If the file doesn't exist, create it.

    The new section is appended at the end of the file with a markdown
    heading and a timestamp comment.

    Args:
        project_root: The project root directory containing AGENT.md.
        section_title: The markdown heading for the new section.
        content: The body text to append under the heading.

    Returns:
        True on success, False on failure.
    """
    _file_path = os.path.join(project_root, "AGENT.md")
    try:
        existing = load_agent_md(project_root)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        new_section = (
            f"\n\n## {section_title}\n\n"
            f"<!-- added {timestamp} -->\n\n"
            f"{content}\n"
        )
        return save_agent_md(project_root, existing + new_section)
    except Exception as e:
        logger.error(f"Failed to merge into AGENT.md: {e}")
        return False


def extract_rules(project_root: str) -> list[dict]:
    """Parse structured rules from AGENT.md.

    Looks for blocks with the pattern::

        ### Rule: <name>
        - key: value
        - key: value

    Each rule block is parsed into a dict with at least a ``name`` key,
    plus any ``key: value`` pairs found in the body.

    Args:
        project_root: The project root directory containing AGENT.md.

    Returns:
        A list of rule dicts. Empty list if AGENT.md doesn't exist or
        contains no rule blocks.
    """
    content = load_agent_md(project_root)
    if not content:
        return []

    rules: list[dict] = []
    # Split on ### Rule: headings
    rule_pattern = re.compile(r"^###\s+Rule:\s+(.+)$", re.MULTILINE)
    splits = rule_pattern.split(content)

    # splits: [preamble, name1, body1, name2, body2, ...]
    idx = 1
    while idx + 1 < len(splits):
        rule_name = splits[idx].strip()
        body = splits[idx + 1]
        rule: dict = {"name": rule_name}
        # Parse key: value lines
        for line in body.splitlines():
            line = line.strip()
            kv_match = re.match(r"^-\s+([^:]+):\s+(.+)$", line)
            if kv_match:
                rule[kv_match.group(1).strip()] = kv_match.group(2).strip()
            elif line and not line.startswith("#") and not line.startswith("<!--"):
                # Accumulate description lines
                rule.setdefault("description", "")
                rule["description"] += line + " "
        if "description" in rule and isinstance(rule["description"], str):
            rule["description"] = rule["description"].strip()
        rules.append(rule)
        idx += 2

    logger.info(f"Extracted {len(rules)} rules from AGENT.md")
    return rules
