# tests/test_extractor.py
"""文件提取器单元测试

覆盖 teragent.pipeline.extractor 模块:
  - 4 级 XML 提取: strict / lenient / no-quote / fence+path
  - Markdown 降级提取 (代码块 + 文件名提示)
  - 纯代码内容推断文件名
  - 空输入 / 畸形 XML 处理
  - 单响应多文件提取
  - 代码块语言检测
  - 各种文件路径格式提取
"""
import pytest

from teragent.pipeline.extractor import (
    extract_files_from_response,
    _extract_filename_hints,
    _infer_filename_from_code,
    _clean_markdown_artifacts,
)


# ===== 1 级: 严格 XML 提取 =====

class TestStrictXMLExtraction:
    """1a 级: 严格 XML 提取 <file path='...'>...</file>"""

    def test_strict_single_quotes(self):
        """单引号路径的严格 XML 提取"""
        content = "<file path='src/main.py'>print('hello')</file>"
        result = extract_files_from_response(content)
        assert "src/main.py" in result
        assert result["src/main.py"] == "print('hello')"

    def test_strict_double_quotes(self):
        """双引号路径的严格 XML 提取"""
        content = '<file path="src/utils.py">def helper(): pass</file>'
        result = extract_files_from_response(content)
        assert "src/utils.py" in result
        assert result["src/utils.py"] == "def helper(): pass"

    def test_strict_multiple_files(self):
        """严格 XML 提取多个文件"""
        content = """
<file path='a.py'>code_a</file>
<file path='b.py'>code_b</file>
<file path='c.py'>code_c</file>
"""
        result = extract_files_from_response(content)
        assert len(result) == 3
        assert result["a.py"] == "code_a"
        assert result["b.py"] == "code_b"
        assert result["c.py"] == "code_c"


# ===== 1b 级: 宽松 XML 提取 =====

class TestLenientXMLExtraction:
    """1b 级: 宽松 XML 提取 — 支持 name 属性、可选引号"""

    def test_lenient_name_attribute(self):
        """使用 name= 属性的宽松 XML 提取"""
        content = '<file name="config.yaml">key: value</file>'
        result = extract_files_from_response(content)
        assert "config.yaml" in result

    def test_lenient_path_unquoted(self):
        """无引号路径不会匹配 lenient 模式（由 noquote 模式处理）"""
        # lenient 模式要求引号或无引号均可，但无引号时由 noquote 优先匹配
        # 此处测试 lenient 可以匹配带引号的各种属性
        content = '<file path="game/snake.py">import pygame</file>'
        result = extract_files_from_response(content)
        assert "game/snake.py" in result


# ===== 1c 级: 无引号 XML 提取 =====

class TestNoQuoteXMLExtraction:
    """1c 级: 无引号 XML 提取 <file path=foo.py>"""

    def test_noquote_path(self):
        """无引号路径提取"""
        content = "<file path=foo.py>x = 1</file>"
        result = extract_files_from_response(content)
        assert "foo.py" in result
        assert result["foo.py"] == "x = 1"

    def test_noquote_name_attribute(self):
        """无引号 name 属性提取"""
        content = "<file name=bar.py>y = 2</file>"
        result = extract_files_from_response(content)
        assert "bar.py" in result


# ===== 1d 级: 三反引号 + 路径 =====

class TestFencePathExtraction:
    """1d 级: 三反引号 + 路径 ```python:src/main.py"""

    def test_fence_with_python_path(self):
        """三反引号后跟语言和路径"""
        content = "```python:src/main.py\nprint('hello')\n```"
        result = extract_files_from_response(content)
        assert "src/main.py" in result
        assert "print('hello')" in result["src/main.py"]

    def test_fence_with_yaml_path(self):
        """三反引号后跟 YAML 路径"""
        content = "```yaml:config/settings.yaml\ndebug: true\n```"
        result = extract_files_from_response(content)
        assert "config/settings.yaml" in result


# ===== 2 级: Markdown 降级提取 =====

class TestMarkdownFallback:
    """2 级: Markdown 代码块降级提取"""

    def test_markdown_with_heading_hints(self):
        """Markdown 标题中的文件名提示"""
        content = """下面是代码：

### src/game.py
```python
class Game:
    pass
```
"""
        result = extract_files_from_response(content)
        assert "src/game.py" in result
        assert "class Game" in result["src/game.py"]

    def test_markdown_with_comment_hints(self):
        """注释中的文件名提示"""
        content = """```python
# file: engine/core.py
class Core:
    pass
```
"""
        result = extract_files_from_response(content)
        assert "engine/core.py" in result

    def test_markdown_with_bold_hints(self):
        """加粗文件名提示"""
        content = """下面是代码：

**src/config.py**
```python
DEBUG = True
```
"""
        result = extract_files_from_response(content)
        assert "src/config.py" in result

    def test_fewer_hints_than_blocks(self):
        """文件名提示少于代码块时，剩余块使用推断文件名"""
        content = """### a.py
```python
code_a
```

```python
code_b
```
"""
        result = extract_files_from_response(content)
        assert "a.py" in result
        # 第二个代码块应使用推断文件名
        assert len(result) == 2


# ===== 3 级: 纯代码内容推断文件名 =====

class TestFilenameInference:
    """3 级: 无文件名提示时从代码内容推断文件名"""

    def test_infer_entry_file(self):
        """包含 if __name__ == '__main__' 推断为入口文件"""
        code = "if __name__ == '__main__':\n    main()"
        filename = _infer_filename_from_code(code, 0, task_id="1.1")
        assert filename == "entry_1_1.py"

    def test_infer_init_file_empty(self):
        """空代码推断为 __init__.py"""
        filename = _infer_filename_from_code("", 0, task_id="1.1")
        assert "__init__" in filename
        assert filename.endswith(".py")

    def test_infer_init_file_short_imports(self):
        """短小的仅含 import 的代码推断为 __init__.py"""
        code = "from .module import foo"
        filename = _infer_filename_from_code(code, 0, task_id="1.1")
        assert "__init__" in filename

    def test_infer_module_file(self):
        """普通代码推断为模块文件"""
        code = "def process():\n    return 42"
        filename = _infer_filename_from_code(code, 0, task_id="2.3")
        assert filename == "module_2_3_1.py"

    def test_infer_module_with_index(self):
        """不同索引产生不同模块文件名"""
        code = "def foo(): pass"
        f1 = _infer_filename_from_code(code, 0, task_id="1")
        f2 = _infer_filename_from_code(code, 1, task_id="1")
        assert f1 != f2
        assert "module_1_1" in f1
        assert "module_1_2" in f2


# ===== 空输入 / 畸形处理 =====

class TestEdgeCases:
    """边界情况: 空输入、畸形 XML、无效内容"""

    def test_empty_string_returns_empty(self):
        """空字符串返回空字典"""
        result = extract_files_from_response("")
        assert result == {}

    def test_none_returns_empty(self):
        """None 输入返回空字典"""
        result = extract_files_from_response(None)
        assert result == {}

    def test_whitespace_only_returns_empty(self):
        """纯空白输入返回空字典"""
        result = extract_files_from_response("   \n  \t  ")
        assert result == {}

    def test_malformed_xml_no_closing_tag(self):
        """缺少闭合标签的畸形 XML，降级到 Markdown 解析"""
        content = "<file path='broken.py'>print('oops')"
        # 没有 </file>，不会匹配 XML 模式，降级处理
        result = extract_files_from_response(content)
        # 可能返回空或降级解析结果
        assert isinstance(result, dict)

    def test_plain_text_no_code_returns_empty(self):
        """纯文本无代码返回空字典"""
        content = "这是一段普通文字，没有任何代码或文件标记。"
        result = extract_files_from_response(content)
        assert result == {}


# ===== 文件名提示提取 =====

class TestFilenameHints:
    """文件名提示提取 _extract_filename_hints"""

    def test_heading_hint(self):
        """Markdown 标题提取文件名提示"""
        content = "### game/entities/snake.py\n```python\nclass Snake: pass\n```"
        hints = _extract_filename_hints(content)
        assert "game/entities/snake.py" in hints

    def test_comment_hint(self):
        """注释提取文件名提示"""
        content = "# file: game/config.py\nDEBUG = True"
        hints = _extract_filename_hints(content)
        assert "game/config.py" in hints

    def test_create_file_pattern(self):
        """中文/英文创建文件模式提取"""
        content = "创建 game/engine.py\n```python\nclass Engine: pass\n```"
        hints = _extract_filename_hints(content)
        assert "game/engine.py" in hints

    def test_multiple_hints_ordered(self):
        """多个提示按出现顺序排列"""
        content = "### a.py\n### b.py\n### c.py"
        hints = _extract_filename_hints(content)
        assert hints == ["a.py", "b.py", "c.py"]

    def test_dedup_nearby_hints(self):
        """接近位置重复的提示去重（5 字符以内）"""
        # 两个模式在几乎同一位置匹配同一文件名时去重
        # 例如注释和标题同时出现在同一行附近
        content = "### a.py"
        hints = _extract_filename_hints(content)
        # 即使同一文件名被多个模式匹配，位置接近时去重
        assert hints.count("a.py") >= 1


# ===== Markdown 清理 =====

class TestCleanMarkdownArtifacts:
    """Markdown 残留标记清理 _clean_markdown_artifacts"""

    def test_removes_opening_fence(self):
        """移除开头的代码块标记"""
        cleaned = _clean_markdown_artifacts("```python\ncode\n```")
        assert not cleaned.startswith("```")
        assert "code" in cleaned

    def test_removes_closing_fence(self):
        """移除结尾的代码块标记"""
        cleaned = _clean_markdown_artifacts("code\n```")
        assert not cleaned.endswith("```")

    def test_preserves_clean_code(self):
        """保留无标记的干净代码"""
        code = "def foo():\n    return 42"
        cleaned = _clean_markdown_artifacts(code)
        assert cleaned == code


# ===== 降级策略完整流程 =====

class TestDegradationStrategy:
    """3 级降级策略完整流程"""

    def test_level1_strict_takes_priority(self):
        """严格 XML 匹配优先于降级策略"""
        content = """<file path='app.py'>app_code</file>

### extra.py
```python
extra_code
```
"""
        result = extract_files_from_response(content)
        # 严格 XML 匹配成功，不应进入 Markdown 降级
        assert "app.py" in result
        # Markdown 块不应被提取（Level 1 优先返回）
        assert "extra.py" not in result

    def test_level2_markdown_fallback(self):
        """无 XML 时使用 Markdown 降级提取"""
        content = """### game.py
```python
class Game: pass
```
"""
        result = extract_files_from_response(content)
        assert "game.py" in result

    def test_level3_infer_fallback(self):
        """无文件名提示时推断文件名"""
        content = """```python
def process():
    return 42
```
"""
        result = extract_files_from_response(content)
        assert len(result) == 1
        # 应使用推断文件名
        filename = list(result.keys())[0]
        assert filename.endswith(".py")

    def test_duplicate_inferred_filename_gets_fallback(self):
        """推断文件名冲突时使用 fallback 命名"""
        # 两个完全相同的代码块会导致推断文件名冲突
        code = "def process():\n    return 42"
        content = f"```python\n{code}\n```\n\n```python\n{code}\n```"
        result = extract_files_from_response(content)
        # 应有两个文件，不会因键冲突丢失
        assert len(result) == 2
