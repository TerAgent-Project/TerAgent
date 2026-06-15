"""文本处理工具函数"""

import logging

__all__ = [
    "strip_code_block",
]

logger = logging.getLogger(__name__)


def strip_code_block(text: str) -> str:
    """移除 Markdown 代码块包裹

    LLM 有时将输出包裹在 ```language ... ``` 中，
    此函数移除这些包裹，返回纯内容。

    Args:
        text: 可能包含代码块包裹的文本

    Returns:
        移除包裹后的纯文本
    """
    if not text:
        return text
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        # Handle single-line case like ```hello``` or ```json{"key":"val"}```
        if "\n" not in text:
            inner = text[3:-3].strip()
            # Strip language prefix if present (e.g., "json" in ```json{...}```)
            if inner and not inner[0].isspace():
                first_word = inner.split()[0]
                # Check if first word is a language identifier
                if (first_word.isalpha() or
                    (first_word.replace('+', '').replace('#', '').isalpha()
                     and len(first_word) <= 20)):
                    rest = inner[len(first_word):]
                    if rest:
                        return rest.lstrip()
                    return ""
            return inner
        lines = text.split("\n")
        # 移除首行 ```language 和尾行 ```
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text
