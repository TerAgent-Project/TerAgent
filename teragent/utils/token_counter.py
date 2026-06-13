"""Token 估算工具 — 渐进式改进精度

采用内容类型感知 + CJK 单独计算的策略，不引入 tiktoken 依赖。
估算偏差目标 < 20%（原实现可能 > 50%）。

长期目标：在 agent.toml 中添加 token_counting_method = "tiktoken" | "estimate" 配置。
"""
import re


def estimate_tokens(text: str | None, content_type: str = "auto") -> int:
    """改进的 Token 估算

    Args:
        text: 待估算文本
        content_type: 内容类型
            - "auto": 自动检测（默认）
            - "code": 代码（Token 密度更高，~3 字符/Token）
            - "natural": 英文自然语言（~4 字符/Token）
            - "mixed": 混合内容（~3.5 字符/Token）

    Returns:
        估算的 Token 数
    """
    if not text:
        return 0

    # 自动检测内容类型
    if content_type == "auto":
        content_type = detect_content_type(text)

    # 按内容类型使用不同参数
    chars_per_token = _CHARS_PER_TOKEN.get(content_type, 3.5)

    # CJK 字符单独计算（CJK 约 1.5 字符/Token）
    cjk_count = _count_cjk(text)
    cjk_tokens = cjk_count / 1.5

    non_cjk_len = len(text) - cjk_count
    non_cjk_tokens = non_cjk_len / chars_per_token

    # 保守因子 1.3（宁可高估，避免截断）
    total = (cjk_tokens + non_cjk_tokens) * 1.3

    return max(1, int(total))


# 内容类型 → 字符/Token 比率
_CHARS_PER_TOKEN: dict[str, float] = {
    "code": 3.0,      # 代码 Token 密度更高
    "natural": 4.0,   # 英文自然语言
    "mixed": 3.5,     # 混合内容取中间值
}

# 代码指标模式
_CODE_INDICATORS = (
    r'(?:def |class |import |from \w+ import |'
    r'    [a-z_]\w* = |-> |:\s*\w+(?:\[|,|\))|'
    r'if __name__|#\!|async def |await |try:|except )'
)

# 预编译正则
_CODE_INDICATOR_RE = re.compile(_CODE_INDICATORS)


def detect_content_type(text: str) -> str:
    """简单的内容类型检测

    Args:
        text: 待检测文本

    Returns:
        "code" / "natural" / "mixed"
    """
    # 统计代码指标命中数
    matches = _CODE_INDICATOR_RE.findall(text)
    code_score = len(matches)

    # 短文本阈值更低
    threshold = 2 if len(text) < 500 else 3

    if code_score >= threshold:
        return "code"

    # 检查缩进模式（4空格缩进是强代码信号）
    lines = text.split('\n')
    indented_lines = sum(1 for line in lines if line.startswith('    ') and line.strip())
    if indented_lines >= 3:
        return "code"

    # 检查自然语言信号
    sentence_endings = text.count('.') + text.count('。') + text.count('!') + text.count('？')
    word_count = len(text.split())
    if word_count > 20 and sentence_endings > word_count * 0.05:
        return "natural"

    return "mixed"


def _count_cjk(text: str) -> int:
    """计算 CJK 字符数量（包括扩展区）"""
    count = 0
    for c in text:
        cp = ord(c)
        # CJK Unified Ideographs + Extensions + Compatibility
        if (0x4E00 <= cp <= 0x9FFF or      # CJK Unified
            0x3400 <= cp <= 0x4DBF or       # CJK Extension A
            0x20000 <= cp <= 0x2A6DF or     # CJK Extension B
            0xF900 <= cp <= 0xFAFF or       # CJK Compatibility
            0x2F800 <= cp <= 0x2FA1F or     # CJK Compatibility Supplement
            0x3000 <= cp <= 0x303F or       # CJK Symbols and Punctuation
            0x3040 <= cp <= 0x309F or       # Hiragana
            0x30A0 <= cp <= 0x30FF or       # Katakana
            0xAC00 <= cp <= 0xD7AF):        # Hangul Syllables
            count += 1
    return count
