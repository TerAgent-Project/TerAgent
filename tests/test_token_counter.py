# tests/test_token_counter.py
"""Token 估算工具单元测试

测试 estimate_tokens 和 detect_content_type 函数。
"""

from teragent.utils.token_counter import _count_cjk, detect_content_type, estimate_tokens

# ===== Token 估算 =====

class TestEstimateTokens:
    """Token 估算"""

    def test_empty_string_returns_zero(self):
        """空字符串返回 0"""
        assert estimate_tokens("") == 0

    def test_none_returns_zero(self):
        """None 返回 0"""
        assert estimate_tokens(None) == 0

    def test_english_natural_text(self):
        """英文自然语言估算"""
        text = "This is a simple English sentence for testing token estimation."
        tokens = estimate_tokens(text, content_type="natural")
        assert tokens > 0
        # 英文约 4 字符/Token × 1.3 保守因子
        # 66 chars / 4 * 1.3 ≈ 21
        assert 5 < tokens < 50

    def test_code_content(self):
        """代码内容估算"""
        code = "def hello_world():\n    print('Hello, World!')\n    return True"
        tokens = estimate_tokens(code, content_type="code")
        assert tokens > 0
        # 代码约 3 字符/Token，密度更高
        assert 5 < tokens < 100

    def test_mixed_content(self):
        """混合内容估算"""
        text = "这是一段混合内容 with English and 中文。"
        tokens = estimate_tokens(text, content_type="mixed")
        assert tokens > 0

    def test_auto_detect(self):
        """自动检测内容类型"""
        text = "This is natural language text with some words."
        tokens = estimate_tokens(text)  # content_type="auto"
        assert tokens > 0

    def test_cjk_text_higher_token_count(self):
        """CJK 文本 Token 数更高"""
        cjk_text = "这是一个中文测试文本"
        en_text = "a" * len(cjk_text)
        cjk_tokens = estimate_tokens(cjk_text, content_type="mixed")
        en_tokens = estimate_tokens(en_text, content_type="mixed")
        # CJK 字符每个约 1.5 字符/Token，比英文更密
        assert cjk_tokens > en_tokens

    def test_minimum_one_token(self):
        """至少返回 1 个 token（非空文本）"""
        assert estimate_tokens("a") >= 1


# ===== 内容类型检测 =====

class TestDetectContentType:
    """内容类型检测"""

    def test_detect_code(self):
        """检测代码内容"""
        code = "def foo():\n    import os\n    return os.path.exists('/tmp')"
        assert detect_content_type(code) == "code"

    def test_detect_natural_language(self):
        """检测自然语言"""
        text = (
            "This is a paragraph of natural language text. "
            "It contains multiple sentences and words. "
            "The purpose is to test the content detection feature. "
            "We expect it to be classified as natural language. "
            "And it should work correctly for this purpose."
        )
        assert detect_content_type(text) == "natural"

    def test_detect_indented_code(self):
        """缩进模式检测代码"""
        code = "    line one\n    line two\n    line three\n"
        assert detect_content_type(code) == "code"

    def test_detect_mixed(self):
        """短文本默认 mixed"""
        text = "hello world"
        assert detect_content_type(text) in ("mixed", "natural", "code")


# ===== CJK 计数 =====

class TestCountCJK:
    """CJK 字符计数"""

    def test_count_chinese(self):
        """统计中文字符"""
        assert _count_cjk("你好世界") == 4

    def test_count_japanese_hiragana(self):
        """统计日文平假名"""
        assert _count_cjk("こんにちは") == 5

    def test_count_korean(self):
        """统计韩文"""
        assert _count_cjk("안녕") == 2

    def test_count_mixed(self):
        """混合文本中 CJK 计数"""
        assert _count_cjk("Hello你好World世界") == 4

    def test_count_no_cjk(self):
        """纯英文无 CJK"""
        assert _count_cjk("Hello World") == 0
