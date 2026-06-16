# tests/test_config_loader.py
"""配置加载器单元测试

测试 TOML 加载、编译器推断、管道配置等。
（旧格式相关测试已移除 — 不再支持旧格式）
"""


from teragent.config.driver_config import DriverConfig
from teragent.config.loader import (
    _is_new_format,
    get_driver_config,
    infer_compiler,
    load_driver_configs,
    load_pipeline_config,
)


# ===== 编译器自动推断 =====

class TestInferCompiler:
    """编译器自动推断"""

    def test_known_identity(self):
        """已知 identity 直接匹配"""
        assert infer_compiler("glm") == "glm"
        assert infer_compiler("claude") == "anthropic"
        assert infer_compiler("deepseek") == "deepseek"

    def test_partial_match(self):
        """identity 包含已知关键词"""
        assert infer_compiler("glm_5") == "glm_5"
        assert infer_compiler("claude_sonnet") == "anthropic"

    def test_unknown_defaults(self):
        """未知 identity 返回 default"""
        assert infer_compiler("my_custom_model") == "default"

    def test_gpt_defaults(self):
        """gpt 系列返回 default"""
        assert infer_compiler("gpt") == "default"
        assert infer_compiler("gpt4o") == "default"


# ===== 格式检测 =====

class TestFormatDetection:
    """配置格式检测"""

    def test_new_format_detected(self):
        """新格式（嵌套 identity）"""
        drivers = {
            "openai_compatible": {
                "glm": {
                    "model": "glm-5",
                    "base_url": "https://api.example.com",
                    "api_key_env": "GLM_KEY",
                }
            }
        }
        assert _is_new_format(drivers) is True

    def test_old_format_detected(self):
        """旧格式（扁平结构）"""
        drivers = {
            "openai_compatible": {
                "base_url": "https://api.example.com",
                "api_key_env": "GLM_KEY",
                "model": "step-3.5",
            }
        }
        assert _is_new_format(drivers) is False

    def test_empty_drivers(self):
        """空 drivers 段"""
        assert _is_new_format({}) is False


# ===== 加载驱动配置 =====

class TestLoadDriverConfigs:
    """加载驱动配置"""

    def test_load_new_format(self, monkeypatch):
        """加载新格式配置"""
        monkeypatch.setenv("GLM_API_KEY", "test-key-123")

        config = {
            "drivers": {
                "openai_compatible": {
                    "glm": {
                        "model": "glm-5",
                        "base_url": "https://api.example.com",
                        "api_key_env": "GLM_API_KEY",
                        "compiler": "glm",
                    }
                }
            }
        }
        drivers = load_driver_configs(config)
        assert "openai_compatible.glm" in drivers
        dc = drivers["openai_compatible.glm"]
        assert isinstance(dc, DriverConfig)
        assert dc.model == "glm-5"
        assert dc.compiler == "glm"

    def test_load_empty_config(self):
        """空配置返回空字典"""
        config = {}
        drivers = load_driver_configs(config)
        assert drivers == {}

    def test_load_missing_drivers_section(self):
        """缺少 drivers 段返回空字典"""
        config = {"other_section": {}}
        drivers = load_driver_configs(config)
        assert drivers == {}


# ===== 管道配置 =====

class TestPipelineConfig:
    """管道配置加载"""

    def test_load_pipeline_new_format(self):
        """加载新格式管道配置"""
        config = {
            "execution": {
                "pipeline": {
                    "design_driver": "openai_compatible.glm",
                    "plan_driver": "openai_compatible.glm",
                    "execute_driver": "openai_compatible.glm",
                    "review_driver": "openai_compatible.glm",
                }
            }
        }
        pipeline = load_pipeline_config(config)
        assert pipeline["design"] == "openai_compatible.glm"
        assert pipeline["review"] == "openai_compatible.glm"

    def test_load_pipeline_missing(self):
        """缺少管道配置返回空字符串"""
        config = {}
        pipeline = load_pipeline_config(config)
        assert pipeline["design"] == ""
        assert pipeline["review"] == ""


# ===== get_driver_config =====

class TestGetDriverConfig:
    """按名称查找 DriverConfig"""

    def test_lookup_by_full_name(self):
        """全名查找"""
        dc = DriverConfig(adapter="openai_compatible", identity="glm", full_name="openai_compatible.glm")
        drivers = {"openai_compatible.glm": dc}
        result = get_driver_config(drivers, "openai_compatible.glm")
        assert result is dc

    def test_lookup_by_identity(self):
        """短名查找（按 identity）"""
        dc = DriverConfig(adapter="openai_compatible", identity="glm", full_name="openai_compatible.glm")
        drivers = {"openai_compatible.glm": dc}
        result = get_driver_config(drivers, "glm")
        assert result is dc

    def test_lookup_not_found(self):
        """查找不存在的驱动返回 None"""
        drivers = {}
        result = get_driver_config(drivers, "nonexistent")
        assert result is None
