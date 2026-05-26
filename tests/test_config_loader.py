# tests/test_config_loader.py
"""配置加载器单元测试

测试 TOML 加载、新旧格式、编译器推断、管道配置等。
"""
import os
import pytest

from teragent.config.loader import (
    load_driver_configs,
    load_pipeline_config,
    infer_compiler,
    _is_new_format,
    _parse_old_driver_name,
    get_driver_config,
)
from teragent.config.driver_config import DriverConfig


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
        assert infer_compiler("glm_5") == "glm"
        assert infer_compiler("claude_sonnet") == "anthropic"

    def test_unknown_defaults(self):
        """未知 identity 返回 default"""
        assert infer_compiler("my_custom_model") == "default"

    def test_gpt_defaults(self):
        """gpt 系列返回 default"""
        assert infer_compiler("gpt") == "default"
        assert infer_compiler("gpt4o") == "default"


# ===== 旧格式驱动名解析 =====

class TestParseOldDriverName:
    """旧格式驱动名解析"""

    def test_known_old_names(self):
        """已知旧格式名"""
        adapter, identity = _parse_old_driver_name("openai_compatible")
        assert adapter == "openai_compatible"
        assert identity == "default"

    def test_glm_driver(self):
        """glm 驱动名"""
        adapter, identity = _parse_old_driver_name("glm")
        assert adapter == "openai_compatible"
        assert identity == "glm"

    def test_unknown_driver(self):
        """未知驱动名默认 openai_compatible"""
        adapter, identity = _parse_old_driver_name("custom_adapter")
        assert adapter == "openai_compatible"
        assert identity == "custom_adapter"


# ===== 新旧格式检测 =====

class TestFormatDetection:
    """配置格式检测"""

    def test_new_format_detected(self):
        """新格式（嵌套 identity）"""
        drivers = {
            "openai_compatible": {
                "glm": {
                    "model": "glm-5.1",
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
                "api_key": "test-key",
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
        # 设置环境变量以提供 API key
        monkeypatch.setenv("GLM_API_KEY", "test-key-123")

        config = {
            "drivers": {
                "openai_compatible": {
                    "glm": {
                        "model": "glm-5.1",
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
        assert dc.model == "glm-5.1"
        assert dc.compiler == "glm"

    def test_load_old_format(self):
        """加载旧格式配置"""
        config = {
            "drivers": {
                "openai_compatible": {
                    "base_url": "https://api.example.com",
                    "api_key": "test-key",
                    "model": "step-3.5",
                }
            }
        }
        drivers = load_driver_configs(config)
        assert "openai_compatible" in drivers
        dc = drivers["openai_compatible"]
        assert dc.model == "step-3.5"
        # 旧格式自动推断 compiler
        assert dc.compiler != ""

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

    def test_load_pipeline_old_format(self):
        """加载旧格式管道配置"""
        config = {
            "execution": {
                "pipeline": {
                    "design_model": "openai_compatible",
                    "plan_model": "openai_compatible",
                    "execute_model": "openai_compatible",
                    "review_model": "openai_compatible",
                }
            }
        }
        pipeline = load_pipeline_config(config)
        assert pipeline["design"] == "openai_compatible"

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
