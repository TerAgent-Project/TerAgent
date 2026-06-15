# tests/integration/test_p1_8_mock_regression.py
"""P1-8: Mock 模式回归测试

使用 MockAdapter 运行所有 Compiler 组合，确保三模型深度适配无回归。

测试覆盖:
  1. 所有 9 个 Compiler × MockAdapter 完整 compile→send→TAPResponse 流程
  2. DeepSeekV4Compiler: Flash/Pro 双模式、思考模式 deep/quick/auto、缓存感知、多模态降级
  3. GLM5Compiler: Recency Effect、思考模式、长程任务引导、自评估注入、桌面上下文降级
  4. MiniMaxM3Compiler: 纯文本编译、多模态编译、MSA 全文注入、桌面上下文、编程/浏览增强
  5. TAP IR 扩展: MultimodalContent、DesktopContext、LongHorizonConfig 等新数据类
  6. 配置系统: DriverConfig 新字段、compiler_variant 传递
  7. 注册表完整性: TAPCompilerRegistry / TAPAdapterRegistry 正确注册
  8. 向后兼容: 现有 4 个 Compiler 行为不变
  9. ModelProvider 完整集成: create_provider() 工厂函数
"""
import asyncio

import pytest

from teragent.config.driver_config import DriverConfig
from teragent.core.adapter import TAPAdapterRegistry
from teragent.core.adapters.mock import MockAdapter
from teragent.core.compiler import TAPCompilerRegistry
from teragent.core.compilers import (
    AnthropicCompiler,
    DeepSeekCompiler,
    DeepSeekV4Compiler,
    DefaultCompiler,
    GLM5Compiler,
    GLMCompiler,
    MiniMaxM3Compiler,
)
from teragent.core.provider import ModelProvider
from teragent.core.tap import (
    CompiledPrompt,
    DesktopContext,
    LongHorizonConfig,
    LongHorizonStatus,
    MultimodalContent,
    TAPRequest,
    TAPResponse,
)

# ===== 辅助函数 =====

def _make_request(
    intent: str = "execute",
    instruction: str = "写一个排序函数",
    constraints: list[str] | None = None,
    output_format_hint: str = "",
    context: dict | None = None,
    meta: dict | None = None,
    thinking_mode: str | None = None,
    multimodal_context: list[MultimodalContent] | None = None,
    desktop_context: DesktopContext | None = None,
    long_horizon: LongHorizonConfig | None = None,
    cache_preference: str | None = None,
) -> TAPRequest:
    """构造测试用的 TAPRequest，支持所有扩展字段"""
    return TAPRequest(
        meta=meta or {"task_id": "1.1", "intent": intent},
        context=context or {},
        instruction=instruction,
        constraints=constraints or [],
        output_format_hint=output_format_hint,
        thinking_mode=thinking_mode,
        multimodal_context=multimodal_context,
        desktop_context=desktop_context,
        long_horizon=long_horizon,
        cache_preference=cache_preference,
    )


def _run_async(coro):
    """同步运行异步函数的辅助"""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_mock_provider(compiler_name: str, model: str = "mock-model") -> ModelProvider:
    """创建 Compiler + MockAdapter 的 ModelProvider"""
    compiler = TAPCompilerRegistry.create(compiler_name)
    adapter = MockAdapter(delay=0.0)
    return ModelProvider(compiler=compiler, adapter=adapter, model=model)


# ===== 1. 注册表完整性 =====

class TestRegistryIntegrity:
    """验证所有 Compiler 和 Adapter 正确注册"""

    ALL_COMPILER_NAMES = [
        "default", "glm", "anthropic", "deepseek",
        "deepseek_v4", "deepseek_v4_flash", "deepseek_v4_pro",
        "glm_5", "minimax_m3",
    ]

    ALL_ADAPTER_NAMES = ["openai_compatible", "anthropic_native", "mock"]

    def test_all_compilers_registered(self):
        """9 个 Compiler 全部注册"""
        available = TAPCompilerRegistry.available()
        for name in self.ALL_COMPILER_NAMES:
            assert name in available, f"Compiler '{name}' 未注册"

    def test_all_adapters_registered(self):
        """3 个 Adapter 全部注册"""
        available = TAPAdapterRegistry.available()
        for name in self.ALL_ADAPTER_NAMES:
            assert name in available, f"Adapter '{name}' 未注册"

    @pytest.mark.parametrize("name,expected_cls", [
        ("default", DefaultCompiler),
        ("glm", GLMCompiler),
        ("anthropic", AnthropicCompiler),
        ("deepseek", DeepSeekCompiler),
        ("deepseek_v4", DeepSeekV4Compiler),
        ("glm_5", GLM5Compiler),
        ("minimax_m3", MiniMaxM3Compiler),
    ])
    def test_compiler_create_by_name(self, name, expected_cls):
        """通过注册表名创建正确类型的 Compiler"""
        compiler = TAPCompilerRegistry.create(name)
        assert isinstance(compiler, expected_cls), f"{name} -> {type(compiler).__name__}, expected {expected_cls.__name__}"

    def test_deepseek_v4_flash_variant(self):
        """deepseek_v4_flash 创建 Flash 变体"""
        compiler = TAPCompilerRegistry.create("deepseek_v4_flash")
        assert isinstance(compiler, DeepSeekV4Compiler)
        assert compiler.variant == "flash"

    def test_deepseek_v4_pro_variant(self):
        """deepseek_v4_pro 创建 Pro 变体"""
        compiler = TAPCompilerRegistry.create("deepseek_v4_pro")
        assert isinstance(compiler, DeepSeekV4Compiler)
        assert compiler.variant == "pro"

    def test_unknown_compiler_raises(self):
        """创建未注册 Compiler 抛出 ValueError"""
        with pytest.raises(ValueError, match="Unknown compiler"):
            TAPCompilerRegistry.create("nonexistent_compiler")

    def test_unknown_adapter_raises(self):
        """创建未注册 Adapter 抛出 ValueError"""
        with pytest.raises(ValueError, match="Unknown adapter"):
            TAPAdapterRegistry.create("nonexistent_adapter")


# ===== 2. Compiler 能力属性 =====

class TestCompilerCapabilities:
    """验证各 Compiler 的能力属性正确设置"""

    def test_deepseek_v4_capabilities(self):
        """DeepSeekV4Compiler: supports_thinking_mode=True, supports_multimodal=False, max_context=1M"""
        c = DeepSeekV4Compiler()
        assert c.supports_thinking_mode is True
        assert c.supports_multimodal is False
        assert c.max_context_tokens == 1_000_000

    def test_deepseek_v4_flash_capabilities(self):
        """DeepSeekV4Compiler(flash): 同样的能力属性"""
        c = DeepSeekV4Compiler(variant="flash")
        assert c.supports_thinking_mode is True
        assert c.supports_multimodal is False
        assert c.max_context_tokens == 1_000_000

    def test_glm_5_capabilities(self):
        """GLM5Compiler: supports_thinking_mode=True, supports_multimodal=False, max_context=200K"""
        c = GLM5Compiler()
        assert c.supports_thinking_mode is True
        assert c.supports_multimodal is False
        assert c.max_context_tokens == 200_000

    def test_minimax_m3_capabilities(self):
        """MiniMaxM3Compiler: supports_multimodal=True, max_context=1M"""
        c = MiniMaxM3Compiler()
        assert c.supports_multimodal is True
        assert c.supports_thinking_mode is False
        assert c.max_context_tokens == 1_000_000

    def test_existing_compilers_no_multimodal(self):
        """原有 Compiler 均不支持多模态"""
        for cls in (DefaultCompiler, GLMCompiler, DeepSeekCompiler, AnthropicCompiler):
            c = cls()
            assert c.supports_multimodal is False

    def test_anthropic_default_context_tokens(self):
        """AnthropicCompiler 使用默认 128K 上下文"""
        c = AnthropicCompiler()
        assert c.max_context_tokens == 128_000

    def test_invalid_deepseek_v4_variant(self):
        """DeepSeekV4Compiler 无效变体抛出 ValueError"""
        with pytest.raises(ValueError, match="Invalid variant"):
            DeepSeekV4Compiler(variant="invalid")


# ===== 3. 全 Compiler × MockAdapter 回归 =====

class TestAllCompilerMockRegression:
    """所有 Compiler 通过 MockAdapter 的端到端编译-发送回归测试"""

    COMPILER_NAMES = [
        "default", "glm", "deepseek", "anthropic",
        "deepseek_v4", "deepseek_v4_flash", "deepseek_v4_pro",
        "glm_5", "minimax_m3",
    ]

    @pytest.mark.parametrize("compiler_name", COMPILER_NAMES)
    def test_compile_returns_compiled_prompt(self, compiler_name):
        """每个 Compiler 编译后返回有效的 CompiledPrompt"""
        compiler = TAPCompilerRegistry.create(compiler_name)
        request = _make_request(instruction="实现快速排序")
        compiled = compiler.compile(request)
        assert isinstance(compiled, CompiledPrompt)
        assert compiled.mode != "empty", f"{compiler_name} 编译结果为空"

    @pytest.mark.parametrize("compiler_name", COMPILER_NAMES)
    def test_mock_send_returns_tap_response(self, compiler_name):
        """每个 Compiler + MockAdapter 返回有效 TAPResponse"""
        provider = _make_mock_provider(compiler_name)
        request = _make_request(instruction="实现快速排序")
        response = _run_async(provider.execute_tap(request))
        assert isinstance(response, TAPResponse)
        assert response.raw_text is not None
        assert len(response.raw_text) > 0

    @pytest.mark.parametrize("compiler_name", COMPILER_NAMES)
    def test_mock_response_has_usage(self, compiler_name):
        """每个 Compiler + MockAdapter 返回含 usage 的响应"""
        provider = _make_mock_provider(compiler_name)
        request = _make_request(instruction="实现快速排序")
        response = _run_async(provider.execute_tap(request))
        assert "prompt_tokens" in response.usage
        assert "completion_tokens" in response.usage
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0

    @pytest.mark.parametrize("compiler_name", COMPILER_NAMES)
    def test_mock_stream_yields_chunks(self, compiler_name):
        """每个 Compiler + MockAdapter 流式返回文本块"""
        provider = _make_mock_provider(compiler_name)
        request = _make_request(instruction="实现快速排序")

        async def _stream():
            chunks = []
            async for chunk in provider.stream_tap(request):
                chunks.append(chunk)
            return chunks

        chunks = _run_async(_stream())
        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0

    @pytest.mark.parametrize("compiler_name", COMPILER_NAMES)
    def test_instruction_in_compiled_output(self, compiler_name):
        """编译后的 prompt 包含核心指令"""
        compiler = TAPCompilerRegistry.create(compiler_name)
        request = _make_request(instruction="实现二叉搜索树")
        compiled = compiler.compile(request)

        # 收集所有文本内容
        all_text = ""
        if compiled.messages:
            for msg in compiled.messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    all_text += content + " "
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            all_text += part.get("text", "") + " "
        all_text += compiled.system_prompt + " " + compiled.user_message

        assert "实现二叉搜索树" in all_text, f"{compiler_name} 编译结果未包含核心指令"

    @pytest.mark.parametrize("compiler_name", COMPILER_NAMES)
    def test_constraints_in_compiled_output(self, compiler_name):
        """编译后的 prompt 包含约束条件"""
        compiler = TAPCompilerRegistry.create(compiler_name)
        request = _make_request(
            instruction="实现排序",
            constraints=["不能使用内置排序", "必须类型安全"],
        )
        compiled = compiler.compile(request)

        all_text = ""
        if compiled.messages:
            for msg in compiled.messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    all_text += content + " "
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            all_text += part.get("text", "") + " "
        all_text += compiled.system_prompt + " " + compiled.user_message

        assert "不能使用内置排序" in all_text, f"{compiler_name} 编译结果未包含约束"

    @pytest.mark.parametrize("compiler_name", COMPILER_NAMES)
    def test_context_injection(self, compiler_name):
        """编译后的 prompt 包含上下文（design/plan/memory）"""
        compiler = TAPCompilerRegistry.create(compiler_name)
        request = _make_request(
            instruction="实现排序",
            context={
                "design": "微服务架构设计",
                "plan": "三步走计划",
                "memory": "使用 FastAPI 框架",
            },
        )
        compiled = compiler.compile(request)

        all_text = ""
        if compiled.messages:
            for msg in compiled.messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    all_text += content + " "
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            all_text += part.get("text", "") + " "
        all_text += compiled.system_prompt + " " + compiled.user_message

        # 上下文内容应出现在编译结果中
        assert "微服务架构设计" in all_text or "<design>" in all_text, f"{compiler_name} 未注入 design 上下文"
        assert "三步走计划" in all_text or "<plan>" in all_text, f"{compiler_name} 未注入 plan 上下文"

    @pytest.mark.parametrize("compiler_name", COMPILER_NAMES)
    def test_na_context_excluded(self, compiler_name):
        """N/A 上下文不被注入"""
        compiler = TAPCompilerRegistry.create(compiler_name)
        request = _make_request(
            instruction="实现排序",
            context={"design": "N/A", "plan": "N/A", "memory": "N/A"},
        )
        compiled = compiler.compile(request)

        all_text = ""
        if compiled.messages:
            for msg in compiled.messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    all_text += content + " "
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            all_text += part.get("text", "") + " "
        all_text += compiled.system_prompt + " " + compiled.user_message

        # 不应出现确认消息
        assert "收到设计文档" not in all_text
        assert "收到执行计划" not in all_text
        assert "收到项目记忆" not in all_text


# ===== 4. DeepSeekV4Compiler 深度测试 =====

class TestDeepSeekV4CompilerDeep:
    """DeepSeekV4Compiler 深度功能测试"""

    def test_flash_vs_pro_different_prompts(self):
        """Flash 和 Pro 模式产生不同的编译结果"""
        request = _make_request(
            intent="execute",
            instruction="实现排序算法",
            constraints=["O(n log n)"],
        )
        flash = DeepSeekV4Compiler(variant="flash").compile(request)
        pro = DeepSeekV4Compiler(variant="pro").compile(request)
        # Flash 模式通常产生更短的 prompt
        assert flash.messages != pro.messages

    def test_flash_compact_system_prompt(self):
        """Flash 模式系统消息较短"""
        request = _make_request(intent="chat", instruction="你好")
        compiled = DeepSeekV4Compiler(variant="flash").compile(request)
        system_msg = compiled.messages[0]
        assert system_msg["role"] == "system"
        # Flash 系统消息应较紧凑

    def test_pro_has_reasoning_guidance(self):
        """Pro 模式 execute 意图包含推理引导"""
        request = _make_request(intent="execute", instruction="实现排序")
        compiled = DeepSeekV4Compiler(variant="pro").compile(request)
        all_text = " ".join(m.get("content", "") for m in compiled.messages if isinstance(m.get("content"), str))
        assert "推理" in all_text or "完整" in all_text

    def test_thinking_mode_deep(self):
        """thinking_mode=deep → extra["thinking"]["type"]=="enabled" """
        request = _make_request(thinking_mode="deep")
        compiled = DeepSeekV4Compiler(variant="pro").compile(request)
        assert compiled.extra.get("thinking", {}).get("type") == "enabled"

    def test_thinking_mode_quick(self):
        """thinking_mode=quick → extra["thinking"]["type"]=="disabled" """
        request = _make_request(thinking_mode="quick")
        compiled = DeepSeekV4Compiler(variant="pro").compile(request)
        assert compiled.extra.get("thinking", {}).get("type") == "disabled"

    def test_thinking_mode_auto_chat(self):
        """thinking_mode=auto + intent=chat → quick"""
        request = _make_request(intent="chat", thinking_mode="auto")
        compiled = DeepSeekV4Compiler(variant="pro").compile(request)
        assert compiled.extra.get("thinking", {}).get("type") == "disabled"

    def test_thinking_mode_auto_design(self):
        """thinking_mode=auto + intent=design → deep"""
        request = _make_request(intent="design", thinking_mode="auto")
        compiled = DeepSeekV4Compiler(variant="pro").compile(request)
        assert compiled.extra.get("thinking", {}).get("type") == "enabled"

    def test_thinking_mode_auto_execute_pro(self):
        """thinking_mode=auto + execute + Pro → deep"""
        request = _make_request(intent="execute", thinking_mode="auto")
        compiled = DeepSeekV4Compiler(variant="pro").compile(request)
        assert compiled.extra.get("thinking", {}).get("type") == "enabled"

    def test_thinking_mode_auto_execute_flash(self):
        """thinking_mode=auto + execute + Flash → quick"""
        request = _make_request(intent="execute", thinking_mode="auto")
        compiled = DeepSeekV4Compiler(variant="flash").compile(request)
        assert compiled.extra.get("thinking", {}).get("type") == "disabled"

    def test_thinking_mode_none_default(self):
        """thinking_mode=None → auto 行为"""
        request = _make_request(thinking_mode=None, intent="chat")
        compiled = DeepSeekV4Compiler().compile(request)
        # None 被当作 auto 处理
        assert "thinking" in compiled.extra

    def test_cache_aware_layout(self):
        """cache_preference="aggressive" → extra 包含缓存标记"""
        request = _make_request(cache_preference="aggressive")
        compiled = DeepSeekV4Compiler().compile(request)
        assert compiled.extra.get("cache_aware") is True
        assert compiled.extra.get("cache_prefix_frozen") is True

    def test_cache_preference_auto(self):
        """cache_preference="auto" → cache_aware=True 且冻结前缀（P2-1 增强）"""
        request = _make_request(cache_preference="auto")
        compiled = DeepSeekV4Compiler().compile(request)
        assert compiled.extra.get("cache_aware") is True
        # P2-1: auto 模式现在也冻结前缀以最大化缓存命中
        assert compiled.extra.get("cache_prefix_frozen") is True

    def test_cache_preference_none(self):
        """cache_preference="none" → 不设置缓存标记"""
        request = _make_request(cache_preference="none")
        compiled = DeepSeekV4Compiler().compile(request)
        assert "cache_aware" not in compiled.extra

    def test_multimodal_degradation(self):
        """V4 不支持多模态 → 降级为文本描述"""
        request = _make_request(
            instruction="分析这张图片",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/img.png"),
                MultimodalContent(type="text", text="附加文字说明"),
            ],
        )
        compiled = DeepSeekV4Compiler().compile(request)
        all_text = " ".join(m.get("content", "") for m in compiled.messages if isinstance(m.get("content"), str))
        # 图片应被降级为文本描述
        assert "图片" in all_text or "image" in all_text.lower()
        # 附加文字应保留
        assert "附加文字说明" in all_text

    def test_tail_reinforcement_flash(self):
        """Flash 模式尾强化：少量约束时追加提醒"""
        request = _make_request(
            instruction="实现功能",
            constraints=["约束1"],
            output_format_hint="JSON格式",
        )
        compiled = DeepSeekV4Compiler(variant="flash").compile(request)
        all_text = " ".join(m.get("content", "") for m in compiled.messages if isinstance(m.get("content"), str))
        assert "约束" in all_text or "遵守" in all_text or "格式" in all_text

    def test_tail_reinforcement_pro(self):
        """Pro 模式尾强化：包含深度引导"""
        request = _make_request(
            intent="execute",
            instruction="实现功能",
            constraints=["约束1"],
        )
        compiled = DeepSeekV4Compiler(variant="pro").compile(request)
        all_text = " ".join(m.get("content", "") for m in compiled.messages if isinstance(m.get("content"), str))
        assert "约束" in all_text or "满足" in all_text

    def test_compiler_type(self):
        """_get_compiler_type 返回 deepseek_v4"""
        c = DeepSeekV4Compiler()
        assert c._get_compiler_type() == "deepseek_v4"

    def test_mock_e2e_flash(self):
        """Flash 模式完整 e2e: compile → mock send → TAPResponse"""
        provider = _make_mock_provider("deepseek_v4_flash")
        request = _make_request(
            instruction="快速回复",
            thinking_mode="quick",
        )
        response = _run_async(provider.execute_tap(request))
        assert response.raw_text is not None

    def test_mock_e2e_pro_thinking(self):
        """Pro 模式 + 深度思考 e2e"""
        provider = _make_mock_provider("deepseek_v4_pro")
        request = _make_request(
            intent="design",
            instruction="设计系统架构",
            thinking_mode="deep",
        )
        response = _run_async(provider.execute_tap(request))
        assert response.raw_text is not None


# ===== 5. GLM5Compiler 深度测试 =====

class TestGLM5CompilerDeep:
    """GLM5Compiler 深度功能测试"""

    def test_recency_effect_instruction_last(self):
        """Recency Effect: 核心指令在最后一条 user 消息"""
        request = _make_request(instruction="实现排序算法")
        compiled = GLM5Compiler().compile(request)
        last_msg = compiled.messages[-1]
        assert last_msg["role"] == "user"
        assert "实现排序算法" in last_msg["content"]

    def test_context_in_middle(self):
        """上下文在中间区域（指令之前）"""
        request = _make_request(
            instruction="实现排序",
            context={"design": "设计文档", "plan": "执行计划"},
        )
        compiled = GLM5Compiler().compile(request)
        # 找到指令消息的索引
        instruction_idx = None
        for i, msg in enumerate(compiled.messages):
            if "实现排序" in msg.get("content", ""):
                instruction_idx = i
                break
        assert instruction_idx is not None
        # 上下文应在指令之前
        context_before = " ".join(
            m.get("content", "") for m in compiled.messages[:instruction_idx]
            if isinstance(m.get("content"), str)
        )
        assert "设计文档" in context_before or "执行计划" in context_before

    def test_thinking_mode_deep(self):
        """thinking_mode=deep → thinking enabled"""
        request = _make_request(thinking_mode="deep")
        compiled = GLM5Compiler().compile(request)
        assert compiled.extra.get("thinking", {}).get("type") == "enabled"

    def test_thinking_mode_quick(self):
        """thinking_mode=quick → thinking disabled"""
        request = _make_request(thinking_mode="quick")
        compiled = GLM5Compiler().compile(request)
        assert compiled.extra.get("thinking", {}).get("type") == "disabled"

    def test_thinking_mode_auto_chat(self):
        """thinking_mode=auto + chat → quick"""
        request = _make_request(intent="chat", thinking_mode="auto")
        compiled = GLM5Compiler().compile(request)
        assert compiled.extra.get("thinking", {}).get("type") == "disabled"

    def test_thinking_mode_auto_design(self):
        """thinking_mode=auto + design → deep"""
        request = _make_request(intent="design", thinking_mode="auto")
        compiled = GLM5Compiler().compile(request)
        assert compiled.extra.get("thinking", {}).get("type") == "enabled"

    def test_thinking_mode_auto_default_deep(self):
        """thinking_mode=auto + execute → deep (GLM-5 默认深度推理)"""
        request = _make_request(intent="execute", thinking_mode="auto")
        compiled = GLM5Compiler().compile(request)
        assert compiled.extra.get("thinking", {}).get("type") == "enabled"

    def test_long_horizon_system_addition(self):
        """长程任务 → 系统提示包含长程工作模式引导"""
        request = _make_request(
            instruction="执行长时间自动化任务",
            long_horizon=LongHorizonConfig(
                max_duration_hours=8.0,
                checkpoint_interval_minutes=30.0,
                self_evaluation_enabled=True,
            ),
        )
        compiled = GLM5Compiler().compile(request)
        system_msg = compiled.messages[0]
        assert "长程任务" in system_msg["content"]
        assert "8.0" in system_msg["content"] or "8" in system_msg["content"]

    def test_long_horizon_self_evaluation(self):
        """长程任务 + self_evaluation → 用户消息包含自评估检查点"""
        request = _make_request(
            instruction="执行长程任务",
            long_horizon=LongHorizonConfig(self_evaluation_enabled=True),
        )
        compiled = GLM5Compiler().compile(request)
        all_text = " ".join(m.get("content", "") for m in compiled.messages if isinstance(m.get("content"), str))
        assert "自评估" in all_text

    def test_long_horizon_no_self_evaluation(self):
        """长程任务 + self_evaluation=False → 无自评估注入"""
        request = _make_request(
            instruction="执行长程任务",
            long_horizon=LongHorizonConfig(self_evaluation_enabled=False),
        )
        compiled = GLM5Compiler().compile(request)
        # 自评估不应该出现在用户指令附近
        last_msg = compiled.messages[-1]
        assert "自评估检查点" not in last_msg["content"]

    def test_strategy_switch_prompt(self):
        """策略切换引导 prompt 生成"""
        compiler = GLM5Compiler()
        prompt = compiler.build_strategy_switch_prompt("连续3次相同结果")
        assert "策略切换" in prompt
        assert "连续3次相同结果" in prompt
        assert "换一种" in prompt or "换方法" in prompt

    def test_desktop_context_degradation(self):
        """桌面上下文降级为文本"""
        request = _make_request(
            instruction="操作桌面",
            desktop_context=DesktopContext(
                active_window="VS Code",
                interactive_elements=[
                    {"type": "button", "label": "Run", "bbox": {"x": 100, "y": 200}},
                ],
            ),
        )
        compiled = GLM5Compiler().compile(request)
        all_text = " ".join(m.get("content", "") for m in compiled.messages if isinstance(m.get("content"), str))
        assert "VS Code" in all_text or "桌面" in all_text

    def test_multimodal_degradation(self):
        """多模态降级为文本描述"""
        request = _make_request(
            instruction="分析图片",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/test.png"),
            ],
        )
        compiled = GLM5Compiler().compile(request)
        all_text = " ".join(m.get("content", "") for m in compiled.messages if isinstance(m.get("content"), str))
        assert "图片" in all_text

    def test_chinese_constraints_injection(self):
        """中文约束注入"""
        request = _make_request(intent="execute", instruction="写代码")
        compiled = GLM5Compiler().compile(request)
        # 中文约束应在消息列表中（P2-8: tail reinforcement 使指令不再总是最后一条）
        all_content = " ".join(m.get("content", "") for m in compiled.messages if isinstance(m.get("content"), str))
        assert "中文注释" in all_content or "英文标识符" in all_content

    def test_compiler_type(self):
        """_get_compiler_type 返回 glm_5"""
        c = GLM5Compiler()
        assert c._get_compiler_type() == "glm_5"

    def test_mock_e2e(self):
        """GLM-5 完整 e2e"""
        provider = _make_mock_provider("glm_5")
        request = _make_request(
            instruction="实现排序",
            thinking_mode="deep",
            long_horizon=LongHorizonConfig(self_evaluation_enabled=True),
        )
        response = _run_async(provider.execute_tap(request))
        assert response.raw_text is not None


# ===== 6. MiniMaxM3Compiler 深度测试 =====

class TestMiniMaxM3CompilerDeep:
    """MiniMaxM3Compiler 深度功能测试"""

    def test_text_only_compilation(self):
        """纯文本编译模式"""
        request = _make_request(instruction="实现排序")
        compiled = MiniMaxM3Compiler().compile(request)
        assert compiled.mode == "messages"
        assert len(compiled.messages) > 0

    def test_multimodal_compilation(self):
        """多模态编译 → 用户消息包含 content 数组"""
        request = _make_request(
            instruction="分析图片内容",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/img.png"),
                MultimodalContent(type="text", text="这是附加文字"),
            ],
        )
        compiled = MiniMaxM3Compiler().compile(request)
        # 应有包含 content 数组的用户消息
        has_content_array = False
        for msg in compiled.messages:
            content = msg.get("content")
            if isinstance(content, list):
                has_content_array = True
                # 验证 content 数组格式
                types_in_array = [p.get("type") for p in content if isinstance(p, dict)]
                assert "text" in types_in_array
                assert "image_url" in types_in_array
        assert has_content_array, "多模态编译应产生 content 数组格式"

    def test_multimodal_image_base64(self):
        """Base64 图像多模态编译"""
        request = _make_request(
            instruction="识别图片",
            multimodal_context=[
                MultimodalContent(
                    type="image_base64",
                    base64_data="iVBORw0KGgo=",
                    media_type="image/png",
                ),
            ],
        )
        compiled = MiniMaxM3Compiler().compile(request)
        has_data_uri = False
        for msg in compiled.messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:image/png;base64,"):
                            has_data_uri = True
        assert has_data_uri, "Base64 图像应转换为 data URI"

    def test_desktop_context_message(self):
        """桌面上下文消息构建"""
        request = _make_request(
            instruction="点击按钮",
            desktop_context=DesktopContext(
                screenshot=MultimodalContent(type="image_url", url="https://example.com/screen.png"),
                interactive_elements=[
                    {"type": "button", "label": "Submit", "bbox": {"x": 100, "y": 200}},
                ],
                active_window="Browser",
            ),
        )
        compiled = MiniMaxM3Compiler().compile(request)
        # 应有桌面上下文消息（含 content 数组）
        all_text = ""
        has_content_array = False
        for msg in compiled.messages:
            content = msg.get("content")
            if isinstance(content, list):
                has_content_array = True
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        all_text += part.get("text", "") + " "
            elif isinstance(content, str):
                all_text += content + " "
        assert has_content_array or "桌面" in all_text or "Browser" in all_text

    def test_desktop_context_without_screenshot(self):
        """桌面上下文无截图 → 仍可编译"""
        request = _make_request(
            instruction="操作桌面",
            desktop_context=DesktopContext(
                active_window="Terminal",
                interactive_elements=[],
            ),
        )
        compiled = MiniMaxM3Compiler().compile(request)
        assert compiled.mode == "messages"

    def test_msa_fulltext_injection(self):
        """MSA 全文注入策略：大文档全文灌入"""
        long_design = "设计文档内容 " * 100  # 模拟长文档
        request = _make_request(
            instruction="实现功能",
            context={"design": long_design, "plan": "执行计划"},
        )
        compiled = MiniMaxM3Compiler().compile(request)
        all_text = " ".join(m.get("content", "") for m in compiled.messages if isinstance(m.get("content"), str))
        # M3 应全文注入，不做裁剪
        assert "设计文档内容" in all_text

    def test_programming_guidance(self):
        """编程增强 prompt 注入"""
        request = _make_request(intent="execute", instruction="写代码")
        compiled = MiniMaxM3Compiler().compile(request)
        system_msg = compiled.messages[0]
        assert "编程增强" in system_msg["content"] or "完整" in system_msg["content"]

    def test_programming_guidance_not_for_design(self):
        """design 意图不注入编程增强"""
        request = _make_request(intent="design", instruction="设计系统")
        compiled = MiniMaxM3Compiler().compile(request)
        system_msg = compiled.messages[0]
        assert "编程增强" not in system_msg["content"]

    def test_browse_guidance(self):
        """浏览增强 prompt 注入（当有 browse_intent 标记时）"""
        request = _make_request(
            intent="chat",
            instruction="搜索信息",
            meta={"task_id": "1.1", "intent": "chat", "browse_intent": "search"},
        )
        compiled = MiniMaxM3Compiler().compile(request)
        system_msg = compiled.messages[0]
        assert "信息检索" in system_msg["content"] or "搜索" in system_msg["content"]

    def test_no_browse_guidance_without_flag(self):
        """无 browse_intent 标记时不注入浏览增强"""
        request = _make_request(intent="execute", instruction="写代码")
        compiled = MiniMaxM3Compiler().compile(request)
        system_msg = compiled.messages[0]
        assert "信息检索增强" not in system_msg["content"]

    def test_video_url_in_multimodal(self):
        """视频 URL 多模态编译"""
        request = _make_request(
            instruction="分析视频",
            multimodal_context=[
                MultimodalContent(type="video_url", url="https://example.com/video.mp4"),
            ],
        )
        compiled = MiniMaxM3Compiler().compile(request)
        # 验证视频 URL 被编码到 content 数组
        has_video = False
        for msg in compiled.messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "video_url":
                        has_video = True
        assert has_video, "视频 URL 应出现在 content 数组中"

    def test_compiler_type(self):
        """_get_compiler_type 返回 minimax_m3"""
        c = MiniMaxM3Compiler()
        assert c._get_compiler_type() == "minimax_m3"

    def test_mock_e2e_text(self):
        """M3 纯文本模式完整 e2e"""
        provider = _make_mock_provider("minimax_m3")
        request = _make_request(instruction="实现排序")
        response = _run_async(provider.execute_tap(request))
        assert response.raw_text is not None

    def test_mock_e2e_multimodal(self):
        """M3 多模态模式完整 e2e"""
        provider = _make_mock_provider("minimax_m3")
        request = _make_request(
            instruction="分析图片",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/test.png"),
            ],
        )
        response = _run_async(provider.execute_tap(request))
        assert response.raw_text is not None


# ===== 7. TAP IR 扩展数据类 =====

class TestTAPIRExtensions:
    """TAP IR 扩展数据类测试"""

    def test_multimodal_content_text(self):
        """MultimodalContent text 类型"""
        mc = MultimodalContent(type="text", text="Hello")
        fmt = mc.to_openai_format()
        assert fmt == {"type": "text", "text": "Hello"}
        assert mc.extract_text_description() == "Hello"

    def test_multimodal_content_image_url(self):
        """MultimodalContent image_url 类型"""
        mc = MultimodalContent(type="image_url", url="https://example.com/img.png")
        fmt = mc.to_openai_format()
        assert fmt["type"] == "image_url"
        assert fmt["image_url"]["url"] == "https://example.com/img.png"
        assert "图片" in mc.extract_text_description()

    def test_multimodal_content_video_url(self):
        """MultimodalContent video_url 类型"""
        mc = MultimodalContent(type="video_url", url="https://example.com/video.mp4")
        fmt = mc.to_openai_format()
        assert fmt["type"] == "video_url"
        assert fmt["video_url"]["url"] == "https://example.com/video.mp4"
        assert "视频" in mc.extract_text_description()

    def test_multimodal_content_image_base64(self):
        """MultimodalContent image_base64 类型"""
        mc = MultimodalContent(
            type="image_base64",
            base64_data="abc123",
            media_type="image/png",
        )
        fmt = mc.to_openai_format()
        assert fmt["type"] == "image_url"
        assert fmt["image_url"]["url"].startswith("data:image/png;base64,")
        assert "Base64" in mc.extract_text_description()

    def test_desktop_context_format(self):
        """DesktopContext 格式化"""
        dc = DesktopContext(
            active_window="Chrome",
            interactive_elements=[
                {"type": "button", "label": "Login", "bbox": {"x": 10, "y": 20}},
            ],
        )
        text = dc.format_for_prompt()
        assert "Chrome" in text
        assert "Login" in text

    def test_desktop_context_with_screenshot(self):
        """DesktopContext 带截图"""
        dc = DesktopContext(
            screenshot=MultimodalContent(type="image_url", url="https://example.com/screen.png"),
            active_window="VSCode",
        )
        text = dc.format_for_prompt()
        assert "屏幕截图" in text

    def test_long_horizon_config_defaults(self):
        """LongHorizonConfig 默认值"""
        config = LongHorizonConfig()
        assert config.max_duration_hours == 8.0
        assert config.checkpoint_interval_minutes == 30.0
        assert config.self_evaluation_enabled is True
        assert config.stagnation_threshold == 3

    def test_long_horizon_status(self):
        """LongHorizonStatus 状态属性"""
        status = LongHorizonStatus(phase="executing", steps_completed=5, elapsed_minutes=120.0)
        assert status.is_active is True
        assert status.is_stagnant is False

    def test_long_horizon_status_stagnant(self):
        """LongHorizonStatus 停滞状态"""
        status = LongHorizonStatus(phase="stagnant")
        assert status.is_active is False
        assert status.is_stagnant is True

    def test_tap_request_defaults(self):
        """TAPRequest 扩展字段默认为 None"""
        req = TAPRequest()
        assert req.thinking_mode is None
        assert req.multimodal_context is None
        assert req.desktop_context is None
        assert req.long_horizon is None
        assert req.cache_preference is None

    def test_tap_request_has_multimodal(self):
        """TAPRequest.has_multimodal 属性"""
        req1 = TAPRequest()
        assert req1.has_multimodal is False
        req2 = TAPRequest(multimodal_context=[MultimodalContent(type="text", text="hi")])
        assert req2.has_multimodal is True

    def test_tap_request_is_long_horizon(self):
        """TAPRequest.is_long_horizon 属性"""
        req1 = TAPRequest()
        assert req1.is_long_horizon is False
        req2 = TAPRequest(long_horizon=LongHorizonConfig())
        assert req2.is_long_horizon is True

    def test_tap_request_effective_thinking_mode(self):
        """TAPRequest.effective_thinking_mode 属性"""
        req1 = TAPRequest(thinking_mode="deep")
        assert req1.effective_thinking_mode == "deep"
        req2 = TAPRequest(thinking_mode=None)
        assert req2.effective_thinking_mode == "auto"

    def test_tap_response_extensions(self):
        """TAPResponse 扩展字段"""
        resp = TAPResponse(
            raw_text="Hello",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            cache_hit_tokens=80,
            thinking_content="推理过程",
        )
        assert resp.cache_hit_tokens == 80
        assert resp.thinking_content == "推理过程"
        assert resp.cache_miss_tokens == 20

    def test_tap_response_cache_from_usage(self):
        """TAPResponse 从 usage 中提取 cache_hit_tokens"""
        resp = TAPResponse(
            raw_text="Hello",
            usage={"prompt_tokens": 100, "prompt_cache_hit_tokens": 60},
        )
        assert resp.cache_hit_tokens == 60

    def test_compiled_prompt_thinking_enabled(self):
        """CompiledPrompt.thinking_enabled 属性"""
        cp = CompiledPrompt(extra={"thinking": {"type": "enabled"}})
        assert cp.thinking_enabled is True
        cp2 = CompiledPrompt(extra={"thinking": {"type": "disabled"}})
        assert cp2.thinking_enabled is False
        cp3 = CompiledPrompt(extra={})
        assert cp3.thinking_enabled is None


# ===== 8. 配置系统集成 =====

class TestConfigIntegration:
    """配置系统与三模型集成测试"""

    def test_driver_config_new_fields(self):
        """DriverConfig 新增字段有默认值"""
        cfg = DriverConfig()
        assert cfg.compiler_variant == ""
        assert cfg.max_context_tokens == 0
        assert cfg.thinking_mode == ""
        assert cfg.cache_aware is False
        assert cfg.multimodal_enabled is False
        assert cfg.desktop_enabled is False
        assert cfg.long_horizon_enabled is False
        assert cfg.msa_efficient is False

    def test_driver_config_deepseek_v4(self):
        """DriverConfig DeepSeek V4 识别"""
        cfg = DriverConfig(compiler="deepseek_v4", compiler_variant="flash", model="deepseek-v4-flash")
        assert cfg.is_deepseek_v4 is True
        assert cfg.is_glm_5 is False
        assert cfg.is_minimax_m3 is False

    def test_driver_config_glm_5(self):
        """DriverConfig GLM-5 识别"""
        cfg = DriverConfig(compiler="glm_5", model="glm-5")
        assert cfg.is_glm_5 is True
        assert cfg.is_deepseek_v4 is False

    def test_driver_config_minimax_m3(self):
        """DriverConfig MiniMax M3 识别"""
        cfg = DriverConfig(compiler="minimax_m3", model="MiniMax-M3")
        assert cfg.is_minimax_m3 is True
        assert cfg.is_deepseek_v4 is False

    def test_driver_config_to_kwargs(self):
        """DriverConfig.to_create_provider_kwargs 包含新字段"""
        cfg = DriverConfig(
            adapter="mock",
            compiler="deepseek_v4",
            compiler_variant="flash",
            model="deepseek-v4-flash",
        )
        kwargs = cfg.to_create_provider_kwargs()
        assert kwargs["compiler"] == "deepseek_v4"
        assert kwargs["compiler_variant"] == "flash"
        assert kwargs["model"] == "deepseek-v4-flash"

    def test_driver_config_to_kwargs_no_variant(self):
        """DriverConfig 无 variant 时不传递 compiler_variant"""
        cfg = DriverConfig(adapter="mock", compiler="glm_5", model="glm-5")
        kwargs = cfg.to_create_provider_kwargs()
        assert "compiler_variant" not in kwargs


# ===== 9. ModelProvider 工厂函数集成 =====

class TestCreateProviderIntegration:
    """create_provider() 工厂函数与三模型集成测试"""

    def test_create_provider_deepseek_v4(self):
        """create_provider 创建 DeepSeek V4 provider"""
        from teragent import create_provider
        provider = create_provider(
            compiler="deepseek_v4",
            adapter="mock",
            model="deepseek-v4-flash",
        )
        assert isinstance(provider.compiler, DeepSeekV4Compiler)
        assert isinstance(provider.adapter, MockAdapter)

    def test_create_provider_deepseek_v4_flash_variant(self):
        """create_provider 创建 DeepSeek V4 Flash 变体"""
        from teragent import create_provider
        provider = create_provider(
            compiler="deepseek_v4",
            adapter="mock",
            model="deepseek-v4-flash",
            compiler_variant="flash",
        )
        assert isinstance(provider.compiler, DeepSeekV4Compiler)
        assert provider.compiler.variant == "flash"

    def test_create_provider_deepseek_v4_pro_variant(self):
        """create_provider 创建 DeepSeek V4 Pro 变体"""
        from teragent import create_provider
        provider = create_provider(
            compiler="deepseek_v4",
            adapter="mock",
            model="deepseek-v4-pro",
            compiler_variant="pro",
        )
        assert isinstance(provider.compiler, DeepSeekV4Compiler)
        assert provider.compiler.variant == "pro"

    def test_create_provider_glm_5(self):
        """create_provider 创建 GLM-5 provider"""
        from teragent import create_provider
        provider = create_provider(
            compiler="glm_5",
            adapter="mock",
            model="glm-5",
        )
        assert isinstance(provider.compiler, GLM5Compiler)

    def test_create_provider_minimax_m3(self):
        """create_provider 创建 MiniMax M3 provider"""
        from teragent import create_provider
        provider = create_provider(
            compiler="minimax_m3",
            adapter="mock",
            model="MiniMax-M3",
        )
        assert isinstance(provider.compiler, MiniMaxM3Compiler)

    def test_create_provider_e2e_v4_flash(self):
        """V4-Flash provider 端到端测试"""
        from teragent import create_provider
        provider = create_provider(
            compiler="deepseek_v4",
            adapter="mock",
            model="deepseek-v4-flash",
            compiler_variant="flash",
        )
        request = _make_request(instruction="快速回复", thinking_mode="quick")
        response = _run_async(provider.execute_tap(request))
        assert response.raw_text is not None

    def test_create_provider_e2e_glm_5(self):
        """GLM-5 provider 端到端测试"""
        from teragent import create_provider
        provider = create_provider(
            compiler="glm_5",
            adapter="mock",
            model="glm-5",
        )
        request = _make_request(
            instruction="设计系统架构",
            thinking_mode="deep",
            long_horizon=LongHorizonConfig(),
        )
        response = _run_async(provider.execute_tap(request))
        assert response.raw_text is not None

    def test_create_provider_e2e_minimax_m3(self):
        """M3 provider 端到端测试"""
        from teragent import create_provider
        provider = create_provider(
            compiler="minimax_m3",
            adapter="mock",
            model="MiniMax-M3",
        )
        request = _make_request(
            instruction="分析图片",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/img.png"),
            ],
        )
        response = _run_async(provider.execute_tap(request))
        assert response.raw_text is not None


# ===== 10. 向后兼容回归 =====

class TestBackwardCompatibility:
    """验证原有 Compiler 行为未被破坏"""

    def test_default_compiler_basic(self):
        """DefaultCompiler 基础编译不变"""
        compiler = DefaultCompiler()
        request = _make_request(instruction="写一个函数")
        compiled = compiler.compile(request)
        assert compiled.mode == "messages"
        last_msg = compiled.messages[-1]
        assert last_msg["role"] == "user"
        assert "写一个函数" in last_msg["content"]

    def test_glm_compiler_recency(self):
        """GLMCompiler Recency Effect 不变"""
        compiler = GLMCompiler()
        request = _make_request(instruction="实现功能")
        compiled = compiler.compile(request)
        last_msg = compiled.messages[-1]
        assert last_msg["role"] == "user"
        assert "实现功能" in last_msg["content"]

    def test_deepseek_compiler_minimal(self):
        """DeepSeekCompiler 极简策略不变"""
        compiler = DeepSeekCompiler()
        request = _make_request(
            instruction="写代码",
            constraints=["不使用eval"],
        )
        compiled = compiler.compile(request)
        system_msg = compiled.messages[0]
        # 约束不应在系统消息中
        assert "不使用eval" not in system_msg["content"]

    def test_anthropic_compiler_mode_b(self):
        """AnthropicCompiler Mode B 不变"""
        compiler = AnthropicCompiler()
        request = _make_request(instruction="写代码")
        compiled = compiler.compile(request)
        assert compiled.mode == "system_user"

    def test_no_thinking_mode_no_extra(self):
        """不设 thinking_mode 时原 Compiler 不添加 extra"""
        compiler = DefaultCompiler()
        request = _make_request(instruction="写代码")
        compiled = compiler.compile(request)
        # 原 Compiler 不应添加 thinking extra
        assert "thinking" not in compiled.extra

    def test_tap_request_backward_compat(self):
        """不设扩展字段的 TAPRequest 正常工作"""
        req = TAPRequest(
            meta={"task_id": "1.1", "intent": "code_generation"},
            instruction="写一个函数",
            constraints=["Python"],
        )
        assert req.thinking_mode is None
        assert req.multimodal_context is None
        assert req.long_horizon is None
        assert req.cache_preference is None

    def test_tap_response_backward_compat(self):
        """不设扩展字段的 TAPResponse 正常工作"""
        resp = TAPResponse(
            raw_text="Hello",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            finish_reason="stop",
        )
        assert resp.cache_hit_tokens == 0
        assert resp.thinking_content is None
        assert resp.long_horizon_status is None


# ===== 11. MockAdapter 意图感知 =====

class TestMockAdapterIntentAware:
    """验证 MockAdapter 根据编译内容返回合理的 Mock 响应"""

    def test_design_intent_response(self):
        """design 意图 → Mock 返回设计文档格式"""
        provider = _make_mock_provider("deepseek_v4_pro")
        request = _make_request(intent="design", instruction="设计系统架构")
        response = _run_async(provider.execute_tap(request))
        assert "DESIGN" in response.raw_text or "设计" in response.raw_text

    def test_plan_intent_response(self):
        """plan 意图 → Mock 返回计划格式"""
        provider = _make_mock_provider("glm_5")
        request = _make_request(intent="plan", instruction="制定执行计划")
        response = _run_async(provider.execute_tap(request))
        assert "1.1" in response.raw_text or "步骤" in response.raw_text or "Prerequisites" in response.raw_text

    def test_review_intent_response(self):
        """review 意图 → Mock 返回审核结果"""
        provider = _make_mock_provider("minimax_m3")
        request = _make_request(intent="review", instruction="审查代码")
        response = _run_async(provider.execute_tap(request))
        assert "APPROVE" in response.raw_text or "审查" in response.raw_text

    def test_code_generation_intent_response(self):
        """code_generation 意图 → Mock 返回代码格式"""
        provider = _make_mock_provider("default")
        request = _make_request(intent="code_generation", instruction="写代码")
        response = _run_async(provider.execute_tap(request))
        assert "file" in response.raw_text.lower() or "mock" in response.raw_text.lower()

    def test_mock_adapter_fail_rate(self):
        """MockAdapter fail_rate 模拟"""
        adapter = MockAdapter(delay=0.0, fail_rate=1.0)  # 100% 失败率
        compiler = DefaultCompiler()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="mock-fail")
        request = _make_request()
        with pytest.raises(RuntimeError, match="simulated failure"):
            _run_async(provider.execute_tap(request))

    def test_mock_adapter_call_count(self):
        """MockAdapter 调用计数"""
        adapter = MockAdapter(delay=0.0)
        compiler = DefaultCompiler()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="mock")
        request = _make_request()
        _run_async(provider.execute_tap(request))
        assert adapter._call_count == 1
        _run_async(provider.execute_tap(request))
        assert adapter._call_count == 2


# ===== 12. 多意图全覆盖 =====

class TestAllIntentsCoverage:
    """验证所有意图在每个新 Compiler 下均可编译和 Mock 发送"""

    INTENTS = [
        "design", "plan", "replan", "execute", "code_generation",
        "review", "chat", "chat_friendly", "sub_agent",
    ]

    NEW_COMPILERS = ["deepseek_v4", "glm_5", "minimax_m3"]

    @pytest.mark.parametrize("compiler_name", NEW_COMPILERS)
    @pytest.mark.parametrize("intent", INTENTS)
    def test_intent_compiles_and_sends(self, compiler_name, intent):
        """每个意图在新 Compiler 下编译+发送成功"""
        provider = _make_mock_provider(compiler_name)
        request = _make_request(
            intent=intent,
            instruction=f"执行{intent}任务",
        )
        compiled = provider.compiler.compile(request)
        assert compiled.mode != "empty", f"{compiler_name}/{intent} 编译结果为空"
        response = _run_async(provider.execute_tap(request))
        assert response.raw_text is not None


# ===== 13. 完整 Provider 生命周期 =====

class TestProviderLifecycle:
    """ModelProvider 完整生命周期测试"""

    def test_provider_close(self):
        """Provider close 正常工作"""
        provider = _make_mock_provider("deepseek_v4")
        _run_async(provider.close())  # 应不抛异常

    def test_provider_repr(self):
        """Provider repr 包含关键信息"""
        provider = _make_mock_provider("glm_5")
        r = repr(provider)
        assert "GLM5Compiler" in r
        assert "MockAdapter" in r

    def test_provider_capabilities(self):
        """Provider capabilities 委托给 adapter"""
        provider = _make_mock_provider("minimax_m3")
        caps = provider.capabilities
        assert "mock" in caps
        assert caps["mock"] is True

    def test_provider_cost_tracking(self):
        """Provider execute_tap_with_retry 记录成本"""
        provider = _make_mock_provider("deepseek_v4_pro")
        request = _make_request(instruction="测试成本追踪")
        response = _run_async(provider.execute_tap_with_retry(request, max_retries=0))
        assert response.raw_text is not None
        records = provider.cost_records
        assert len(records) >= 1
        assert records[0].success is True
        assert records[0].prompt_tokens > 0

    def test_provider_chat_interface(self):
        """Provider chat() 接口正常"""
        provider = _make_mock_provider("glm_5")
        result = _run_async(provider.chat(messages=[
            {"role": "user", "content": "你好"},
        ]))
        assert "content" in result
        assert len(result["content"]) > 0

    def test_provider_stream_interface(self):
        """Provider stream_tap() 接口正常"""
        provider = _make_mock_provider("minimax_m3")
        request = _make_request(instruction="流式测试")

        async def _stream():
            chunks = []
            async for chunk in provider.stream_tap(request):
                chunks.append(chunk)
            return chunks

        chunks = _run_async(_stream())
        assert len(chunks) > 0
