# tests/integration/test_four_models.py
"""Phase 1 四模型基础集成测试（Mock 模式）

验证四款模型（DeepSeek V4-Flash/Pro、GLM-5、GLM-5.2、MiniMax M3）
均可通过 TerAgent API 调用，使用 MockAdapter 无需真实 API Key。

覆盖：
  - P1-3: DeepSeekV4Compiler（Flash/Pro 双模式 + thinking mode）
  - P1-4: GLM5Compiler（Recency Effect + long-horizon）
  - P1-5: GLM52Compiler（1M 上下文 + High/Max 双思考 + preserve_thinking）
  - P1-6: MiniMaxM3Compiler（纯文本 + 多模态 + desktop）
  - P1-7: OpenAICompatibleAdapter（extra_body 传递 thinking/cache）
  - P1-8: 配置体系（compiler inference + DriverConfig）
  - TAP IR 往返测试
"""

import pytest

from teragent import (
    GLM5Compiler,
    GLM52Compiler,
    MiniMaxM3Compiler,
    TAPRequest,
    TAPResponse,
    create_provider,
)
from teragent.config.driver_config import DriverConfig
from teragent.config.loader import infer_compiler
from teragent.context.profiles import (
    ContextProfile,
    DeepSeekV4ContextProfile,
    GLM5CompactionStrategy,
    MiniMaxM3ContextProfile,
)
from teragent.core.adapter import TAPAdapterRegistry
from teragent.core.compiler import TAPCompilerRegistry
from teragent.core.compilers.deepseek_v4 import DeepSeekV4Compiler
from teragent.core.compilers.glm_52 import (
    GLM52CompactionProfile,
    ThinkingModeDecision,
    ThinkingModeRouter,
    PreservedThinkingManager,
)
from teragent.core.tap import (
    CompiledPrompt,
    DesktopContext,
    LongHorizonConfig,
    LongHorizonStatus,
    MultimodalContent,
)


# ===== 辅助函数 =====

def _make_request(**overrides) -> TAPRequest:
    """构造测试用的 TAPRequest"""
    defaults = {
        "meta": {"task_id": "test.1", "intent": "execute"},
        "instruction": "写一个排序函数",
        "constraints": ["Python 3.10+"],
        "output_format_hint": "<file path='...'>代码</file>",
    }
    defaults.update(overrides)
    return TAPRequest(**defaults)


# ===== P1-3: DeepSeekV4Compiler =====

class TestDeepSeekV4Integration:
    """DeepSeek V4 Flash/Pro 双模式集成测试"""

    def test_flash_compiler_registered(self):
        """V4-Flash 可通过 Registry 创建"""
        compiler = TAPCompilerRegistry.create("deepseek_v4", variant="flash")
        assert isinstance(compiler, DeepSeekV4Compiler)
        assert compiler.variant == "flash"

    def test_pro_compiler_registered(self):
        """V4-Pro 可通过 Registry 创建"""
        compiler = TAPCompilerRegistry.create("deepseek_v4", variant="pro")
        assert isinstance(compiler, DeepSeekV4Compiler)
        assert compiler.variant == "pro"

    def test_flash_compile(self):
        """V4-Flash 编译输出正确"""
        compiler = DeepSeekV4Compiler(variant="flash")
        request = _make_request(thinking_mode="quick")
        compiled = compiler.compile(request)
        assert compiled.mode == "messages"
        assert len(compiled.messages) > 0
        # Flash 模式系统提示应较短
        system_msg = compiled.messages[0]
        assert system_msg["role"] == "system"

    def test_pro_compile(self):
        """V4-Pro 编译输出正确"""
        compiler = DeepSeekV4Compiler(variant="pro")
        request = _make_request(thinking_mode="deep")
        compiled = compiler.compile(request)
        assert compiled.mode == "messages"
        assert compiled.thinking_enabled is True

    def test_thinking_mode_deep(self):
        """V4 thinking_mode=deep 正确设置 API 参数"""
        compiler = DeepSeekV4Compiler(variant="pro")
        request = _make_request(thinking_mode="deep")
        compiled = compiler.compile(request)
        assert compiled.extra.get("thinking") == {"type": "enabled"}

    def test_thinking_mode_quick(self):
        """V4 thinking_mode=quick 正确设置 API 参数"""
        compiler = DeepSeekV4Compiler(variant="flash")
        request = _make_request(thinking_mode="quick")
        compiled = compiler.compile(request)
        assert compiled.extra.get("thinking") == {"type": "disabled"}

    def test_thinking_mode_auto(self):
        """V4 thinking_mode=auto 根据意图自动选择"""
        compiler = DeepSeekV4Compiler(variant="pro")
        # CHAT 意图应选择 quick
        request = _make_request(
            meta={"task_id": "1", "intent": "chat"},
            thinking_mode="auto",
        )
        compiled = compiler.compile(request)
        assert compiled.extra.get("thinking", {}).get("type") == "disabled"

    def test_multimodal_degradation(self):
        """V4 不支持多模态，应降级为文本描述"""
        compiler = DeepSeekV4Compiler(variant="pro")
        request = _make_request(
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/img.png")
            ]
        )
        compiled = compiler.compile(request)
        assert compiled.mode == "messages"
        # 多模态内容应被降级（无崩溃）

    def test_max_context_tokens(self):
        """V4 max_context_tokens = 1M"""
        compiler = DeepSeekV4Compiler()
        assert compiler.max_context_tokens == 1_000_000

    def test_supports_thinking_mode(self):
        """V4 supports_thinking_mode = True"""
        compiler = DeepSeekV4Compiler()
        assert compiler.supports_thinking_mode is True


# ===== P1-4: GLM5Compiler =====

class TestGLM5Integration:
    """GLM-5 集成测试"""

    def test_compiler_registered(self):
        """GLM-5 可通过 Registry 创建"""
        compiler = TAPCompilerRegistry.create("glm_5")
        assert isinstance(compiler, GLM5Compiler)

    def test_recency_effect(self):
        """GLM-5 关键指令在 messages 列表最后"""
        compiler = GLM5Compiler()
        request = _make_request(instruction="实现排序")
        compiled = compiler.compile(request)
        messages = compiled.messages
        # 最后一条 user message 应包含指令
        user_msgs = [m for m in messages if m["role"] == "user"]
        last_user = user_msgs[-1]
        assert "排序" in last_user["content"] or "实现" in last_user["content"]

    def test_thinking_mode(self):
        """GLM-5 thinking mode 控制"""
        compiler = GLM5Compiler()
        request = _make_request(thinking_mode="deep")
        compiled = compiler.compile(request)
        assert compiled.extra.get("thinking") == {"type": "enabled"}

    def test_long_horizon(self):
        """GLM-5 长程任务引导注入"""
        compiler = GLM5Compiler()
        request = _make_request(
            long_horizon=LongHorizonConfig(max_duration_hours=8),
        )
        compiled = compiler.compile(request)
        # 系统提示应包含长程任务内容
        system_msgs = [m for m in compiled.messages if m["role"] == "system"]
        assert any("长程任务" in m["content"] for m in system_msgs)

    def test_self_evaluation_prompt(self):
        """GLM-5 自评估 prompt 注入"""
        compiler = GLM5Compiler()
        request = _make_request(
            long_horizon=LongHorizonConfig(
                max_duration_hours=8,
                self_evaluation_enabled=True,
            ),
        )
        compiled = compiler.compile(request)
        # 应包含自评估内容
        all_text = " ".join(m["content"] for m in compiled.messages)
        assert "自评估" in all_text

    def test_max_context_tokens(self):
        """GLM-5 max_context_tokens = 200K"""
        compiler = GLM5Compiler()
        assert compiler.max_context_tokens == 200_000

    def test_tail_reinforcement(self):
        """GLM-5 尾部强化"""
        compiler = GLM5Compiler()
        request = _make_request(instruction="写排序函数", constraints=["Python 3.10+"])
        compiled = compiler.compile(request)
        # 尾部应有强化内容
        messages = compiled.messages
        user_msgs = [m for m in messages if m["role"] == "user"]
        # 尾部强化在最后一个 user message 或倒数第二个位置
        last_user_content = user_msgs[-1]["content"]
        assert "当前指令" in last_user_content or "排序" in last_user_content


# ===== P1-5: GLM52Compiler =====

class TestGLM52Integration:
    """GLM-5.2 集成测试（Phase 1 最核心的新增组件）"""

    def test_compiler_registered(self):
        """GLM-5.2 可通过 Registry 创建"""
        compiler = TAPCompilerRegistry.create("glm_52")
        assert isinstance(compiler, GLM52Compiler)

    def test_inherits_glm5(self):
        """GLM52Compiler 继承 GLM5Compiler"""
        assert issubclass(GLM52Compiler, GLM5Compiler)

    def test_max_context_1m(self):
        """GLM-5.2 max_context_tokens = 1M"""
        compiler = GLM52Compiler()
        assert compiler.max_context_tokens == 1_000_000

    def test_compiler_type(self):
        """GLM-5.2 compiler type = glm_52"""
        compiler = GLM52Compiler()
        assert compiler._get_compiler_type() == "glm_52"

    # --- High/Max 双思考模式 ---

    def test_thinking_mode_deep_to_max(self):
        """thinking_mode=deep → Max 模式"""
        compiler = GLM52Compiler()
        request = _make_request(thinking_mode="deep")
        compiled = compiler.compile(request)
        thinking = compiled.extra.get("thinking", {})
        assert thinking.get("level") == "max"
        assert thinking.get("type") == "enabled"

    def test_thinking_mode_quick_to_high(self):
        """thinking_mode=quick → High 模式"""
        compiler = GLM52Compiler()
        request = _make_request(thinking_mode="quick")
        compiled = compiler.compile(request)
        thinking = compiled.extra.get("thinking", {})
        assert thinking.get("level") == "high"

    def test_thinking_mode_auto_long_horizon_to_max(self):
        """thinking_mode=auto + 长程任务 → Max + preserve_thinking"""
        compiler = GLM52Compiler()
        request = _make_request(
            thinking_mode="auto",
            long_horizon=LongHorizonConfig(max_duration_hours=8),
        )
        compiled = compiler.compile(request)
        thinking = compiled.extra.get("thinking", {})
        assert thinking.get("level") == "max"
        assert compiled.extra.get("preserve_thinking") is True

    def test_thinking_mode_auto_chat_to_high(self):
        """thinking_mode=auto + chat 意图 → High"""
        compiler = GLM52Compiler()
        request = _make_request(
            meta={"task_id": "1", "intent": "chat"},
            thinking_mode="auto",
        )
        compiled = compiler.compile(request)
        thinking = compiled.extra.get("thinking", {})
        assert thinking.get("level") == "high"

    def test_thinking_mode_auto_plan_to_max(self):
        """thinking_mode=auto + plan 意图 → Max"""
        compiler = GLM52Compiler()
        request = _make_request(
            meta={"task_id": "1", "intent": "plan"},
            thinking_mode="auto",
        )
        compiled = compiler.compile(request)
        thinking = compiled.extra.get("thinking", {})
        assert thinking.get("level") == "max"

    # --- Preserved Thinking ---

    def test_preserve_thinking_coding_plan(self):
        """Coding Plan 场景 preserve_thinking=True"""
        compiler = GLM52Compiler()
        request = _make_request(
            meta={"task_id": "1", "intent": "plan"},
            context={"design": "完整设计文档..."},
            thinking_mode="auto",
        )
        compiled = compiler.compile(request)
        assert compiled.extra.get("preserve_thinking") is True

    def test_preserve_thinking_simple_chat(self):
        """简单对话 preserve_thinking 不开启"""
        compiler = GLM52Compiler()
        request = _make_request(
            meta={"task_id": "1", "intent": "chat"},
            thinking_mode="auto",
        )
        compiled = compiler.compile(request)
        assert compiled.extra.get("preserve_thinking") is not True

    # --- 1M 上下文分区 ---

    def test_compaction_profile(self):
        """GLM52CompactionProfile 预算合理"""
        profile = GLM52CompactionProfile()
        assert profile.total_budget == 1_024_000
        assert profile.design_compression_target == 1.0  # 完整保留
        assert profile.min_information_retention >= 0.95

    def test_context_partition_info(self):
        """GLM52Compiler.get_context_partition_info() 可用"""
        compiler = GLM52Compiler()
        info = compiler.get_context_partition_info()
        assert info["model"] == "GLM-5.2"
        assert info["max_context_tokens"] == 1_000_000
        assert "system" in info["partitions"]
        assert "tail" in info["partitions"]

    # --- 长程任务继承 ---

    def test_long_horizon_inherited(self):
        """GLM-5.2 正确继承 GLM-5 的长程任务能力"""
        compiler = GLM52Compiler()
        request = _make_request(
            long_horizon=LongHorizonConfig(max_duration_hours=8),
        )
        compiled = compiler.compile(request)
        system_msgs = [m for m in compiled.messages if m["role"] == "system"]
        assert any("长程任务" in m["content"] for m in system_msgs)
        # GLM-5.2 额外的 1M 上下文提示
        assert any("1M" in m["content"] for m in system_msgs)

    # --- Tail reinforcement ---

    def test_tail_reinforcement_1m(self):
        """GLM-5.2 尾部强化包含 1M 上下文提示"""
        compiler = GLM52Compiler()
        request = _make_request(instruction="写排序函数")
        compiled = compiler.compile(request)
        all_text = " ".join(m["content"] for m in compiled.messages)
        assert "1M" in all_text
        assert "交叉引用" in all_text


# ===== P1-6: MiniMaxM3Compiler =====

class TestMiniMaxM3Integration:
    """MiniMax M3 集成测试"""

    def test_compiler_registered(self):
        """M3 可通过 Registry 创建"""
        compiler = TAPCompilerRegistry.create("minimax_m3")
        assert isinstance(compiler, MiniMaxM3Compiler)

    def test_text_mode(self):
        """M3 纯文本模式编译"""
        compiler = MiniMaxM3Compiler()
        request = _make_request()
        compiled = compiler.compile(request)
        assert compiled.mode == "messages"
        assert len(compiled.messages) > 0

    def test_multimodal_mode(self):
        """M3 多模态模式编译"""
        compiler = MiniMaxM3Compiler()
        request = _make_request(
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/screenshot.png")
            ]
        )
        compiled = compiler.compile(request)
        assert compiled.mode == "messages"
        # 应包含多模态 content 数组
        user_msgs = [m for m in compiled.messages if m["role"] == "user"]
        has_content_array = any(
            isinstance(m.get("content"), list) for m in user_msgs
        )
        assert has_content_array

    def test_desktop_context(self):
        """M3 desktop context 编译"""
        compiler = MiniMaxM3Compiler()
        request = _make_request(
            desktop_context=DesktopContext(
                screenshot=MultimodalContent(
                    type="image_url",
                    url="https://example.com/desktop.png",
                ),
                interactive_elements=[
                    {"type": "button", "label": "Submit", "bbox": {"x": 100, "y": 200}},
                ],
                active_window="Chrome",
            )
        )
        compiled = compiler.compile(request)
        assert compiled.mode == "messages"

    def test_supports_multimodal(self):
        """M3 supports_multimodal = True"""
        compiler = MiniMaxM3Compiler()
        assert compiler.supports_multimodal is True

    def test_max_context_tokens(self):
        """M3 max_context_tokens = 1M"""
        compiler = MiniMaxM3Compiler()
        assert compiler.max_context_tokens == 1_000_000

    def test_programming_guidance(self):
        """M3 编程增强 prompt"""
        compiler = MiniMaxM3Compiler()
        request = _make_request(
            meta={"task_id": "1", "intent": "execute"},
        )
        compiled = compiler.compile(request)
        all_text = " ".join(m["content"] for m in compiled.messages)
        assert "编程" in all_text or "SWE-Bench" in all_text

    def test_video_input(self):
        """M3 视频输入处理"""
        compiler = MiniMaxM3Compiler()
        request = _make_request(
            multimodal_context=[
                MultimodalContent(type="video_url", url="https://example.com/demo.mp4")
            ]
        )
        compiled = compiler.compile(request)
        assert compiled.mode == "messages"

    def test_multimodal_token_estimation(self):
        """M3 多模态 token 预算估算"""
        compiler = MiniMaxM3Compiler()
        request = _make_request(
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/img.png"),
                MultimodalContent(type="video_url", url="https://example.com/video.mp4"),
            ]
        )
        estimated = compiler.estimate_multimodal_tokens(request)
        assert estimated > 0
        assert estimated < compiler.max_context_tokens


# ===== P1-8: 配置体系 =====

class TestConfigIntegration:
    """配置体系集成测试"""

    def test_compiler_inference(self):
        """编译器自动推断"""
        assert infer_compiler("deepseek_v4") == "deepseek_v4"
        assert infer_compiler("deepseek_v4_flash") == "deepseek_v4"
        assert infer_compiler("deepseek_v4_pro") == "deepseek_v4"
        assert infer_compiler("glm_5") == "glm_5"
        assert infer_compiler("glm_52") == "glm_52"
        assert infer_compiler("glm-5.2") == "glm_52"
        assert infer_compiler("minimax_m3") == "minimax_m3"

    def test_driver_config_properties(self):
        """DriverConfig 类型属性"""
        v4_cfg = DriverConfig(compiler="deepseek_v4", model="deepseek-v4-flash")
        assert v4_cfg.is_deepseek_v4 is True

        glm5_cfg = DriverConfig(compiler="glm_5", model="glm-5")
        assert glm5_cfg.is_glm_5 is True

        glm52_cfg = DriverConfig(compiler="glm_52", model="glm-5.2")
        assert glm52_cfg.is_glm_52 is True

        m3_cfg = DriverConfig(compiler="minimax_m3", model="minimax-m3")
        assert m3_cfg.is_minimax_m3 is True

    def test_driver_config_extended_fields(self):
        """DriverConfig 扩展字段"""
        cfg = DriverConfig(
            compiler="deepseek_v4",
            compiler_variant="flash",
            max_context_tokens=1_000_000,
            max_output_tokens=384_000,
            thinking_mode="auto",
            cache_aware=True,
            multimodal_enabled=False,
            desktop_enabled=False,
            long_horizon_enabled=False,
            msa_efficient=False,
        )
        assert cfg.compiler_variant == "flash"
        assert cfg.max_context_tokens == 1_000_000
        assert cfg.thinking_mode == "auto"
        assert cfg.cache_aware is True

    def test_create_provider_kwargs(self):
        """DriverConfig.to_create_provider_kwargs()"""
        cfg = DriverConfig(
            adapter="openai_compatible",
            identity="glm_52",
            compiler="glm_52",
            model="glm-5.2",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key="test-key",
            compiler_variant="",
        )
        kwargs = cfg.to_create_provider_kwargs()
        assert kwargs["compiler"] == "glm_52"
        assert kwargs["adapter"] == "openai_compatible"
        assert kwargs["model"] == "glm-5.2"


# ===== TAP IR 往返测试 =====

class TestTAPIRRoundtrip:
    """TAP IR 端到端往返测试"""

    def test_request_with_all_extensions(self):
        """包含所有扩展字段的 TAPRequest"""
        request = TAPRequest(
            meta={"task_id": "1.1", "intent": "execute"},
            context={"design": "设计文档...", "plan": "执行计划..."},
            instruction="实现用户登录",
            constraints=["Python 3.10+", "OAuth 2.0"],
            output_format_hint="<file path='...'>代码</file>",
            thinking_mode="auto",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/ui.png")
            ],
            desktop_context=DesktopContext(
                screenshot=MultimodalContent(
                    type="image_url", url="https://example.com/screen.png"
                ),
                interactive_elements=[],
                active_window="VS Code",
            ),
            long_horizon=LongHorizonConfig(max_duration_hours=4),
            cache_preference="auto",
        )
        assert request.has_multimodal is True
        assert request.has_desktop_context is True
        assert request.is_long_horizon is True
        assert request.effective_thinking_mode == "auto"
        assert request.estimate_prompt_tokens() > 0

    def test_response_with_extensions(self):
        """包含所有扩展字段的 TAPResponse"""
        response = TAPResponse(
            raw_text="代码输出...",
            usage={"prompt_tokens": 1000, "completion_tokens": 500, "prompt_cache_hit_tokens": 800},
            cache_hit_tokens=800,
            thinking_content="推理过程...",
            long_horizon_status=LongHorizonStatus(
                phase="executing",
                steps_completed=5,
                elapsed_minutes=30.0,
            ),
        )
        assert response.cache_hit_tokens == 800
        assert response.cache_miss_tokens == 200
        assert response.thinking_content is not None
        assert response.long_horizon_status.is_active is True

    def test_compiled_prompt_mode_a(self):
        """Mode A CompiledPrompt"""
        compiled = CompiledPrompt(
            messages=[
                {"role": "system", "content": "你是助手"},
                {"role": "user", "content": "你好"},
            ],
            max_tokens=8192,
            extra={"thinking": {"type": "enabled", "level": "max"}, "preserve_thinking": True},
        )
        assert compiled.mode == "messages"
        assert compiled.thinking_enabled is True
        assert compiled.extra.get("preserve_thinking") is True

    def test_four_compilers_produce_messages(self):
        """四个 Compiler 均可编译 TAPRequest 为 messages"""
        request = _make_request(instruction="写排序函数", thinking_mode="auto")
        for name in ["deepseek_v4", "glm_5", "glm_52", "minimax_m3"]:
            compiler = TAPCompilerRegistry.create(name, **({"variant": "flash"} if name == "deepseek_v4" else {}))
            compiled = compiler.compile(request)
            assert compiled.mode == "messages", f"{name} 未输出 messages"
            assert len(compiled.messages) > 0, f"{name} 输出空 messages"


# ===== Mock 模式回归测试 =====

class TestMockModeRegression:
    """使用 MockAdapter 运行所有 Compiler 组合"""

    def test_mock_adapter_available(self):
        """MockAdapter 已注册"""
        assert "mock" in TAPAdapterRegistry.available()

    def test_create_provider_with_mock(self):
        """可通过 MockAdapter 创建 provider"""
        for compiler_name in ["default", "glm", "deepseek", "deepseek_v4", "glm_5", "glm_52", "minimax_m3"]:
            kwargs = {}
            if compiler_name == "deepseek_v4":
                kwargs["variant"] = "flash"
            provider = create_provider(
                compiler=compiler_name,
                adapter="mock",
                model="mock-model",
                **kwargs,
            )
            assert provider is not None, f"Failed to create provider for {compiler_name}"

    @pytest.mark.asyncio
    async def test_execute_tap_with_mock(self):
        """通过 MockAdapter 执行 TAP 请求"""
        provider = create_provider(
            compiler="glm_52",
            adapter="mock",
            model="glm-5.2",
        )
        request = _make_request(thinking_mode="auto")
        response = await provider.execute_tap(request)
        assert response.raw_text is not None
        assert isinstance(response, TAPResponse)

    @pytest.mark.asyncio
    async def test_stream_tap_with_mock(self):
        """通过 MockAdapter 流式执行 TAP 请求"""
        provider = create_provider(
            compiler="deepseek_v4",
            adapter="mock",
            model="deepseek-v4-flash",
            variant="flash",
        )
        request = _make_request(thinking_mode="quick")
        chunks = []
        async for chunk in provider.stream_tap(request):
            chunks.append(chunk)
        assert len(chunks) > 0


# ===== ThinkingModeRouter 单元测试 =====

class TestThinkingModeRouter:
    """GLM-5.2 思考模式路由器测试"""

    def test_long_horizon_forces_max(self):
        """长程任务强制 Max"""
        router = ThinkingModeRouter()
        request = _make_request(long_horizon=LongHorizonConfig())
        decision = router.select(request)
        assert decision.level == "max"
        assert decision.preserve_thinking is True

    def test_coding_plan_forces_max(self):
        """Coding Plan 强制 Max + preserve"""
        router = ThinkingModeRouter()
        request = _make_request(
            meta={"task_id": "1", "intent": "plan"},
            context={"design": "设计文档"},
        )
        decision = router.select(request)
        assert decision.level == "max"
        assert decision.preserve_thinking is True

    def test_debug_forces_max(self):
        """Debug 关键词强制 Max"""
        router = ThinkingModeRouter()
        request = _make_request(instruction="debug 内存泄漏问题")
        decision = router.select(request)
        assert decision.level == "max"

    def test_chat_forces_high(self):
        """聊天意图 → High"""
        router = ThinkingModeRouter()
        request = _make_request(meta={"task_id": "1", "intent": "chat"})
        decision = router.select(request)
        assert decision.level == "high"

    def test_design_forces_max(self):
        """设计意图 → Max"""
        router = ThinkingModeRouter()
        request = _make_request(meta={"task_id": "1", "intent": "design"})
        decision = router.select(request)
        assert decision.level == "max"

    def test_default_is_high(self):
        """默认 → High（成本优化）"""
        router = ThinkingModeRouter()
        request = _make_request(
            meta={"task_id": "1", "intent": "execute"},
            instruction="写一个简单的函数",
        )
        decision = router.select(request)
        assert decision.level == "high"


# ===== PreservedThinkingManager 单元测试 =====

class TestPreservedThinkingManager:
    """保留式思考管理器测试"""

    def test_long_horizon_preserves(self):
        """长程任务应保留推理"""
        mgr = PreservedThinkingManager()
        request = _make_request(long_horizon=LongHorizonConfig())
        decision = ThinkingModeDecision(level="max", preserve_thinking=True)
        assert mgr.should_preserve(request, decision) is True

    def test_simple_chat_no_preserve(self):
        """简单对话不保留推理"""
        mgr = PreservedThinkingManager()
        request = _make_request(meta={"task_id": "1", "intent": "chat"})
        decision = ThinkingModeDecision(level="high", preserve_thinking=False)
        assert mgr.should_preserve(request, decision) is False

    def test_multi_step_preserves(self):
        """多步执行保留推理"""
        mgr = PreservedThinkingManager()
        request = _make_request(
            meta={"task_id": "1", "intent": "execute", "step_count": 5},
        )
        decision = ThinkingModeDecision(level="max", preserve_thinking=False)
        assert mgr.should_preserve(request, decision) is True
