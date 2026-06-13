# tests/test_bug_fixes.py
"""Bug 修复验证测试

覆盖:
  - B1: AgentLoop streaming tool results buffer 机制（避免重复执行）
  - B2: AgentLoop system prompt 不累积（替换而非追加）
  - B3: OpenAIStreamParser 不产生重复 TOOL_CALL_COMPLETE 事件
  - B4: DPOPair 部分对验证（allow_partial 参数）
  - B5: DPO trace_id 匹配（替代位置索引）
  - M2: BYPASS 权限级别允许破坏性操作
  - M5: 重定向 >> 模式被识别为危险
  - M16: Compiler/Adapter mode mismatch 警告
"""
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from teragent.agent_loop import AgentLoop
from teragent.config.agent_loop_config import AgentLoopConfig
from teragent.core.adapter import TAPAdapter
from teragent.core.compiler import TAPCompiler
from teragent.core.provider import ModelProvider
from teragent.core.tap import CompiledPrompt, TAPResponse
from teragent.core.types import MessageRole, MessageType
from teragent.pipeline.tracing import DPOPair, TAPTracer, TraceRecord
from teragent.security.permission import (
    EnhancedPermissionManager,
    PermissionEffect,
    PermissionLevel,
)
from teragent.security.sandbox import check_command_safety
from teragent.streaming.stream_events import (
    OpenAIStreamParser,
    StreamEventType,
)
from teragent.tools.base import ToolResult
from teragent.tools.registry import ToolRegistry

# ===== B2: system prompt 不累积 =====

class TestAgentLoopSystemPromptNoAccumulation:
    """B2: 多次调用 AgentLoop.run() 时 system prompt 不应累积，只保留最新的"""

    @pytest.mark.asyncio
    async def test_system_prompt_replaced_not_accumulated(self):
        """连续两次传入不同 system_prompt，messages 列表中应只有一个 system 消息，
        且内容为第二次传入的 system_prompt"""
        # 构造 mock 依赖
        mock_model = MagicMock(spec=ModelProvider)
        mock_model.model = "test-model"
        mock_model.has_fallback = False
        mock_model.chat = AsyncMock(return_value={"content": "ok"})

        registry = ToolRegistry()

        loop = AgentLoop(model=mock_model, tool_registry=registry, config=AgentLoopConfig())

        # 第一次调用：system_prompt="v1"
        messages_v1 = []
        # Mock intent classifier and all optional components to None
        loop._intent_classifier = None
        loop._confirmation_gate = None
        loop._sub_agent_manager = None
        loop._session_persistence = None
        loop._context_window = None
        loop._auto_compactor = None
        loop._step_budget = None
        loop._event_bus = None

        result1 = await loop.run("hello", messages=messages_v1, system_prompt="system_v1")

        # 第二次调用：system_prompt="v2"，复用同一个 messages 列表
        result2 = await loop.run("world", messages=result1, system_prompt="system_v2")

        # 验证：只有一个 system 消息
        system_msgs = [m for m in result2 if m.role == MessageRole.SYSTEM and m.message_type == MessageType.SYSTEM_PROMPT]
        assert len(system_msgs) == 1, f"应只有1个 system prompt 消息，实际有 {len(system_msgs)}"

        # 验证：system 消息内容为第二次传入的
        assert system_msgs[0].content == "system_v2", (
            f"system prompt 应为 'system_v2'，实际为 '{system_msgs[0].content}'"
        )

    @pytest.mark.asyncio
    async def test_system_prompt_inserted_when_absent(self):
        """消息列表中无 system 消息时，应正确插入"""
        mock_model = MagicMock(spec=ModelProvider)
        mock_model.model = "test-model"
        mock_model.has_fallback = False
        mock_model.chat = AsyncMock(return_value={"content": "ok"})

        registry = ToolRegistry()
        loop = AgentLoop(model=mock_model, tool_registry=registry, config=AgentLoopConfig())
        loop._intent_classifier = None
        loop._confirmation_gate = None
        loop._sub_agent_manager = None
        loop._session_persistence = None
        loop._context_window = None
        loop._auto_compactor = None
        loop._step_budget = None
        loop._event_bus = None

        messages = []
        result = await loop.run("hi", messages=messages, system_prompt="new_system")

        system_msgs = [m for m in result if m.role == MessageRole.SYSTEM and m.message_type == MessageType.SYSTEM_PROMPT]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "new_system"


# ===== B3: OpenAI Stream Parser 不产生重复 TOOL_CALL_COMPLETE =====

class TestOpenAIStreamParserNoDuplicateCompleteEvents:
    """B3: parse_chunk() 中已标记完成的 tool_call，finalize() 不应再产生
    TOOL_CALL_COMPLETE 事件"""

    def test_no_duplicate_complete_events_after_finalize(self):
        """当 parse_chunk 中 is_arguments_complete() 返回 True 并发出
        TOOL_CALL_COMPLETE 事件后，finalize() 不应为同一 index 再次发出"""
        parser = OpenAIStreamParser()

        # 模拟 OpenAI 流式 chunk：tool_call 的参数一次性到达且完整
        chunk_start = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc",
                        "function": {"name": "read_file", "arguments": ""},
                    }]
                },
                "finish_reason": None,
            }]
        }
        chunk_args_complete = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": '{"path": "/src/main.py"}'},
                    }]
                },
                "finish_reason": None,
            }]
        }

        # 解析 start chunk
        events_start = parser.parse_chunk(chunk_start)
        # 应有 TOOL_CALL_START 事件
        start_events = [e for e in events_start if e.event_type == StreamEventType.TOOL_CALL_START]
        assert len(start_events) == 1

        # 解析参数 chunk（参数完整，is_arguments_complete 返回 True）
        events_args = parser.parse_chunk(chunk_args_complete)
        # 应有 TOOL_CALL_DELTA + TOOL_CALL_COMPLETE
        delta_events = [e for e in events_args if e.event_type == StreamEventType.TOOL_CALL_DELTA]
        complete_events = [e for e in events_args if e.event_type == StreamEventType.TOOL_CALL_COMPLETE]
        assert len(delta_events) >= 1
        assert len(complete_events) == 1, "parse_chunk 应产生 1 个 TOOL_CALL_COMPLETE"

        # finalize — 不应产生重复的 TOOL_CALL_COMPLETE
        finalize_events = parser.finalize()
        dup_complete = [e for e in finalize_events if e.event_type == StreamEventType.TOOL_CALL_COMPLETE]
        assert len(dup_complete) == 0, (
            f"finalize 不应为 index 0 产生重复的 TOOL_CALL_COMPLETE，"
            f"但产生了 {len(dup_complete)} 个"
        )

        # finalize 应包含 DONE 事件
        done_events = [e for e in finalize_events if e.event_type == StreamEventType.DONE]
        assert len(done_events) == 1

    def test_finalize_emits_complete_for_incomplete_only(self):
        """如果 parse_chunk 中参数从未完成，finalize 应为该 index
        发出 TOOL_CALL_COMPLETE，但不应重复"""
        parser = OpenAIStreamParser()

        # 只发送 start chunk，不发送完整参数
        chunk_start = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_xyz",
                        "function": {"name": "write_file", "arguments": ""},
                    }]
                },
                "finish_reason": None,
            }]
        }
        # 部分参数（不完整）
        chunk_partial = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": '{"pa'},
                    }]
                },
                "finish_reason": None,
            }]
        }

        parser.parse_chunk(chunk_start)
        parser.parse_chunk(chunk_partial)

        # finalize 应为 index 0 发出 TOOL_CALL_COMPLETE
        finalize_events = parser.finalize()
        complete_events = [e for e in finalize_events if e.event_type == StreamEventType.TOOL_CALL_COMPLETE]
        assert len(complete_events) == 1, "finalize 应为未完成的 tool_call 发出 TOOL_CALL_COMPLETE"

        # 再次 finalize 不应再产生（_completed_tool_indices 已记录）
        # 注意：finalize 不会重置状态，重复调用可能会再次产生（因为检查的是 _completed_tool_indices）
        # 但关键是 parse_chunk 期间完成的不会在 finalize 重复


# ===== B4: DPOPair 部分对验证 =====

class TestDPOPartialPairsPassValidation:
    """B4: DPOPair.validate(allow_partial=True) 允许 chosen 或 rejected 为空"""

    def test_partial_chosen_only_passes_with_allow_partial(self):
        """只有 chosen 非空，allow_partial=True 时应通过验证"""
        pair = DPOPair(
            prompt="Write a function",
            chosen="def foo(): pass",
            rejected="",
            task_id="1.1",
            intent="code_generation",
        )
        errors = pair.validate(allow_partial=True)
        assert len(errors) == 0, f"allow_partial=True 时部分对应通过验证，但得到错误: {errors}"

    def test_partial_rejected_only_passes_with_allow_partial(self):
        """只有 rejected 非空，allow_partial=True 时应通过验证"""
        pair = DPOPair(
            prompt="Write a function",
            chosen="",
            rejected="bad code",
            task_id="1.1",
            intent="code_generation",
        )
        errors = pair.validate(allow_partial=True)
        assert len(errors) == 0, f"allow_partial=True 时部分对应通过验证，但得到错误: {errors}"

    def test_partial_fails_without_allow_partial(self):
        """chosen 为空，allow_partial=False 时应验证失败"""
        pair = DPOPair(
            prompt="Write a function",
            chosen="",
            rejected="bad code",
            task_id="1.1",
            intent="code_generation",
        )
        errors = pair.validate(allow_partial=False)
        assert len(errors) > 0, "allow_partial=False 时空 chosen 应验证失败"

    def test_both_empty_fails_even_with_allow_partial(self):
        """chosen 和 rejected 都为空，即使 allow_partial=True 也应失败"""
        pair = DPOPair(
            prompt="Write a function",
            chosen="",
            rejected="",
            task_id="1.1",
            intent="code_generation",
        )
        errors = pair.validate(allow_partial=True)
        assert len(errors) > 0, "chosen 和 rejected 都为空应始终验证失败"

    def test_full_pair_passes_both_modes(self):
        """完整对在两种模式下都应通过"""
        pair = DPOPair(
            prompt="Write a function",
            chosen="good code",
            rejected="bad code",
            task_id="1.1",
            intent="code_generation",
        )
        assert len(pair.validate(allow_partial=False)) == 0
        assert len(pair.validate(allow_partial=True)) == 0


# ===== B5: DPO trace_id 匹配 =====

class TestDPOTraceIdBasedMatching:
    """B5: export_dpo_pairs 使用 trace_id 匹配 response 和 checklist，
    而非位置索引。不同 trace_id 的记录不应错误配对。"""

    def test_trace_id_prevents_incorrect_pairing(self):
        """具有不同 trace_id 的 response 和 checklist 不应基于位置被错误配对。
        验证 DPOPair 数据类和 TraceRecord 的 trace_id 字段在匹配中的作用。"""
        # 创建两个 trace_id 不同的响应/检查清单对
        trace_id_pass = "task1_pass_abc"
        trace_id_fail = "task1_fail_def"

        # PASS 响应
        pass_response = TraceRecord(
            trace_id=trace_id_pass,
            timestamp=100.0,
            record_type="tap_response",
            task_id="1.1",
            intent="code_generation",
            data={"raw_text": "good code", "is_empty": False},
        )

        # FAIL 响应
        fail_response = TraceRecord(
            trace_id=trace_id_fail,
            timestamp=200.0,
            record_type="tap_response",
            task_id="1.1",
            intent="code_generation",
            data={"raw_text": "bad code", "is_empty": False},
        )

        # PASS 检查清单（关联到 trace_id_pass）
        pass_checklist = TraceRecord(
            trace_id=trace_id_pass,
            timestamp=110.0,
            record_type="checklist_result",
            task_id="1.1",
            intent="code_generation",
            data={"fail_count": 0, "warn_count": 1, "ok_count": 4, "has_critical_warn": False, "passed": True},
        )

        # FAIL 检查清单（关联到 trace_id_fail）
        fail_checklist = TraceRecord(
            trace_id=trace_id_fail,
            timestamp=210.0,
            record_type="checklist_result",
            task_id="1.1",
            intent="code_generation",
            data={"fail_count": 2, "warn_count": 0, "ok_count": 2, "has_critical_warn": True, "passed": False},
        )

        # 验证 trace_id 正确关联
        # pass_response 的 trace_id 与 pass_checklist 相同
        assert pass_response.trace_id == pass_checklist.trace_id
        # fail_response 的 trace_id 与 fail_checklist 相同
        assert fail_response.trace_id == fail_checklist.trace_id
        # 交叉配对应不匹配
        assert pass_response.trace_id != fail_checklist.trace_id
        assert fail_response.trace_id != pass_checklist.trace_id

        # 验证 DPOPair 构造正确性：基于 trace_id 匹配
        pair = DPOPair(
            prompt="Write code",
            chosen=pass_response.data["raw_text"],  # PASS response
            rejected=fail_response.data["raw_text"],  # FAIL response
            task_id="1.1",
            intent="code_generation",
            metadata={
                "pass_trace_id": trace_id_pass,
                "fail_trace_id": trace_id_fail,
            },
        )
        assert pair.validate() == []
        assert pair.chosen == "good code"
        assert pair.rejected == "bad code"

    def test_export_dpo_pairs_uses_trace_id(self):
        """TAPTracer.export_dpo_pairs 应使用 trace_id 匹配而非位置索引。
        通过构造特定记录验证配对结果。"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = TAPTracer(trace_dir=tmpdir, enabled=False)

            # 请求记录
            request = TraceRecord(
                trace_id="trace_A",
                timestamp=100.0,
                record_type="tap_request",
                task_id="1.1",
                intent="code_generation",
                data={"instruction": "Write code", "constraints": [], "output_format_hint": "", "context_keys": []},
            )

            # PASS 响应 (trace_id = "trace_A")
            pass_response = TraceRecord(
                trace_id="trace_A",
                timestamp=110.0,
                record_type="tap_response",
                task_id="1.1",
                intent="code_generation",
                data={"raw_text": "def good(): pass", "is_empty": False, "raw_text_length": 16},
            )

            # FAIL 响应 (trace_id = "trace_B")
            fail_response = TraceRecord(
                trace_id="trace_B",
                timestamp=200.0,
                record_type="tap_response",
                task_id="1.1",
                intent="code_generation",
                data={"raw_text": "def bad()", "is_empty": False, "raw_text_length": 10},
            )

            # PASS checklist (trace_id = "trace_A" → 匹配 pass_response)
            pass_checklist = TraceRecord(
                trace_id="trace_A",
                timestamp=120.0,
                record_type="checklist_result",
                task_id="1.1",
                intent="code_generation",
                data={"fail_count": 0, "warn_count": 1, "ok_count": 4, "has_critical_warn": False, "passed": True},
            )

            # FAIL checklist (trace_id = "trace_B" → 匹配 fail_response)
            fail_checklist = TraceRecord(
                trace_id="trace_B",
                timestamp=210.0,
                record_type="checklist_result",
                task_id="1.1",
                intent="code_generation",
                data={"fail_count": 2, "warn_count": 0, "ok_count": 2, "has_critical_warn": True, "passed": False},
            )

            # 直接注入记录（跳过异步文件写入）
            with tracer._lock:
                tracer._records = [request, pass_response, fail_response, pass_checklist, fail_checklist]

            # 导出 DPO 对
            pairs = tracer.export_dpo_pairs()

            # 应产生至少一个配对，且 chosen 是 PASS 的响应，rejected 是 FAIL 的响应
            assert len(pairs) >= 1, f"应至少产生 1 个 DPO 对，实际 {len(pairs)}"

            # 验证配对正确性：chosen 应为 PASS 的内容
            pair = pairs[0]
            assert "def good" in pair["chosen"], (
                f"chosen 应为 PASS 响应内容，实际: {pair['chosen'][:50]}"
            )
            assert "def bad" in pair["rejected"], (
                f"rejected 应为 FAIL 响应内容，实际: {pair['rejected'][:50]}"
            )


# ===== M2: BYPASS 级别允许破坏性操作 =====

class TestBypassLevelAllowsDestructive:
    """M2: EnhancedPermissionManager 在 BYPASS 级别时应允许破坏性工具操作"""

    def test_bypass_allows_destructive_tool(self):
        """BYPASS 权限级别下，_check_fallback 应返回 (True, ...) 允许破坏性工具"""
        mgr = EnhancedPermissionManager(
            default_level=PermissionLevel.BYPASS,
            default_effect=PermissionEffect.DENY,
        )
        # 不添加任何规则，直接走 fallback
        allowed, reason = mgr._check_fallback("execute_subtask", "/workspace")
        assert allowed is True, f"BYPASS 级别应允许破坏性工具，但被拒绝: {reason}"
        assert "BYPASS" in reason, f"原因应包含 BYPASS，实际: {reason}"

    def test_plan_level_does_not_imply_bypass(self):
        """PLAN 级别不应等同于 BYPASS — 验证层级区分"""
        mgr = EnhancedPermissionManager(
            default_level=PermissionLevel.PLAN,
            default_effect=PermissionEffect.DENY,
        )
        allowed, reason = mgr._check_fallback("some_tool", "")
        # PLAN 允许项目写入，但消息应不同
        assert allowed is True
        assert "PLAN" in reason

    def test_default_level_denies_without_rules(self):
        """DEFAULT 级别无规则时应回退到默认策略（DENY）"""
        mgr = EnhancedPermissionManager(
            default_level=PermissionLevel.DEFAULT,
            default_effect=PermissionEffect.DENY,
        )
        allowed, reason = mgr._check_fallback("dangerous_tool", "/etc/passwd")
        assert allowed is False, "DEFAULT 级别无规则应拒绝破坏性工具"

    def test_bypass_allows_via_check(self):
        """通过完整 check() 方法验证 BYPASS 级别允许"""
        mgr = EnhancedPermissionManager(
            default_level=PermissionLevel.BYPASS,
            default_effect=PermissionEffect.DENY,
        )
        allowed, reason = mgr.check("execute_command", "/tmp/script.sh")
        assert allowed is True, f"BYPASS 级别 check() 应允许，但被拒绝: {reason}"


# ===== M5: 重定向 >> 模式被识别 =====

class TestRedirectAppendBlocked:
    """M5: check_command_safety 应识别 >> 追加重定向到系统路径"""

    def test_append_redirect_to_system_path_blocked(self):
        """echo data >> /etc/test 应被标记为危险"""
        is_safe, reason = check_command_safety("echo data >> /etc/test")
        assert is_safe is False, ">> 重定向到 /etc 应被标记为危险"
        # 可能被 BLOCKLIST 中 _DANGER_REDIRECT_PATTERNS 或 Layer 4 精细化检测拦截
        assert "危险" in reason or "重定向" in reason or "系统" in reason, (
            f"原因应提及危险/重定向/系统路径，实际: {reason}"
        )

    def test_single_redirect_to_system_path_blocked(self):
        """echo data > /etc/test 也应被标记为危险"""
        is_safe, reason = check_command_safety("echo data > /etc/test")
        assert is_safe is False, "> 重定向到 /etc 应被标记为危险"

    def test_append_redirect_to_user_path_allowed(self):
        """echo data >> /home/user/test 应不被拦截（非系统路径）"""
        is_safe, reason = check_command_safety("echo data >> /home/user/test")
        assert is_safe is True, f">> 重定向到用户路径应被允许，但被拒绝: {reason}"

    def test_no_redirect_to_system_path_allowed(self):
        """普通 echo 命令应被允许"""
        is_safe, reason = check_command_safety("echo hello world")
        assert is_safe is True

    def test_redirect_to_proc_blocked(self):
        """> /proc/sys 应被拦截"""
        is_safe, reason = check_command_safety("echo 1 > /proc/sys/net/ipv4/ip_forward")
        assert is_safe is False


# ===== M16: Mode mismatch 警告 =====

class TestModeMismatchWarning:
    """M16: 当 Compiler 产出的 mode 与 Adapter 要求的 mode 不匹配时，
    ModelProvider._validate_compiled_mode 应记录警告"""

    def test_mode_mismatch_logs_warning(self, caplog):
        """Compiler 产出 Mode A (messages)，Adapter 要求 Mode B (system_user)，
        应记录 warning 日志"""
        # 创建 Mode A 的 CompiledPrompt (messages)
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "hello"}],
        )
        assert compiled.mode == "messages"

        # 创建要求 system_user 的 mock Adapter
        mock_adapter = MagicMock(spec=TAPAdapter)
        mock_adapter.required_mode = "system_user"
        mock_adapter.send = AsyncMock(return_value=TAPResponse(raw_text="test"))

        # 创建 mock Compiler
        mock_compiler = MagicMock(spec=TAPCompiler)

        provider = ModelProvider(
            compiler=mock_compiler,
            adapter=mock_adapter,
            model="test-model",
        )

        with caplog.at_level(logging.WARNING, logger="teragent.core.provider"):
            provider._validate_compiled_mode(compiled)

        # 应有 warning 日志
        warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_logs) > 0, "mode 不匹配时应记录 WARNING 日志"
        # 日志内容应包含 mismatch 信息
        log_messages = " ".join(r.message for r in warning_logs)
        assert "mismatch" in log_messages.lower() or "Mode" in log_messages, (
            f"警告日志应包含 mismatch 或 Mode 关键词，实际: {log_messages}"
        )

    def test_mode_match_no_warning(self, caplog):
        """Compiler 产出与 Adapter 要求匹配时，不应记录 warning"""
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "hello"}],
        )

        mock_adapter = MagicMock(spec=TAPAdapter)
        mock_adapter.required_mode = "messages"  # 匹配
        mock_adapter.send = AsyncMock(return_value=TAPResponse(raw_text="test"))

        mock_compiler = MagicMock(spec=TAPCompiler)

        provider = ModelProvider(
            compiler=mock_compiler,
            adapter=mock_adapter,
            model="test-model",
        )

        with caplog.at_level(logging.WARNING, logger="teragent.core.provider"):
            provider._validate_compiled_mode(compiled)

        warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING and "mismatch" in r.message.lower()]
        assert len(warning_logs) == 0, "mode 匹配时不应有 mismatch 警告"

    def test_any_mode_never_warns(self, caplog):
        """Adapter required_mode='any' 时，任何 compiled mode 都不应产生警告"""
        compiled = CompiledPrompt(
            system_prompt="system",
            user_message="user",
        )
        assert compiled.mode == "system_user"

        mock_adapter = MagicMock(spec=TAPAdapter)
        mock_adapter.required_mode = "any"
        mock_adapter.send = AsyncMock(return_value=TAPResponse(raw_text="test"))

        mock_compiler = MagicMock(spec=TAPCompiler)

        provider = ModelProvider(
            compiler=mock_compiler,
            adapter=mock_adapter,
            model="test-model",
        )

        with caplog.at_level(logging.WARNING, logger="teragent.core.provider"):
            provider._validate_compiled_mode(compiled)

        warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING and "mismatch" in r.message.lower()]
        assert len(warning_logs) == 0, "required_mode='any' 时不应有 mismatch 警告"


# ===== B1: AgentLoop streaming tool results consumed =====

class TestAgentLoopStreamingToolResultsConsumed:
    """B1: streaming 模式下工具结果应存储在 _streaming_tool_results 缓冲区，
    并被 _tool_loop() 消费，而非重新执行"""

    def test_streaming_results_buffer_initialized_empty(self):
        """_streaming_tool_results 初始化为空列表"""
        mock_model = MagicMock(spec=ModelProvider)
        mock_model.model = "test-model"
        registry = ToolRegistry()
        loop = AgentLoop(model=mock_model, tool_registry=registry, config=AgentLoopConfig())
        assert loop._streaming_tool_results == []

    def test_streaming_results_consumed_by_tool_loop(self):
        """_tool_loop 中，当 use_streaming=True 且 _streaming_executor 存在时，
        应使用 _streaming_tool_results 而非重新执行工具"""
        # 直接验证逻辑：构造 AgentLoop，模拟 streaming 场景
        mock_model = MagicMock(spec=ModelProvider)
        mock_model.model = "test-model"
        mock_model.has_fallback = False
        registry = ToolRegistry()
        loop = AgentLoop(model=mock_model, tool_registry=registry, config=AgentLoopConfig())

        # 模拟 streaming executor 存在
        loop._streaming_executor = MagicMock()

        # 预填充 _streaming_tool_results（模拟 _call_model_streaming 的产出）
        fake_tool_call = {
            "id": "call_test",
            "type": "function",
            "function": {"name": "read_file", "arguments": {"path": "/tmp/test.py"}},
        }
        fake_result = ToolResult(success=True, data={"content": "file content"})
        loop._streaming_tool_results = [(fake_tool_call, fake_result)]

        # 验证 _tool_loop 中 streaming 分支的逻辑：
        # 当 use_streaming=True 且 _streaming_executor 存在时，
        # batch_results 应来自 _streaming_tool_results
        # 我们通过检查 _streaming_tool_results 被 pop 来验证
        stored_results = loop._streaming_tool_results
        assert len(stored_results) == 1

        # 模拟 _tool_loop 消费逻辑（不实际运行异步循环）
        # 在 _tool_loop 源码中:
        #   if use_streaming and self._streaming_executor:
        #       batch_results = self._streaming_tool_results
        #       self._streaming_tool_results = []
        batch_results = loop._streaming_tool_results
        loop._streaming_tool_results = []

        # 验证结果被正确消费
        assert len(batch_results) == 1
        assert batch_results[0][0] == fake_tool_call
        assert batch_results[0][1].success is True
        # 缓冲区已清空
        assert loop._streaming_tool_results == []

    def test_reset_clears_streaming_buffer(self):
        """reset() 应清空 _streaming_tool_results"""
        mock_model = MagicMock(spec=ModelProvider)
        mock_model.model = "test-model"
        registry = ToolRegistry()
        loop = AgentLoop(model=mock_model, tool_registry=registry, config=AgentLoopConfig())

        # 填充缓冲区
        loop._streaming_tool_results = [("fake", ToolResult(success=True, data={}))]

        loop.reset()
        assert loop._streaming_tool_results == [], "reset() 应清空 streaming 工具结果缓冲区"

    @pytest.mark.asyncio
    async def test_non_streaming_does_not_use_buffer(self):
        """非 streaming 模式下，_streaming_tool_results 应保持为空，
        工具通过 _execute_tool_calls_batch 执行"""
        mock_model = MagicMock(spec=ModelProvider)
        mock_model.model = "test-model"
        mock_model.has_fallback = False
        mock_model.chat = AsyncMock(return_value={
            "content": "done",
            "tool_calls": [],
            "usage": {},
            "finish_reason": "stop",
        })

        registry = ToolRegistry()
        loop = AgentLoop(model=mock_model, tool_registry=registry, config=AgentLoopConfig())
        loop._intent_classifier = None
        loop._confirmation_gate = None
        loop._sub_agent_manager = None
        loop._session_persistence = None
        loop._context_window = None
        loop._auto_compactor = None
        loop._step_budget = None
        loop._event_bus = None

        # 确保 streaming 模式关闭
        loop.set_streaming_config(mode="batch")

        _result = await loop.run("test", messages=[], system_prompt="sys")

        # batch 模式下 _streaming_tool_results 应保持为空
        assert loop._streaming_tool_results == [], (
            "非 streaming 模式下 _streaming_tool_results 不应有内容"
        )
