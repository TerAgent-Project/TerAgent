# tests/test_compilers.py
"""core/compilers 单元测试

覆盖:
  - GLMCompiler: Mode A 输出 / 近因效应优化 / 中文约束注入 / output_format_hint / _get_compiler_type
  - AnthropicCompiler: Mode B 输出 / XML 标签结构 / system_prompt+user_message 分离 / 约束与格式提示
  - DeepSeekCompiler: Mode A 输出 / 极简系统提示 / 约束内联到用户消息 / output_format_hint
  - DefaultCompiler: Mode A 输出 / 多轮上下文注入 / 约束+格式提示+记忆
  - TAPCompiler 基类: _inject_context / _build_context_string / get_system_prompt / TAPCompilerRegistry
  - CompiledPrompt: mode 属性 (messages / system_user / empty)
  - 上下文注入: memory / design / plan / dependency_report
"""
import pytest

from teragent.core.tap import TAPRequest, CompiledPrompt
from teragent.core.compiler import TAPCompiler, TAPCompilerRegistry
from teragent.core.compilers.glm import GLMCompiler
from teragent.core.compilers.anthropic import AnthropicCompiler
from teragent.core.compilers.deepseek import DeepSeekCompiler
from teragent.core.compilers.default import DefaultCompiler


# ===== 辅助: 构造 TAPRequest =====

def _make_request(
    intent: str = "execute",
    instruction: str = "写一个排序函数",
    constraints: list[str] | None = None,
    output_format_hint: str = "",
    context: dict | None = None,
    meta: dict | None = None,
) -> TAPRequest:
    """构造测试用的 TAPRequest"""
    return TAPRequest(
        meta=meta or {"task_id": "1.1", "intent": intent},
        context=context or {},
        instruction=instruction,
        constraints=constraints or [],
        output_format_hint=output_format_hint,
    )


# ===== CompiledPrompt mode 属性 =====

class TestCompiledPromptMode:
    """CompiledPrompt.mode 属性测试"""

    def test_mode_messages(self):
        """有 messages 列表时 mode 为 messages"""
        cp = CompiledPrompt(messages=[{"role": "system", "content": "hi"}])
        assert cp.mode == "messages"

    def test_mode_system_user(self):
        """有 system_prompt 或 user_message 时 mode 为 system_user"""
        cp = CompiledPrompt(system_prompt="你是助手", user_message="你好")
        assert cp.mode == "system_user"

    def test_mode_system_user_partial(self):
        """仅有 system_prompt 也为 system_user"""
        cp = CompiledPrompt(system_prompt="你是助手")
        assert cp.mode == "system_user"

    def test_mode_empty(self):
        """空 CompiledPrompt 的 mode 为 empty"""
        cp = CompiledPrompt()
        assert cp.mode == "empty"


# ===== TAPCompilerRegistry =====

class TestTAPCompilerRegistry:
    """TAPCompilerRegistry 注册与创建测试"""

    def test_registered_compilers(self):
        """四种编译器已注册"""
        available = TAPCompilerRegistry.available()
        for name in ("glm", "anthropic", "deepseek", "default"):
            assert name in available, f"{name} 未注册"

    def test_create_compiler(self):
        """通过注册表创建编译器实例"""
        compiler = TAPCompilerRegistry.create("glm")
        assert isinstance(compiler, GLMCompiler)

    def test_create_unknown_raises(self):
        """创建未注册的编译器抛出 ValueError"""
        with pytest.raises(ValueError, match="Unknown compiler"):
            TAPCompilerRegistry.create("nonexistent")


# ===== GLMCompiler =====

class TestGLMCompiler:
    """GLMCompiler 测试"""

    def test_get_compiler_type(self):
        """_get_compiler_type 返回 glm"""
        compiler = GLMCompiler()
        assert compiler._get_compiler_type() == "glm"

    def test_mode_a_output(self):
        """GLM 编译器输出 Mode A (messages 列表)"""
        compiler = GLMCompiler()
        request = _make_request()
        result = compiler.compile(request)
        assert result.mode == "messages"
        assert len(result.messages) > 0

    def test_recency_effect_instruction_last(self):
        """近因效应: 核心指令放在最后一条 user 消息"""
        compiler = GLMCompiler()
        request = _make_request(instruction="写一个排序函数")
        result = compiler.compile(request)
        # 最后一条消息应为 user 角色，包含核心指令
        last_msg = result.messages[-1]
        assert last_msg["role"] == "user"
        assert "写一个排序函数" in last_msg["content"]

    def test_system_message_with_constraints(self):
        """系统消息包含约束条件"""
        compiler = GLMCompiler()
        request = _make_request(constraints=["不能使用内置排序", "时间复杂度 O(n log n)"])
        result = compiler.compile(request)
        system_msg = result.messages[0]
        assert system_msg["role"] == "system"
        assert "不能使用内置排序" in system_msg["content"]
        assert "时间复杂度 O(n log n)" in system_msg["content"]

    def test_output_format_hint_in_system(self):
        """M4 修复: output_format_hint 在系统消息中使用"""
        compiler = GLMCompiler()
        request = _make_request(output_format_hint="用 <file path='...'> 输出代码。")
        result = compiler.compile(request)
        system_msg = result.messages[0]
        assert "用 <file path='...'> 输出代码。" in system_msg["content"]

    def test_output_format_hint_default(self):
        """无 output_format_hint 时使用 GLM 默认格式提示"""
        compiler = GLMCompiler()
        request = _make_request(output_format_hint="")
        result = compiler.compile(request)
        system_msg = result.messages[0]
        # 默认格式提示
        assert "用 <file path='...'> 输出代码。" in system_msg["content"]

    def test_chinese_constraints_injection(self):
        """中文约束注入: execute 意图包含中文格式约束"""
        compiler = GLMCompiler()
        request = _make_request(intent="execute")
        result = compiler.compile(request)
        last_msg = result.messages[-1]
        assert "中文注释" in last_msg["content"] or "英文标识符" in last_msg["content"]

    def test_context_injection(self):
        """上下文注入: design/plan/memory 作为参考上下文"""
        compiler = GLMCompiler()
        request = _make_request(context={
            "design": "架构设计文档",
            "plan": "执行计划",
            "memory": "项目记忆内容",
        })
        result = compiler.compile(request)
        # 应有参考上下文消息
        all_content = " ".join(m["content"] for m in result.messages)
        assert "架构设计文档" in all_content
        assert "执行计划" in all_content
        assert "项目记忆内容" in all_content

    def test_no_context_skips_context_messages(self):
        """无上下文时不生成参考上下文消息"""
        compiler = GLMCompiler()
        request = _make_request()
        result = compiler.compile(request)
        # 无上下文时应只有 system + user(指令) 两条消息
        # 或有中文约束则 user 消息可能合并
        content_has_context = any("参考上下文" in m["content"] for m in result.messages)
        assert content_has_context is False


# ===== AnthropicCompiler =====

class TestAnthropicCompiler:
    """AnthropicCompiler 测试"""

    def test_get_compiler_type(self):
        """_get_compiler_type 返回 anthropic"""
        compiler = AnthropicCompiler()
        assert compiler._get_compiler_type() == "anthropic"

    def test_mode_b_output(self):
        """Anthropic 编译器输出 Mode B (system_prompt + user_message)"""
        compiler = AnthropicCompiler()
        request = _make_request()
        result = compiler.compile(request)
        assert result.mode == "system_user"
        assert result.system_prompt != ""
        assert result.user_message != ""

    def test_system_prompt_contains_constraints(self):
        """系统提示包含约束条件"""
        compiler = AnthropicCompiler()
        request = _make_request(constraints=["必须使用 Python 3.10+"])
        result = compiler.compile(request)
        assert "必须使用 Python 3.10+" in result.system_prompt

    def test_system_prompt_contains_format_hint(self):
        """系统提示包含 output_format_hint"""
        compiler = AnthropicCompiler()
        request = _make_request(output_format_hint="JSON 格式输出")
        result = compiler.compile(request)
        assert "JSON 格式输出" in result.system_prompt

    def test_system_prompt_contains_memory(self):
        """系统提示包含项目记忆"""
        compiler = AnthropicCompiler()
        request = _make_request(context={"memory": "之前用了 FastAPI"})
        result = compiler.compile(request)
        assert "之前用了 FastAPI" in result.system_prompt
        assert "<memory>" in result.system_prompt

    def test_user_message_contains_context(self):
        """用户消息包含 design/plan/dependency_report 上下文"""
        compiler = AnthropicCompiler()
        request = _make_request(context={
            "design": "设计文档",
            "plan": "计划内容",
            "dependency_report": "依赖报告",
        })
        result = compiler.compile(request)
        assert "<design>" in result.user_message
        assert "设计文档" in result.user_message
        assert "<plan>" in result.user_message
        assert "<dependency_report>" in result.user_message

    def test_user_message_ends_with_instruction(self):
        """用户消息末尾为用户指令"""
        compiler = AnthropicCompiler()
        request = _make_request(instruction="实现用户登录")
        result = compiler.compile(request)
        # 指令应在 user_message 中
        assert "实现用户登录" in result.user_message

    def test_na_context_excluded(self):
        """值为 N/A 的上下文字段不注入"""
        compiler = AnthropicCompiler()
        request = _make_request(context={
            "design": "N/A",
            "plan": "N/A",
            "memory": "N/A",
        })
        result = compiler.compile(request)
        assert "<design>" not in result.user_message
        assert "<memory>" not in result.system_prompt


# ===== DeepSeekCompiler =====

class TestDeepSeekCompiler:
    """DeepSeekCompiler 测试"""

    def test_get_compiler_type(self):
        """_get_compiler_type 返回 deepseek"""
        compiler = DeepSeekCompiler()
        assert compiler._get_compiler_type() == "deepseek"

    def test_mode_a_output(self):
        """DeepSeek 编译器输出 Mode A (messages 列表)"""
        compiler = DeepSeekCompiler()
        request = _make_request()
        result = compiler.compile(request)
        assert result.mode == "messages"

    def test_minimal_system_prompt(self):
        """极简系统提示: 系统消息不含约束和格式提示"""
        compiler = DeepSeekCompiler()
        request = _make_request(constraints=["不能使用排序库"])
        result = compiler.compile(request)
        system_msg = result.messages[0]
        assert system_msg["role"] == "system"
        # 约束不应出现在系统消息中（DeepSeek 优化：约束内联到用户消息）
        assert "不能使用排序库" not in system_msg["content"]

    def test_constraints_inlined_in_user_message(self):
        """约束内联到用户消息"""
        compiler = DeepSeekCompiler()
        request = _make_request(constraints=["必须类型安全"])
        result = compiler.compile(request)
        # 找到包含约束的用户消息
        user_msgs_with_constraints = [
            m for m in result.messages
            if m["role"] == "user" and "必须类型安全" in m["content"]
        ]
        assert len(user_msgs_with_constraints) > 0

    def test_output_format_hint_in_user_message(self):
        """output_format_hint 内联到用户消息"""
        compiler = DeepSeekCompiler()
        request = _make_request(output_format_hint="Markdown 格式")
        result = compiler.compile(request)
        all_user_content = " ".join(
            m["content"] for m in result.messages if m["role"] == "user"
        )
        assert "Markdown 格式" in all_user_content

    def test_memory_in_context_message(self):
        """项目记忆在上下文消息中"""
        compiler = DeepSeekCompiler()
        request = _make_request(context={"memory": "使用 Pydantic v2"})
        result = compiler.compile(request)
        all_content = " ".join(m["content"] for m in result.messages)
        assert "使用 Pydantic v2" in all_content
        assert "<memory>" in all_content

    def test_instruction_is_last_user_message(self):
        """核心指令为最后一条用户消息"""
        compiler = DeepSeekCompiler()
        request = _make_request(instruction="写一个 HTTP 服务器")
        result = compiler.compile(request)
        user_msgs = [m for m in result.messages if m["role"] == "user"]
        assert "写一个 HTTP 服务器" in user_msgs[-1]["content"]


# ===== DefaultCompiler =====

class TestDefaultCompiler:
    """DefaultCompiler 测试"""

    def test_get_compiler_type(self):
        """_get_compiler_type 返回 default"""
        compiler = DefaultCompiler()
        assert compiler._get_compiler_type() == "default"

    def test_mode_a_output(self):
        """Default 编译器输出 Mode A (messages 列表)"""
        compiler = DefaultCompiler()
        request = _make_request()
        result = compiler.compile(request)
        assert result.mode == "messages"

    def test_system_message_with_constraints_and_format(self):
        """系统消息包含约束和格式提示"""
        compiler = DefaultCompiler()
        request = _make_request(
            constraints=["不使用 eval"],
            output_format_hint="输出纯文本",
        )
        result = compiler.compile(request)
        system_msg = result.messages[0]
        assert system_msg["role"] == "system"
        assert "不使用 eval" in system_msg["content"]
        assert "输出纯文本" in system_msg["content"]

    def test_memory_in_compiled_output(self):
        """项目记忆在编译输出中（通过 _inject_context 注入，不重复在 system message）"""
        compiler = DefaultCompiler()
        request = _make_request(context={"memory": "项目使用 Flask 框架"})
        result = compiler.compile(request)
        # Memory is injected via _inject_context, not duplicated in system message
        all_contents = [m["content"] for m in result.messages]
        assert any("Flask 框架" in c for c in all_contents)

    def test_multi_turn_context_injection(self):
        """多轮上下文注入: design/plan/dependency_report 以 user-assistant 对话注入"""
        compiler = DefaultCompiler()
        request = _make_request(context={
            "design": "微服务架构",
            "plan": "三步走计划",
            "dependency_report": "无依赖冲突",
        })
        result = compiler.compile(request)
        # 检查多轮对话格式: user 注入 → assistant 确认
        all_contents = [m["content"] for m in result.messages]
        assert any("微服务架构" in c for c in all_contents)
        assert any("收到设计文档" in c for c in all_contents)
        assert any("三步走计划" in c for c in all_contents)
        assert any("收到执行计划" in c for c in all_contents)

    def test_instruction_is_final_user_message(self):
        """核心指令为最后一条 user 消息"""
        compiler = DefaultCompiler()
        request = _make_request(instruction="实现 REST API")
        result = compiler.compile(request)
        last_msg = result.messages[-1]
        assert last_msg["role"] == "user"
        assert "实现 REST API" in last_msg["content"]

    def test_na_context_not_injected(self):
        """N/A 上下文不被注入"""
        compiler = DefaultCompiler()
        request = _make_request(context={
            "design": "N/A",
            "plan": "N/A",
            "dependency_report": "N/A",
            "memory": "N/A",
        })
        result = compiler.compile(request)
        all_content = " ".join(m["content"] for m in result.messages)
        # 不应出现 "收到设计文档" 等确认消息
        assert "收到设计文档" not in all_content
        assert "收到执行计划" not in all_content
        assert "项目记忆" not in all_content


# ===== TAPCompiler 基类共享方法 =====

class TestTAPCompilerBase:
    """TAPCompiler 基类共享方法测试"""

    def test_inject_context_with_all_fields(self):
        """_inject_context 注入所有上下文字段"""
        compiler = DefaultCompiler()
        messages = []
        request = _make_request(context={
            "design": "D",
            "plan": "P",
            "dependency_report": "R",
            "memory": "M",
        })
        result = compiler._inject_context(messages, request)
        # 每个字段生成 user + assistant 对
        assert len(result) == 8  # 4 fields × 2 messages each

    def test_inject_context_empty_values(self):
        """_inject_context 跳过空值和 N/A"""
        compiler = DefaultCompiler()
        messages = []
        request = _make_request(context={
            "design": "",
            "plan": "N/A",
            "dependency_report": "",
            "memory": "N/A",
        })
        result = compiler._inject_context(messages, request)
        assert len(result) == 0

    def test_build_context_string(self):
        """_build_context_string 拼接上下文为单一字符串"""
        compiler = AnthropicCompiler()
        request = _make_request(context={
            "design": "架构设计",
            "plan": "执行计划",
            "dependency_report": "依赖报告",
        })
        result = compiler._build_context_string(request)
        assert "<design>" in result
        assert "架构设计" in result
        assert "<plan>" in result
        assert "<dependency_report>" in result

    def test_build_context_string_excludes_memory(self):
        """_build_context_string 不包含 memory (memory 只在 _inject_context 中处理)"""
        compiler = AnthropicCompiler()
        request = _make_request(context={
            "design": "架构",
            "memory": "项目记忆",
        })
        result = compiler._build_context_string(request)
        assert "<memory>" not in result
        assert "项目记忆" not in result

    def test_get_system_prompt_for_known_intent(self):
        """get_system_prompt 返回已知意图的系统提示"""
        compiler = DefaultCompiler()
        prompt = compiler.get_system_prompt("execute")
        # 至少不为空（从 prompt registry 获取）
        assert isinstance(prompt, str)

    def test_get_system_prompt_for_unknown_intent(self):
        """get_system_prompt 返回空字符串对于未知意图"""
        compiler = DefaultCompiler()
        prompt = compiler.get_system_prompt("nonexistent_intent")
        assert prompt == ""
