class AgentError(Exception):
    """所有 Agent 业务异常的基类"""
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)

class PlanParseError(AgentError):
    """当 PLAN.md 格式不符合正则解析要求时抛出"""
    pass

class SandboxViolation(AgentError):
    """当文件写入路径穿越或执行黑名单命令时抛出"""
    pass

class PermissionDenied(AgentError):
    """当权限不足以执行请求的操作时抛出"""
    pass

class ModelUnavailableError(AgentError):
    """当 LLM API 不可用或超时时抛出"""
    def __init__(self, message: str, provider: str = "", status_code: int = 0) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code

class ContextWindowExceededError(AgentError):
    """当 TAP 请求的上下文超过模型限制时抛出"""
    def __init__(self, message: str, estimated_tokens: int = 0, max_tokens: int = 0) -> None:
        super().__init__(message)
        self.estimated_tokens = estimated_tokens
        self.max_tokens = max_tokens

class ReplanMeltdownError(AgentError):
    """当重规划熔断触发时抛出（连续重规划超过阈值）"""
    def __init__(self, message: str, group_id: str = "", attempts: int = 0) -> None:
        super().__init__(message)
        self.group_id = group_id
        self.attempts = attempts

class DependencyExplosionError(AgentError):
    """当依赖分析报告超出 Token 预算时抛出"""
    def __init__(self, message: str, budget: int = 0, actual: int = 0) -> None:
        super().__init__(message)
        self.budget = budget
        self.actual = actual

class PipelineStateError(AgentError):
    """当流水线状态机收到不合法的状态转换时抛出"""
    def __init__(self, message: str, current_state: str = "", target_state: str = "") -> None:
        super().__init__(message)
        self.current_state = current_state
        self.target_state = target_state
