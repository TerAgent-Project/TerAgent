# tests/test_subagent_worker.py
"""SubAgent Worker 单元测试

测试 teragent.pipeline.subagent_worker 模块:
  - 命令提取
  - 危险命令检测
  - 完整执行流程
"""
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from teragent.pipeline.subagent_worker import (
    SubAgentWorker,
    extract_commands_from_response,
    is_dangerous_command,
)


class TestExtractCommands:
    """extract_commands_from_response 命令提取测试"""

    def test_single_command(self):
        """提取单个命令"""
        content = '<command cwd="/home">pip install requests</command>'
        cmds = extract_commands_from_response(content)
        assert len(cmds) == 1
        assert cmds[0]["cwd"] == "/home"
        assert cmds[0]["command"] == "pip install requests"

    def test_multiple_commands(self):
        """提取多个命令"""
        content = """
        <command cwd=".">pip install flask</command>
        <command cwd="src">python main.py</command>
        """
        cmds = extract_commands_from_response(content)
        assert len(cmds) == 2

    def test_single_quote_cwd(self):
        """单引号 cwd 属性"""
        content = "<command cwd='/tmp'>ls</command>"
        cmds = extract_commands_from_response(content)
        assert len(cmds) == 1
        assert cmds[0]["cwd"] == "/tmp"

    def test_no_commands(self):
        """无命令标签返回空列表"""
        content = "<file path='main.py'>print('hello')</file>"
        cmds = extract_commands_from_response(content)
        assert len(cmds) == 0

    def test_empty_content(self):
        """空内容返回空列表"""
        assert extract_commands_from_response("") == []


class TestIsDangerousCommand:
    """is_dangerous_command 危险命令检测测试"""

    def test_rm_rf_is_dangerous(self):
        """rm -rf 是危险命令"""
        assert is_dangerous_command("rm -rf /") is True

    def test_sudo_is_dangerous(self):
        """sudo 是危险命令"""
        assert is_dangerous_command("sudo apt-get install") is True

    def test_mkfs_is_dangerous(self):
        """mkfs 是危险命令"""
        assert is_dangerous_command("mkfs.ext4 /dev/sda1") is True

    def test_safe_command(self):
        """安全命令返回 False"""
        assert is_dangerous_command("ls -la") is False
        assert is_dangerous_command("python main.py") is False

    def test_pip_install_not_dangerous(self):
        """pip install 是 WARNING 级别，需要权限检查"""
        assert is_dangerous_command("pip install requests") is True


class TestSubAgentWorkerInit:
    """SubAgentWorker 初始化测试"""

    def test_init_attributes(self):
        """初始化属性正确"""
        from teragent.core.provider import ModelProvider
        mock_model = MagicMock(spec=ModelProvider)
        worker = SubAgentWorker(
            task_id="1.1",
            design_md="design",
            plan_md="plan",
            task_desc="implement feature",
            code_summary="summary",
            model=mock_model,
            workspace_root="/tmp/project",
        )
        assert worker.task_id == "1.1"
        assert worker.design_md == "design"
        assert worker.workspace_root == "/tmp/project"


class TestSubAgentWorkerExecute:
    """SubAgentWorker 执行流程测试"""

    @pytest.mark.asyncio
    async def test_execute_empty_response_returns_error(self):
        """模型返回空内容时报错"""
        from teragent.core.provider import ModelProvider
        from teragent.core.tap import TAPResponse

        mock_model = AsyncMock(spec=ModelProvider)
        mock_model.execute_tap = AsyncMock(return_value=TAPResponse(raw_text=""))
        mock_model._tracer = None

        worker = SubAgentWorker(
            task_id="1.1",
            design_md="",
            plan_md="",
            task_desc="test task",
            code_summary="",
            model=mock_model,
            workspace_root="/tmp/test_project",
        )
        result = await worker.execute()
        assert result["task_id"] == "1.1"
        assert result["error"] is not None
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_with_file_output(self):
        """模型返回文件内容时提取并写入"""
        import tempfile

        from teragent.core.provider import ModelProvider
        from teragent.core.tap import TAPResponse

        mock_model = AsyncMock(spec=ModelProvider)
        response_text = '<file path="hello.py">print("hello")</file>'
        mock_model.execute_tap = AsyncMock(
            return_value=TAPResponse(raw_text=response_text, usage={})
        )
        mock_model._tracer = None

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = SubAgentWorker(
                task_id="1.1",
                design_md="",
                plan_md="",
                task_desc="write hello.py",
                code_summary="",
                model=mock_model,
                workspace_root=tmpdir,
            )
            result = await worker.execute()
            assert result["task_id"] == "1.1"

    @pytest.mark.asyncio
    async def test_execute_with_dangerous_command(self):
        """执行包含危险命令时返回权限拒绝"""
        import tempfile

        from teragent.core.provider import ModelProvider
        from teragent.core.tap import TAPResponse

        mock_model = AsyncMock(spec=ModelProvider)
        response_text = '<command cwd=".">rm -rf /</command>'
        mock_model.execute_tap = AsyncMock(
            return_value=TAPResponse(raw_text=response_text, usage={})
        )
        mock_model._tracer = None

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = SubAgentWorker(
                task_id="1.1",
                design_md="",
                plan_md="",
                task_desc="dangerous task",
                code_summary="",
                model=mock_model,
                workspace_root=tmpdir,
            )
            result = await worker.execute()
            assert result.get("error") == "PERMISSION_DENIED"

    @pytest.mark.asyncio
    async def test_execute_model_exception(self):
        """模型调用异常时返回错误"""
        from teragent.core.provider import ModelProvider

        mock_model = AsyncMock(spec=ModelProvider)
        mock_model.execute_tap = AsyncMock(side_effect=RuntimeError("API down"))
        mock_model._tracer = None

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = SubAgentWorker(
                task_id="1.1",
                design_md="",
                plan_md="",
                task_desc="test task",
                code_summary="",
                model=mock_model,
                workspace_root=tmpdir,
            )
            result = await worker.execute()
            assert result["error"] is not None


class TestSubAgentWorkerTracer:
    """SubAgentWorker tracer 集成测试"""

    def test_get_tracer_explicit(self):
        """显式设置 tracer"""
        from teragent.core.provider import ModelProvider
        mock_model = MagicMock(spec=ModelProvider)
        mock_tracer = MagicMock()
        worker = SubAgentWorker(
            task_id="1.1",
            design_md="", plan_md="", task_desc="", code_summary="",
            model=mock_model,
            workspace_root="/tmp",
            tracer=mock_tracer,
        )
        assert worker._get_tracer() is mock_tracer

    def test_get_tracer_from_model(self):
        """从 ModelProvider 获取 tracer"""
        from teragent.core.provider import ModelProvider
        mock_model = MagicMock(spec=ModelProvider)
        mock_tracer = MagicMock()
        mock_model._tracer = mock_tracer
        worker = SubAgentWorker(
            task_id="1.1",
            design_md="", plan_md="", task_desc="", code_summary="",
            model=mock_model,
            workspace_root="/tmp",
        )
        assert worker._get_tracer() is mock_tracer

    def test_get_tracer_none(self):
        """无 tracer 返回 None"""
        from teragent.core.provider import ModelProvider
        mock_model = MagicMock(spec=ModelProvider)
        mock_model._tracer = None
        worker = SubAgentWorker(
            task_id="1.1",
            design_md="", plan_md="", task_desc="", code_summary="",
            model=mock_model,
            workspace_root="/tmp",
        )
        assert worker._get_tracer() is None
