# tests/test_firecracker_sandbox.py
"""Firecracker 沙箱单元测试

覆盖:
  - _generate_config: shlex.quote 注入防护 (M6 修复验证)
  - _prepare_rootfs: 使用 workspace_root (M7 修复验证)
  - _check_prerequisites: 预检检查
  - 降级逻辑: Firecracker→Docker→subprocess
  - _capture_output: 输出捕获
"""
import os
import shlex
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from teragent.security.firecracker_sandbox import FirecrackerSandbox
from teragent.security.sandbox import execute_in_sandbox


# ===== _generate_config (M6: shlex.quote) =====

class TestGenerateConfig:
    """配置生成 — 验证 M6 修复 (shlex.quote)"""

    def test_command_is_base64_encoded(self):
        """命令通过 base64 编码（防命令注入）"""
        import base64
        sandbox = FirecrackerSandbox(workspace_root="/workspace")
        config = sandbox._generate_config("rm -rf /", "/fake/rootfs.ext4")
        boot_args = config["boot-source"]["boot_args"]
        # 命令应通过 base64 编码传递
        expected_b64 = base64.b64encode("rm -rf /".encode()).decode()
        assert f"agentd_cmd_b64={expected_b64}" in boot_args

    def test_command_with_shell_injection_is_base64_encoded(self):
        """含 shell 注入的命令通过 base64 安全编码"""
        import base64
        sandbox = FirecrackerSandbox(workspace_root="/workspace")
        malicious = 'echo hello; rm -rf /'
        config = sandbox._generate_config(malicious, "/fake/rootfs.ext4")
        boot_args = config["boot-source"]["boot_args"]
        expected_b64 = base64.b64encode(malicious.encode()).decode()
        assert f"agentd_cmd_b64={expected_b64}" in boot_args
        # 确保原始未编码命令不在 boot_args 中
        assert malicious not in boot_args

    def test_config_structure(self):
        """配置结构完整"""
        sandbox = FirecrackerSandbox(workspace_root="/workspace")
        config = sandbox._generate_config("ls", "/fake/rootfs.ext4")
        assert "boot-source" in config
        assert "drives" in config
        assert "machine-config" in config
        assert config["machine-config"]["vcpu_count"] == 1
        assert config["machine-config"]["mem_size_mib"] == 128
        assert config["drives"][0]["path_on_host"] == "/fake/rootfs.ext4"

    def test_custom_resource_config(self):
        """自定义资源配置"""
        sandbox = FirecrackerSandbox(
            workspace_root="/workspace",
            vcpu_count=4,
            mem_size_mib=512,
        )
        config = sandbox._generate_config("ls", "/fake/rootfs.ext4")
        assert config["machine-config"]["vcpu_count"] == 4
        assert config["machine-config"]["mem_size_mib"] == 512


# ===== _prepare_rootfs (M7: workspace_root) =====

class TestPrepareRootfs:
    """Rootfs 准备 — 验证 M7 修复 (使用 workspace_root)"""

    def test_rootfs_path_uses_workspace_root(self, tmp_path):
        """rootfs 路径基于 workspace_root（M7 修复）"""
        sandbox = FirecrackerSandbox(workspace_root=str(tmp_path))
        # 创建 rootfs 文件
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        (agent_dir / "rootfs.ext4").write_bytes(b"\x00" * 1024)
        rootfs_path = sandbox._prepare_rootfs()
        assert rootfs_path.startswith(str(tmp_path))
        assert rootfs_path == str(tmp_path / ".agent" / "rootfs.ext4")

    def test_rootfs_missing_raises_error(self, tmp_path):
        """rootfs 不存在时抛出异常"""
        sandbox = FirecrackerSandbox(workspace_root=str(tmp_path))
        with pytest.raises(RuntimeError, match="Rootfs image not found"):
            sandbox._prepare_rootfs()


# ===== _check_prerequisites =====

class TestCheckPrerequisites:
    """预检检查"""

    def test_kvm_missing_raises_error(self):
        """KVM 不可用抛出异常"""
        sandbox = FirecrackerSandbox(workspace_root="/workspace")
        with patch("os.path.exists", return_value=False):
            with pytest.raises(RuntimeError, match="KVM not available"):
                sandbox._check_prerequisites()

    def test_kernel_missing_raises_error(self):
        """内核镜像缺失抛出异常"""
        sandbox = FirecrackerSandbox(workspace_root="/workspace")
        # KVM 存在、firecracker/jailer 都找到，但内核不存在
        with patch("os.path.exists", side_effect=lambda p: p == "/dev/kvm"):
            with patch("os.path.isfile", side_effect=lambda p: p != sandbox.kernel_path):
                with patch("shutil.which", return_value="/usr/bin/found"):
                    with pytest.raises(RuntimeError, match="Kernel image not found"):
                        sandbox._check_prerequisites()


# ===== 降级逻辑 =====

class TestSandboxDegradation:
    """沙箱降级逻辑: Firecracker→Docker→subprocess"""

    @pytest.mark.asyncio
    async def test_firecracker_runtime_error_falls_back(self, tmp_path):
        """Firecracker RuntimeError 时降级到 Level 0"""
        mock_fc_instance = MagicMock()
        mock_fc_instance.run = AsyncMock(side_effect=RuntimeError("KVM not available"))

        with patch("teragent.security.sandbox.check_command_safety", return_value=(True, "")):
            with patch("teragent.security.firecracker_sandbox.FirecrackerSandbox", return_value=mock_fc_instance):
                # 没有 docker，降级到 Level 0
                with patch("shutil.which", side_effect=lambda name: None if name == "docker" else "/usr/bin/echo"):
                    with patch("asyncio.create_subprocess_exec") as mock_exec:
                        mock_process = MagicMock()
                        mock_process.communicate = AsyncMock(return_value=(b"output", b""))
                        mock_process.returncode = 0
                        mock_exec.return_value = mock_process
                        code, output = await execute_in_sandbox(
                            "echo hello", str(tmp_path), level=2, timeout=5
                        )
                        # 应降级到 Level 0 执行
                        assert code == 0


# ===== _capture_output =====

class TestCaptureOutput:
    """输出捕获"""

    def test_stdout_only(self):
        """仅 stdout"""
        result = FirecrackerSandbox._capture_output(b"hello", b"")
        assert "hello" in result

    def test_stdout_and_stderr(self):
        """stdout + stderr"""
        result = FirecrackerSandbox._capture_output(b"output", b"daemon log")
        assert "output" in result
        assert "Firecracker log" in result

    def test_empty_output(self):
        """空输出"""
        result = FirecrackerSandbox._capture_output(b"", b"")
        assert result == ""
