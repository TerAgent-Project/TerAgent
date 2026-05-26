# teragent/security/firecracker_sandbox.py
"""Firecracker microVM 沙箱 — Level 2 命令隔离
"""
import asyncio
import base64
import json
import logging
import os
import shutil
import signal
import time

logger = logging.getLogger(__name__)

# Default resource configuration
DEFAULT_VCPU_COUNT = 1
DEFAULT_MEM_SIZE_MIB = 128
DEFAULT_TIMEOUT = 60  # seconds
DEFAULT_KERNEL_PATH = "/usr/local/bin/vmlinux"
DEFAULT_FIRECRACKER_PATH = "/usr/local/bin/firecracker"
DEFAULT_JAILER_PATH = "/usr/local/bin/jailer"


class FirecrackerSandbox:
    """Firecracker microVM-based sandbox for Level 2 command isolation.

    Provides strong isolation by running commands inside a lightweight virtual
    machine with configurable resource limits and automatic lifecycle management.

    Args:
        workspace_root: The host directory to mount into the VM.
        vcpu_count: Number of virtual CPUs.
        mem_size_mib: Memory size in MiB.
        timeout: Maximum VM execution time in seconds.
        kernel_path: Path to the vmlinux kernel binary on the host.
        firecracker_path: Path to the firecracker binary.
        jailer_path: Path to the jailer binary.
    """

    def __init__(
        self,
        workspace_root: str,
        vcpu_count: int = DEFAULT_VCPU_COUNT,
        mem_size_mib: int = DEFAULT_MEM_SIZE_MIB,
        timeout: float = DEFAULT_TIMEOUT,
        kernel_path: str = DEFAULT_KERNEL_PATH,
        firecracker_path: str = DEFAULT_FIRECRACKER_PATH,
        jailer_path: str = DEFAULT_JAILER_PATH,
    ) -> None:
        self.workspace_root = workspace_root
        self.vcpu_count = vcpu_count
        self.mem_size_mib = mem_size_mib
        self.timeout = timeout
        self.kernel_path = kernel_path
        self.firecracker_path = firecracker_path
        self.jailer_path = jailer_path

        self._vm_id: str | None = None
        self._config_path: str | None = None
        self._started_at: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, cmd: str, timeout: float | None = None) -> tuple[int, str]:
        """Execute *cmd* inside a Firecracker microVM.

        Args:
            cmd: The shell command to execute inside the VM.
            timeout: Override the instance-level timeout for this run.

        Returns:
            A tuple of (exit_code, output_text).

        Raises:
            RuntimeError: If KVM or required binaries are unavailable.
        """
        effective_timeout = timeout if timeout is not None else self.timeout

        # 1. Pre-flight checks
        self._check_prerequisites()

        # 2. Prepare rootfs
        rootfs_path = self._prepare_rootfs()

        # 3. Generate config and start VM
        self._vm_id = f"agent_vm_{os.getpid()}_{int(time.time())}"
        self._config_path = f"/tmp/{self._vm_id}_config.json"

        config = self._generate_config(cmd, rootfs_path)
        with open(self._config_path, "w") as f:
            json.dump(config, f)

        self._started_at = time.time()
        logger.info(f"Executing (Level 2 Firecracker): {cmd}")

        # 4. Launch via jailer
        try:
            exit_code, output = await self._launch_vm(effective_timeout)
            return exit_code, output
        finally:
            await self._cleanup()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _launch_vm(self, timeout: float) -> tuple[int, str]:
        """Start the Firecracker VM and wait for output with timeout."""
        jailer_cmd = [
            self.jailer_path,
            "--id", self._vm_id or "unknown",
            "--exec-file", self.firecracker_path,
            "--uid", "0", "--gid", "0",
            "--node", "0",
            "--config-file", self._config_path or "",
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *jailer_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            output = self._capture_output(stdout, stderr)
            return process.returncode or 0, output
        except asyncio.TimeoutError:
            logger.error(f"Firecracker VM timed out after {timeout}s")
            try:
                pgid = os.getpgid(process.pid)
                if pgid != os.getpid():
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    process.kill()
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    process.kill()
                except (ProcessLookupError, OSError):
                    pass
            await process.wait()
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
            return -1, f"VM execution timed out after {timeout}s"
        except Exception as e:
            logger.error(f"Firecracker execution error: {e}")
            return -1, str(e)

    async def _cleanup(self) -> None:
        """Clean up temporary config files and reset VM state after execution."""
        if self._config_path and os.path.exists(self._config_path):
            try:
                os.remove(self._config_path)
                logger.debug(f"Cleaned up config: {self._config_path}")
            except OSError as e:
                logger.warning(f"Failed to remove config file: {e}")
        self._config_path = None
        self._vm_id = None
        self._started_at = None

    # ------------------------------------------------------------------
    # Pre-flight & rootfs
    # ------------------------------------------------------------------

    def _check_prerequisites(self) -> None:
        """Validate that KVM, firecracker, and jailer are available."""
        if not os.path.exists("/dev/kvm"):
            raise RuntimeError("KVM not available (/dev/kvm missing). Cannot run Level 2 sandbox.")
        if not shutil.which(self.firecracker_path) and not (
            os.path.isfile(self.firecracker_path) and os.access(self.firecracker_path, os.X_OK)
        ):
            raise RuntimeError(
                f"Firecracker binary not found or not executable at {self.firecracker_path}"
            )
        if not shutil.which(self.jailer_path) and not (
            os.path.isfile(self.jailer_path) and os.access(self.jailer_path, os.X_OK)
        ):
            raise RuntimeError(
                f"Jailer binary not found or not executable at {self.jailer_path}"
            )
        if not os.path.isfile(self.kernel_path):
            raise RuntimeError(
                f"Kernel image not found at {self.kernel_path}"
            )

    def _prepare_rootfs(self) -> str:
        """Locate and validate the rootfs image.

        Returns:
            Absolute path to the rootfs ext4 image.

        Raises:
            RuntimeError: If the rootfs image is missing.
        """
        rootfs_path = os.path.join(self.workspace_root, ".agent", "rootfs.ext4")
        if not os.path.isfile(rootfs_path):
            raise RuntimeError(
                f"Rootfs image not found at {rootfs_path}. "
                "Prepare it before using Level 2 sandbox."
            )
        return rootfs_path

    # ------------------------------------------------------------------
    # Config generation & output capture
    # ------------------------------------------------------------------

    def _generate_config(self, cmd: str, rootfs_path: str) -> dict:
        """生成 Firecracker 启动配置。

        The command is injected via boot_args using base64 encoding to avoid
        command injection through shell metacharacters in boot_args. The init
        script inside rootfs should decode agentd_cmd_b64 to retrieve the
        original command.
        """
        encoded_cmd = base64.b64encode(cmd.encode()).decode()
        boot_args = (
            f"console=ttyS0 reboot=k panic=1 pci=off "
            f"agentd_cmd_b64={encoded_cmd}"
        )
        if len(boot_args) > 4096:
            raise RuntimeError(
                f"Command too long for boot_args ({len(boot_args)} > 4096 bytes). "
                f"Shorten the command or use a different sandbox level."
            )
        return {
            "boot-source": {
                "kernel_image_path": self.kernel_path,
                "boot_args": boot_args,
            },
            "drives": [
                {
                    "drive_id": "rootfs",
                    "path_on_host": rootfs_path,
                    "is_root_device": True,
                    "is_read_only": True,
                }
            ],
            "machine-config": {
                "vcpu_count": self.vcpu_count,
                "mem_size_mib": self.mem_size_mib,
            },
        }

    MAX_FIRECRACKER_OUTPUT = 1_048_576  # 1MB

    @staticmethod
    def _capture_output(stdout: bytes, stderr: bytes) -> str:
        """Capture and decode serial console output from the VM.

        Firecracker directs guest console output to stdout; stderr
        contains firecracker's own logs.
        """
        output_parts: list[str] = []
        if stdout:
            decoded = stdout.decode(errors="replace")
            encoded = decoded.encode('utf-8')
            if len(encoded) > FirecrackerSandbox.MAX_FIRECRACKER_OUTPUT:
                # Truncate at a safe UTF-8 character boundary
                truncated = encoded[:FirecrackerSandbox.MAX_FIRECRACKER_OUTPUT]
                # Find the last valid UTF-8 start byte
                while truncated and (truncated[-1] & 0xC0) == 0x80:
                    truncated = truncated[:-1]
                if truncated:
                    decoded = truncated.decode('utf-8', errors='replace') + f"\n... [TRUNCATED: output exceeded {FirecrackerSandbox.MAX_FIRECRACKER_OUTPUT} bytes]"
                else:
                    decoded = f"... [TRUNCATED: output exceeded {FirecrackerSandbox.MAX_FIRECRACKER_OUTPUT} bytes]"
            output_parts.append(decoded)
        if stderr:
            # Append firecracker daemon logs for debugging
            fc_log = stderr.decode(errors="replace")
            if fc_log.strip():
                output_parts.append(f"\n[Firecracker log]\n{fc_log}")
        return "\n".join(output_parts)
