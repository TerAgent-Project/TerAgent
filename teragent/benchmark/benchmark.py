"""teragent.benchmark.benchmark — Performance benchmark framework for three-model evaluation

Provides:
  1. CompilationBenchmark: Measure TAPRequest→CompiledPrompt compilation latency
  2. LatencyBenchmark: Measure end-to-end first-token and total latency (with MockAdapter)
  3. ContextManagementBenchmark: Test 1M context (V4/M3) and 200K compression (GLM-5)
  4. MultimodalBenchmark: Test image/video processing latency with M3
  5. LongHorizonBenchmark: Simulate long-horizon task stability
  6. CostEfficiencyBenchmark: Measure token consumption and cost metrics
  7. RouterBenchmark: Test ModelRouter decision accuracy and latency
  8. FaultRecoveryBenchmark: Test circuit breaker and degradation chain analysis
  9. BenchmarkRunner: Orchestrate all benchmarks and generate reports

Design principles:
  - All benchmarks use MockAdapter (no real API calls) for deterministic results
  - Each benchmark produces structured BenchmarkResult with statistical metrics
  - BenchmarkRunner generates comprehensive reports with cross-model comparison
  - Statistical analysis: mean, median, p95, p99, std_dev
  - All results are dataclass-based for serialization
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from teragent.core.adapters.mock import MockAdapter
from teragent.core.compiler import TAPCompilerRegistry
from teragent.core.tap import (
    DesktopContext,
    LongHorizonConfig,
    MultimodalContent,
    TAPRequest,
    TAPResponse,
)

logger = logging.getLogger(__name__)


# =========================================================================
# Core Data Classes
# =========================================================================


@dataclass
class BenchmarkMetric:
    """A single statistical metric from a benchmark run.

    Attributes:
        name: Metric name (e.g., "compilation_latency_ms")
        value: Primary value (typically mean)
        unit: Unit of measurement (e.g., "ms", "tokens", "CNY")
        mean: Arithmetic mean
        median: Median value
        p95: 95th percentile
        p99: 99th percentile
        std_dev: Standard deviation
        min: Minimum observed value
        max: Maximum observed value
        sample_count: Number of observations
    """

    name: str
    value: float
    unit: str = ""
    mean: float = 0.0
    median: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    std_dev: float = 0.0
    min: float = 0.0
    max: float = 0.0
    sample_count: int = 0

    @classmethod
    def from_samples(cls, name: str, samples: list[float], unit: str = "") -> BenchmarkMetric:
        """Create a BenchmarkMetric from a list of sample values.

        Computes statistical measures (mean, median, p95, p99, std_dev, min, max).

        Args:
            name: Metric name
            samples: List of observed values
            unit: Unit of measurement

        Returns:
            BenchmarkMetric with computed statistics
        """
        if not samples:
            return cls(name=name, value=0.0, unit=unit)

        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        mean_val = statistics.mean(sorted_samples)
        median_val = statistics.median(sorted_samples)
        p95_idx = min(int(n * 0.95), n - 1)
        p99_idx = min(int(n * 0.99), n - 1)
        std_dev_val = statistics.stdev(sorted_samples) if n > 1 else 0.0

        return cls(
            name=name,
            value=mean_val,
            unit=unit,
            mean=mean_val,
            median=median_val,
            p95=sorted_samples[p95_idx],
            p99=sorted_samples[p99_idx],
            std_dev=std_dev_val,
            min=sorted_samples[0],
            max=sorted_samples[-1],
            sample_count=n,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "mean": self.mean,
            "median": self.median,
            "p95": self.p95,
            "p99": self.p99,
            "std_dev": self.std_dev,
            "min": self.min,
            "max": self.max,
            "sample_count": self.sample_count,
        }


@dataclass
class BenchmarkResult:
    """Result from a single benchmark suite.

    Attributes:
        suite_name: Name of the benchmark suite (e.g., "CompilationBenchmark")
        model: Model name (e.g., "deepseek_v4", "minimax_m3", "glm_5")
        metrics: List of BenchmarkMetric instances
        metadata: Additional metadata about the benchmark run
        timestamp: Unix timestamp of when the benchmark was run
    """

    suite_name: str
    model: str
    metrics: list[BenchmarkMetric] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def add_metric(self, metric: BenchmarkMetric) -> None:
        """Add a metric to the result."""
        self.metrics.append(metric)

    def get_metric(self, name: str) -> Optional[BenchmarkMetric]:
        """Get a metric by name."""
        for m in self.metrics:
            if m.name == name:
                return m
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "suite_name": self.suite_name,
            "model": self.model,
            "metrics": [m.to_dict() for m in self.metrics],
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report across all suites and models.

    Attributes:
        results: List of BenchmarkResult instances from all suites
        summary: High-level summary dict
        timestamp: Report generation timestamp
    """

    results: list[BenchmarkResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to the report."""
        self.results.append(result)

    def to_text(self) -> str:
        """Generate a human-readable text report.

        Returns:
            Formatted string with all benchmark results organized by suite and model.
        """
        lines: list[str] = []
        lines.append("=" * 80)
        lines.append("TERAGENT PERFORMANCE BENCHMARK REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}")
        lines.append(f"Total benchmark results: {len(self.results)}")
        lines.append("")

        # Group results by suite name
        suites: dict[str, list[BenchmarkResult]] = {}
        for r in self.results:
            suites.setdefault(r.suite_name, []).append(r)

        for suite_name, suite_results in suites.items():
            lines.append("-" * 70)
            lines.append(f"  {suite_name}")
            lines.append("-" * 70)

            for result in suite_results:
                lines.append(f"  Model: {result.model}")
                for m in result.metrics:
                    lines.append(
                        f"    {m.name}: {m.value:.3f} {m.unit} "
                        f"(median={m.median:.3f}, p95={m.p95:.3f}, "
                        f"p99={m.p99:.3f}, std={m.std_dev:.3f}, n={m.sample_count})"
                    )
                if result.metadata:
                    for k, v in result.metadata.items():
                        lines.append(f"    [{k}] {v}")
                lines.append("")

        # Summary
        if self.summary:
            lines.append("=" * 80)
            lines.append("SUMMARY")
            lines.append("=" * 80)
            for k, v in self.summary.items():
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the report to JSON.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation of the report
        """
        return json.dumps(
            {
                "results": [r.to_dict() for r in self.results],
                "summary": self.summary,
                "timestamp": self.timestamp,
            },
            indent=indent,
            ensure_ascii=False,
        )


# =========================================================================
# Benchmark Input Generators
# =========================================================================


# Intent types for benchmark
INTENTS = ["design", "plan", "execute", "review", "chat", "code_generation"]

# Compiler names for the three new models
THREE_MODEL_COMPILERS = ["deepseek_v4", "minimax_m3", "glm_5"]

# All 9 compiler names
ALL_COMPILERS = [
    "default", "glm", "anthropic", "deepseek",
    "deepseek_v4", "minimax_m3", "glm_5",
]

# Model names for routing
MODEL_DRIVERS = {
    "deepseek_v4_flash": "openai_compatible.deepseek_v4_flash",
    "deepseek_v4_pro": "openai_compatible.deepseek_v4_pro",
    "minimax_m3": "openai_compatible.minimax_m3",
    "glm_5": "openai_compatible.glm_5",
}


def make_tap_request(
    intent: str = "execute",
    instruction: str = "实现一个用户登录模块，包含表单验证和JWT令牌生成。",
    context_size: str = "small",  # "small", "medium", "large"
    has_multimodal: bool = False,
    has_desktop: bool = False,
    is_long_horizon: bool = False,
    thinking_mode: Optional[str] = None,
) -> TAPRequest:
    """Generate a TAPRequest for benchmarking.

    Args:
        intent: Intent type (design, plan, execute, review, chat, code_generation)
        instruction: Core instruction text
        context_size: Size of context — "small" (~500 tokens), "medium" (~5K tokens), "large" (~50K tokens)
        has_multimodal: Include multimodal content (for M3)
        has_desktop: Include desktop context (for M3)
        is_long_horizon: Include long-horizon config (for GLM-5)
        thinking_mode: Override thinking mode (deep/quick/auto)

    Returns:
        A TAPRequest suitable for benchmarking
    """
    # Generate context based on size
    design_text = ""
    plan_text = ""
    memory_text = ""

    if context_size == "small":
        design_text = "# 设计文档\n\n实现用户登录模块，包含表单验证和JWT令牌。" * 2
        plan_text = "### 1.1 实现核心模块\n- 输出文件: auth.py\n- 优先级: 必须" * 2
        memory_text = "项目使用 FastAPI 框架，数据库为 PostgreSQL。" * 2
    elif context_size == "medium":
        design_text = "# 设计文档\n\n## 1. 背景\n\n" + ("系统需要支持多种认证方式。 " * 50) + "\n\n## 2. 技术方案\n\n" + ("采用JWT + OAuth2.0方案。 " * 50)
        plan_text = "### 1.1 实现核心模块\n" + ("- 步骤描述 " * 20 + "\n") * 10
        memory_text = "项目技术栈: " + ("FastAPI, PostgreSQL, Redis, Celery, Docker. " * 20)
    elif context_size == "large":
        # Generate ~50K tokens worth of context (~200K chars)
        design_text = "# 详细设计文档\n\n" + ("## 模块设计\n\n" + "系统包含多个子系统。" * 50 + "\n\n") * 20
        plan_text = "### 执行计划\n\n" + ("- 步骤: " + "详细描述 " * 30 + "\n") * 50
        memory_text = "项目记忆: " + ("关键技术决策和依赖信息。 " * 100)

    request = TAPRequest(
        meta={"intent": intent, "task_id": "bench_001"},
        context={
            "design": design_text,
            "plan": plan_text,
            "memory": memory_text,
            "dependency_report": "fastapi>=0.100, pyjwt>=2.0, sqlalchemy>=2.0",
        },
        instruction=instruction,
        constraints=[
            "使用类型注解",
            "包含错误处理",
            "遵循 PEP 8 规范",
            "输出格式使用 <file path='...'> 标签",
        ],
        output_format_hint="用 <file path='...'> 输出代码",
    )

    # Multimodal content
    if has_multimodal:
        request.multimodal_context = [
            MultimodalContent(
                type="image_url",
                url="https://example.com/mock_screenshot.png",
            ),
            MultimodalContent(
                type="text",
                text="截图显示了登录表单的UI布局",
            ),
        ]

    # Desktop context
    if has_desktop:
        request.desktop_context = DesktopContext(
            screenshot=MultimodalContent(
                type="image_url",
                url="https://example.com/desktop_screenshot.png",
            ),
            interactive_elements=[
                {"type": "button", "label": "登录", "bbox": {"x": 100, "y": 200, "w": 80, "h": 30}, "action": "click"},
                {"type": "input", "label": "用户名", "bbox": {"x": 100, "y": 100, "w": 200, "h": 24}, "action": "type"},
            ],
            active_window="浏览器 - 登录页面",
        )

    # Long-horizon config
    if is_long_horizon:
        request.long_horizon = LongHorizonConfig(
            max_duration_hours=8.0,
            checkpoint_interval_minutes=30.0,
            self_evaluation_enabled=True,
            stagnation_threshold=3,
        )

    # Thinking mode
    if thinking_mode:
        request.thinking_mode = thinking_mode

    return request


# =========================================================================
# Benchmark Suites
# =========================================================================


class BaseBenchmark(ABC):
    """Abstract base class for benchmark suites.

    Each benchmark suite measures a specific aspect of the three-model
    adaptation performance. All suites use MockAdapter for deterministic
    results without real API calls.
    """

    def __init__(self, iterations: int = 50, seed: int = 42) -> None:
        """Initialize the benchmark.

        Args:
            iterations: Number of iterations per benchmark scenario
            seed: Random seed for reproducibility
        """
        self.iterations = iterations
        self.seed = seed
        random.seed(seed)

    @abstractmethod
    def run(self) -> list[BenchmarkResult]:
        """Run the benchmark suite.

        Returns:
            List of BenchmarkResult instances, one per model/scenario
        """
        ...


class CompilationBenchmark(BaseBenchmark):
    """Benchmark: TAPRequest→CompiledPrompt compilation latency.

    Measures the time each Compiler takes to compile a TAPRequest into
    a CompiledPrompt. This is a pure CPU operation (no I/O), so it
    measures the overhead of the compilation logic itself.

    Scenarios:
      - Per-compiler compilation latency across all 9 compilers
      - Per-intent compilation latency for the 3 new compilers
      - Large context compilation latency (50K+ tokens context)
    """

    def run(self) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []

        # Ensure compilers are registered
        import teragent.core.compilers  # noqa: F401 — triggers registration

        # Scenario 1: All 9 compilers compilation latency
        all_compiler_result = BenchmarkResult(
            suite_name="CompilationBenchmark",
            model="all_compilers",
            metadata={"scenario": "all_compilers_latency"},
        )

        for compiler_name in ALL_COMPILERS:
            compiler_cls = TAPCompilerRegistry.get(compiler_name)
            if compiler_cls is None:
                logger.warning(f"Compiler not registered: {compiler_name}")
                continue

            # Use appropriate variant for DeepSeek V4
            if compiler_name == "deepseek_v4":
                compiler = compiler_cls(variant="pro")
            else:
                compiler = compiler_cls()

            samples: list[float] = []
            for _ in range(self.iterations):
                request = make_tap_request(intent="execute")
                start = time.perf_counter()
                compiled = compiler.compile(request)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                samples.append(elapsed)

            metric = BenchmarkMetric.from_samples(
                name=f"compile_latency_{compiler_name}",
                samples=samples,
                unit="ms",
            )
            all_compiler_result.add_metric(metric)

            # Also record compiled prompt size
            prompt_size = len(str(compiled.messages)) if compiled.messages else 0
            all_compiler_result.add_metric(BenchmarkMetric(
                name=f"compiled_size_{compiler_name}",
                value=prompt_size,
                unit="chars",
                sample_count=1,
            ))

        results.append(all_compiler_result)

        # Scenario 2: Per-intent latency for 3 new compilers
        for compiler_name in THREE_MODEL_COMPILERS:
            compiler_cls = TAPCompilerRegistry.get(compiler_name)
            if compiler_cls is None:
                continue

            if compiler_name == "deepseek_v4":
                compiler = compiler_cls(variant="pro")
            else:
                compiler = compiler_cls()

            result = BenchmarkResult(
                suite_name="CompilationBenchmark",
                model=compiler_name,
                metadata={"scenario": "per_intent_latency"},
            )

            for intent in INTENTS:
                samples: list[float] = []
                for _ in range(self.iterations):
                    request = make_tap_request(intent=intent)
                    start = time.perf_counter()
                    compiler.compile(request)
                    elapsed = (time.perf_counter() - start) * 1000
                    samples.append(elapsed)

                result.add_metric(BenchmarkMetric.from_samples(
                    name=f"compile_latency_{intent}",
                    samples=samples,
                    unit="ms",
                ))

            results.append(result)

        # Scenario 3: Large context compilation
        for compiler_name in THREE_MODEL_COMPILERS:
            compiler_cls = TAPCompilerRegistry.get(compiler_name)
            if compiler_cls is None:
                continue

            if compiler_name == "deepseek_v4":
                compiler = compiler_cls(variant="pro")
            else:
                compiler = compiler_cls()

            result = BenchmarkResult(
                suite_name="CompilationBenchmark",
                model=compiler_name,
                metadata={"scenario": "large_context_latency"},
            )

            for size in ["small", "medium", "large"]:
                samples: list[float] = []
                for _ in range(max(10, self.iterations // 5)):
                    request = make_tap_request(intent="execute", context_size=size)
                    start = time.perf_counter()
                    compiler.compile(request)
                    elapsed = (time.perf_counter() - start) * 1000
                    samples.append(elapsed)

                result.add_metric(BenchmarkMetric.from_samples(
                    name=f"compile_latency_ctx_{size}",
                    samples=samples,
                    unit="ms",
                ))

            results.append(result)

        return results


class LatencyBenchmark(BaseBenchmark):
    """Benchmark: End-to-end first-token and total latency with MockAdapter.

    Uses MockAdapter with simulated delay to measure the complete
    compilation + send pipeline latency.

    Scenarios:
      - Per-model first-token latency
      - Per-model total latency
      - Per-intent latency breakdown
    """

    def run(self) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        import teragent.core.compilers  # noqa: F401

        adapter = MockAdapter(delay=0.05)  # 50ms simulated delay

        for compiler_name in THREE_MODEL_COMPILERS:
            compiler_cls = TAPCompilerRegistry.get(compiler_name)
            if compiler_cls is None:
                continue

            if compiler_name == "deepseek_v4":
                compiler = compiler_cls(variant="pro")
            else:
                compiler = compiler_cls()

            model_name = f"mock_{compiler_name}"

            # Per-intent latency
            result = BenchmarkResult(
                suite_name="LatencyBenchmark",
                model=compiler_name,
                metadata={"scenario": "e2e_latency_per_intent"},
            )

            for intent in INTENTS:
                compile_samples: list[float] = []
                send_samples: list[float] = []
                total_samples: list[float] = []

                for _ in range(self.iterations):
                    request = make_tap_request(intent=intent)

                    # Compile
                    compile_start = time.perf_counter()
                    compiled = compiler.compile(request)
                    compile_elapsed = (time.perf_counter() - compile_start) * 1000
                    compile_samples.append(compile_elapsed)

                    # Send (async, so we run in a new event loop for benchmarking)
                    send_start = time.perf_counter()
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If we're inside an existing event loop, create a task
                            _response = asyncio.ensure_future(
                                adapter.send(compiled, model_name)
                            )
                        else:
                            _response = loop.run_until_complete(
                                adapter.send(compiled, model_name)
                            )
                    except RuntimeError:
                        # No event loop exists, create one
                        _response = asyncio.run(
                            adapter.send(compiled, model_name)
                        )
                    send_elapsed = (time.perf_counter() - send_start) * 1000
                    send_samples.append(send_elapsed)
                    total_samples.append(compile_elapsed + send_elapsed)

                result.add_metric(BenchmarkMetric.from_samples(
                    name=f"compile_latency_{intent}",
                    samples=compile_samples,
                    unit="ms",
                ))
                result.add_metric(BenchmarkMetric.from_samples(
                    name=f"send_latency_{intent}",
                    samples=send_samples,
                    unit="ms",
                ))
                result.add_metric(BenchmarkMetric.from_samples(
                    name=f"total_latency_{intent}",
                    samples=total_samples,
                    unit="ms",
                ))

            results.append(result)

        return results


class ContextManagementBenchmark(BaseBenchmark):
    """Benchmark: Context management performance.

    Tests how each model's compiler handles different context sizes
    and whether the compiled prompt stays within the model's context limit.

    Scenarios:
      - V4/M3: 1M context — test large context compilation and prompt sizing
      - GLM-5: 200K context — test extreme compression effectiveness
      - Context budget utilization across models
    """

    def run(self) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        import teragent.core.compilers  # noqa: F401

        context_sizes = {
            "10K": 2_500,    # ~10K chars ≈ 2.5K tokens
            "50K": 12_500,   # ~50K chars ≈ 12.5K tokens
            "200K": 50_000,  # ~200K chars ≈ 50K tokens
            "500K": 125_000, # ~500K chars ≈ 125K tokens
            "1M": 250_000,   # ~1M chars ≈ 250K tokens
        }

        for compiler_name in THREE_MODEL_COMPILERS:
            compiler_cls = TAPCompilerRegistry.get(compiler_name)
            if compiler_cls is None:
                continue

            if compiler_name == "deepseek_v4":
                compiler = compiler_cls(variant="pro")
            else:
                compiler = compiler_cls()

            result = BenchmarkResult(
                suite_name="ContextManagementBenchmark",
                model=compiler_name,
                metadata={"scenario": "context_budget_utilization"},
            )

            max_ctx = compiler.max_context_tokens

            for size_label, char_count in context_sizes.items():
                # Skip sizes that exceed the model's context window
                estimated_tokens = char_count // 4
                if estimated_tokens > max_ctx:
                    continue

                # Generate a request with the specified context size
                context_text = "A" * char_count
                request = TAPRequest(
                    meta={"intent": "execute", "task_id": "ctx_bench"},
                    context={"design": context_text},
                    instruction="根据设计文档实现功能",
                    constraints=["遵循设计规范"],
                    output_format_hint="用 <file path='...'> 输出代码",
                )

                compile_samples: list[float] = []
                token_estimates: list[int] = []
                prompt_sizes: list[int] = []

                for _ in range(max(5, self.iterations // 10)):
                    start = time.perf_counter()
                    compiled = compiler.compile(request)
                    elapsed = (time.perf_counter() - start) * 1000
                    compile_samples.append(elapsed)

                    # Estimate compiled prompt tokens
                    total_chars = sum(
                        len(str(msg.get("content", "")))
                        for msg in compiled.messages
                    ) if compiled.messages else 0
                    estimated_tokens = total_chars // 4
                    token_estimates.append(estimated_tokens)
                    prompt_sizes.append(total_chars)

                if compile_samples:
                    result.add_metric(BenchmarkMetric.from_samples(
                        name=f"compile_latency_{size_label}",
                        samples=compile_samples,
                        unit="ms",
                    ))

                if token_estimates:
                    result.add_metric(BenchmarkMetric.from_samples(
                        name=f"estimated_tokens_{size_label}",
                        samples=[float(t) for t in token_estimates],
                        unit="tokens",
                    ))

                # Budget utilization ratio
                if token_estimates and max_ctx > 0:
                    avg_tokens = statistics.mean(token_estimates)
                    utilization = avg_tokens / max_ctx
                    result.add_metric(BenchmarkMetric(
                        name=f"budget_utilization_{size_label}",
                        value=utilization,
                        unit="ratio",
                        sample_count=1,
                    ))

            # Record max context limit
            result.add_metric(BenchmarkMetric(
                name="max_context_tokens",
                value=float(max_ctx),
                unit="tokens",
                sample_count=1,
            ))

            results.append(result)

        return results


class MultimodalBenchmark(BaseBenchmark):
    """Benchmark: Multimodal processing latency.

    Tests MiniMax M3's native multimodal support vs other compilers'
    degradation handling.

    Scenarios:
      - M3 native multimodal compilation (image, video, desktop)
      - V4/GLM-5 multimodal degradation overhead
      - Mixed content (multi-image, image+video) compilation
    """

    def run(self) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        import teragent.core.compilers  # noqa: F401

        # Scenario 1: M3 native multimodal vs degradation
        for compiler_name in THREE_MODEL_COMPILERS:
            compiler_cls = TAPCompilerRegistry.get(compiler_name)
            if compiler_cls is None:
                continue

            if compiler_name == "deepseek_v4":
                compiler = compiler_cls(variant="pro")
            else:
                compiler = compiler_cls()

            result = BenchmarkResult(
                suite_name="MultimodalBenchmark",
                model=compiler_name,
                metadata={"scenario": "multimodal_vs_degradation"},
            )

            # Text-only baseline
            baseline_samples: list[float] = []
            for _ in range(self.iterations):
                request = make_tap_request(intent="execute", has_multimodal=False)
                start = time.perf_counter()
                compiler.compile(request)
                elapsed = (time.perf_counter() - start) * 1000
                baseline_samples.append(elapsed)
            result.add_metric(BenchmarkMetric.from_samples(
                name="text_only_latency",
                samples=baseline_samples,
                unit="ms",
            ))

            # Image multimodal
            image_samples: list[float] = []
            for _ in range(self.iterations):
                request = make_tap_request(intent="execute", has_multimodal=True)
                start = time.perf_counter()
                compiler.compile(request)
                elapsed = (time.perf_counter() - start) * 1000
                image_samples.append(elapsed)
            result.add_metric(BenchmarkMetric.from_samples(
                name="image_multimodal_latency",
                samples=image_samples,
                unit="ms",
            ))

            # Desktop context
            desktop_samples: list[float] = []
            for _ in range(self.iterations):
                request = make_tap_request(intent="execute", has_desktop=True)
                start = time.perf_counter()
                compiler.compile(request)
                elapsed = (time.perf_counter() - start) * 1000
                desktop_samples.append(elapsed)
            result.add_metric(BenchmarkMetric.from_samples(
                name="desktop_context_latency",
                samples=desktop_samples,
                unit="ms",
            ))

            # Multimodal overhead (difference from baseline)
            if baseline_samples and image_samples:
                baseline_mean = statistics.mean(baseline_samples)
                image_mean = statistics.mean(image_samples)
                overhead = image_mean - baseline_mean
                result.add_metric(BenchmarkMetric(
                    name="multimodal_overhead_ms",
                    value=overhead,
                    unit="ms",
                    sample_count=1,
                ))

            results.append(result)

        # Scenario 2: M3 mixed content types
        m3_cls = TAPCompilerRegistry.get("minimax_m3")
        if m3_cls is not None:
            m3 = m3_cls()
            mixed_result = BenchmarkResult(
                suite_name="MultimodalBenchmark",
                model="minimax_m3",
                metadata={"scenario": "mixed_content_types"},
            )

            # Multi-image
            multi_img_samples: list[float] = []
            for _ in range(self.iterations):
                request = TAPRequest(
                    meta={"intent": "execute"},
                    instruction="分析多张截图",
                    multimodal_context=[
                        MultimodalContent(type="image_url", url=f"https://example.com/img{i}.png")
                        for i in range(5)
                    ],
                )
                start = time.perf_counter()
                m3.compile(request)
                elapsed = (time.perf_counter() - start) * 1000
                multi_img_samples.append(elapsed)
            mixed_result.add_metric(BenchmarkMetric.from_samples(
                name="multi_image_latency",
                samples=multi_img_samples,
                unit="ms",
            ))

            # Image + video
            img_video_samples: list[float] = []
            for _ in range(self.iterations):
                request = TAPRequest(
                    meta={"intent": "execute"},
                    instruction="分析截图和视频",
                    multimodal_context=[
                        MultimodalContent(type="image_url", url="https://example.com/img.png"),
                        MultimodalContent(type="video_url", url="https://example.com/vid.mp4"),
                    ],
                )
                start = time.perf_counter()
                m3.compile(request)
                elapsed = (time.perf_counter() - start) * 1000
                img_video_samples.append(elapsed)
            mixed_result.add_metric(BenchmarkMetric.from_samples(
                name="image_video_mixed_latency",
                samples=img_video_samples,
                unit="ms",
            ))

            results.append(mixed_result)

        return results


class LongHorizonBenchmark(BaseBenchmark):
    """Benchmark: Long-horizon task stability simulation.

    Simulates long-horizon task execution patterns for GLM-5,
    measuring compilation stability across checkpoint cycles.

    Scenarios:
      - GLM-5 long-horizon compilation with checkpoints
      - Self-evaluation prompt injection overhead
      - Strategy switch prompt injection overhead
      - Multi-step compilation stability (variance across steps)
    """

    def run(self) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        import teragent.core.compilers  # noqa: F401

        # GLM-5 long-horizon benchmark
        glm_cls = TAPCompilerRegistry.get("glm_5")
        if glm_cls is None:
            return results

        glm = glm_cls()

        # Scenario 1: Normal vs long-horizon compilation
        result = BenchmarkResult(
            suite_name="LongHorizonBenchmark",
            model="glm_5",
            metadata={"scenario": "long_horizon_vs_normal"},
        )

        normal_samples: list[float] = []
        for _ in range(self.iterations):
            request = make_tap_request(intent="execute", is_long_horizon=False)
            start = time.perf_counter()
            glm.compile(request)
            elapsed = (time.perf_counter() - start) * 1000
            normal_samples.append(elapsed)
        result.add_metric(BenchmarkMetric.from_samples(
            name="normal_compilation_latency",
            samples=normal_samples,
            unit="ms",
        ))

        long_horizon_samples: list[float] = []
        for _ in range(self.iterations):
            request = make_tap_request(intent="execute", is_long_horizon=True)
            start = time.perf_counter()
            compiled = glm.compile(request)
            elapsed = (time.perf_counter() - start) * 1000
            long_horizon_samples.append(elapsed)
        result.add_metric(BenchmarkMetric.from_samples(
            name="long_horizon_compilation_latency",
            samples=long_horizon_samples,
            unit="ms",
        ))

        # Overhead
        if normal_samples and long_horizon_samples:
            overhead = statistics.mean(long_horizon_samples) - statistics.mean(normal_samples)
            result.add_metric(BenchmarkMetric(
                name="long_horizon_overhead_ms",
                value=overhead,
                unit="ms",
                sample_count=1,
            ))

        results.append(result)

        # Scenario 2: Multi-step simulation (simulating 100 steps)
        steps_result = BenchmarkResult(
            suite_name="LongHorizonBenchmark",
            model="glm_5",
            metadata={"scenario": "multi_step_stability"},
        )

        step_latencies: list[float] = []
        step_prompt_sizes: list[int] = []
        num_steps = min(100, self.iterations * 2)

        for step in range(num_steps):
            # Simulate evolving context (growing as steps accumulate)
            accumulated_context = f"步骤 {step + 1}: 已完成部分工作。" * (1 + step // 10)
            request = TAPRequest(
                meta={"intent": "execute", "task_id": f"step_{step}"},
                context={"design": accumulated_context},
                instruction=f"继续执行步骤 {step + 1}",
                long_horizon=LongHorizonConfig(
                    max_duration_hours=8.0,
                    checkpoint_interval_minutes=30.0,
                    self_evaluation_enabled=(step % 5 == 0),  # Every 5th step has self-eval
                    stagnation_threshold=3,
                ),
            )

            start = time.perf_counter()
            compiled = glm.compile(request)
            elapsed = (time.perf_counter() - start) * 1000
            step_latencies.append(elapsed)

            prompt_size = sum(
                len(str(msg.get("content", "")))
                for msg in compiled.messages
            ) if compiled.messages else 0
            step_prompt_sizes.append(prompt_size)

        steps_result.add_metric(BenchmarkMetric.from_samples(
            name="step_latency",
            samples=step_latencies,
            unit="ms",
        ))
        steps_result.add_metric(BenchmarkMetric.from_samples(
            name="step_prompt_size",
            samples=[float(s) for s in step_prompt_sizes],
            unit="chars",
        ))

        # Stability metric: coefficient of variation
        if step_latencies:
            cv = statistics.stdev(step_latencies) / statistics.mean(step_latencies) if statistics.mean(step_latencies) > 0 else 0
            steps_result.add_metric(BenchmarkMetric(
                name="latency_cv",
                value=cv,
                unit="ratio",
                sample_count=1,
            ))

        results.append(steps_result)

        # Scenario 3: Strategy switch prompt injection
        switch_result = BenchmarkResult(
            suite_name="LongHorizonBenchmark",
            model="glm_5",
            metadata={"scenario": "strategy_switch"},
        )

        switch_samples: list[float] = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            _switch_prompt = glm.build_strategy_switch_prompt("连续3次相同结果")
            elapsed = (time.perf_counter() - start) * 1000
            switch_samples.append(elapsed)

        switch_result.add_metric(BenchmarkMetric.from_samples(
            name="strategy_switch_prompt_latency",
            samples=switch_samples,
            unit="ms",
        ))

        results.append(switch_result)

        return results


class CostEfficiencyBenchmark(BaseBenchmark):
    """Benchmark: Token consumption and cost metrics.

    Uses MockAdapter to simulate token usage and CostTracker to
    measure per-model cost efficiency.

    Scenarios:
      - Per-model token consumption per intent
      - Per-model estimated cost per intent
      - Cache hit rate analysis (DeepSeek V4)
      - Cross-model cost comparison
    """

    # Pricing from RoutingTable (CNY per million tokens)
    PRICING = {
        "deepseek_v4_flash": {
            "prompt_per_million": 1.0,
            "completion_per_million": 2.0,
            "cache_hit_per_million": 0.1,
            "cache_miss_per_million": 1.0,
        },
        "deepseek_v4_pro": {
            "prompt_per_million": 4.0,
            "completion_per_million": 16.0,
            "cache_hit_per_million": 0.4,
            "cache_miss_per_million": 4.0,
        },
        "minimax_m3": {
            "prompt_per_million": 1.0,
            "completion_per_million": 2.0,
        },
        "glm_5": {
            "prompt_per_million": 2.0,
            "completion_per_million": 8.0,
        },
    }

    def run(self) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        import teragent.core.compilers  # noqa: F401

        adapter = MockAdapter(delay=0.01)

        for compiler_name in THREE_MODEL_COMPILERS:
            compiler_cls = TAPCompilerRegistry.get(compiler_name)
            if compiler_cls is None:
                continue

            if compiler_name == "deepseek_v4":
                compiler = compiler_cls(variant="pro")
            else:
                compiler = compiler_cls()

            result = BenchmarkResult(
                suite_name="CostEfficiencyBenchmark",
                model=compiler_name,
                metadata={"scenario": "per_intent_cost"},
            )

            for intent in INTENTS:
                prompt_token_samples: list[float] = []
                completion_token_samples: list[float] = []
                cost_samples: list[float] = []

                model_key = compiler_name.replace("deepseek_v4", "deepseek_v4_pro")
                pricing = self.PRICING.get(model_key, {})

                for _ in range(self.iterations):
                    request = make_tap_request(intent=intent)
                    compiled = compiler.compile(request)

                    try:
                        response = asyncio.run(
                            adapter.send(compiled, f"mock_{compiler_name}")
                        )
                    except RuntimeError:
                        response = TAPResponse(raw_text="mock", usage={})

                    prompt_tokens = response.prompt_tokens
                    completion_tokens = response.completion_tokens
                    prompt_token_samples.append(float(prompt_tokens))
                    completion_token_samples.append(float(completion_tokens))

                    # Calculate cost
                    if pricing:
                        prompt_cost = prompt_tokens * pricing.get("prompt_per_million", 0) / 1_000_000
                        completion_cost = completion_tokens * pricing.get("completion_per_million", 0) / 1_000_000
                        # Cache hit savings
                        cache_hits = response.cache_hit_tokens
                        cache_savings = cache_hits * (
                            pricing.get("prompt_per_million", 0)
                            - pricing.get("cache_hit_per_million", pricing.get("prompt_per_million", 0))
                        ) / 1_000_000
                        total_cost = prompt_cost + completion_cost - cache_savings
                        cost_samples.append(total_cost * 1_000_000)  # in micro-CNY

                result.add_metric(BenchmarkMetric.from_samples(
                    name=f"prompt_tokens_{intent}",
                    samples=prompt_token_samples,
                    unit="tokens",
                ))
                result.add_metric(BenchmarkMetric.from_samples(
                    name=f"completion_tokens_{intent}",
                    samples=completion_token_samples,
                    unit="tokens",
                ))

                if cost_samples:
                    result.add_metric(BenchmarkMetric.from_samples(
                        name=f"estimated_cost_{intent}",
                        samples=cost_samples,
                        unit="μCNY",
                    ))

            results.append(result)

        # Cross-model cost comparison
        comparison_result = BenchmarkResult(
            suite_name="CostEfficiencyBenchmark",
            model="comparison",
            metadata={"scenario": "cross_model_cost_comparison"},
        )

        # Calculate average cost per model across all intents
        for r in results:
            if r.model == "comparison":
                continue
            cost_metrics = [m for m in r.metrics if m.name.startswith("estimated_cost_")]
            if cost_metrics:
                avg_cost = statistics.mean([m.mean for m in cost_metrics])
                comparison_result.add_metric(BenchmarkMetric(
                    name=f"avg_cost_{r.model}",
                    value=avg_cost,
                    unit="μCNY",
                    sample_count=len(cost_metrics),
                ))

        results.append(comparison_result)

        return results


class RouterBenchmark(BaseBenchmark):
    """Benchmark: ModelRouter decision accuracy and latency.

    Tests the ModelRouter's ability to correctly route TAP requests
    to the optimal model based on intent, multimodal content, context
    length, and long-horizon requirements.

    Scenarios:
      - Routing decision latency
      - Intent-based routing accuracy
      - Override routing (multimodal, context length, long-horizon)
      - Pipeline profile switching
    """

    def run(self) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        from teragent.router.model_router import ModelRouter, RoutingTable

        routing_table = RoutingTable()
        router = ModelRouter(routing_table=routing_table)

        # Scenario 1: Routing decision latency
        latency_result = BenchmarkResult(
            suite_name="RouterBenchmark",
            model="all_models",
            metadata={"scenario": "routing_decision_latency"},
        )

        routing_samples: list[float] = []
        test_requests = [
            make_tap_request(intent="design"),
            make_tap_request(intent="plan"),
            make_tap_request(intent="execute"),
            make_tap_request(intent="review"),
            make_tap_request(intent="chat"),
            make_tap_request(intent="execute", has_multimodal=True),
            make_tap_request(intent="execute", has_desktop=True),
            make_tap_request(intent="execute", is_long_horizon=True),
        ]

        for _ in range(self.iterations):
            for request in test_requests:
                start = time.perf_counter()
                decision = router.route(request)
                elapsed = (time.perf_counter() - start) * 1000
                routing_samples.append(elapsed)

        latency_result.add_metric(BenchmarkMetric.from_samples(
            name="routing_decision_latency",
            samples=routing_samples,
            unit="ms",
        ))
        results.append(latency_result)

        # Scenario 2: Override routing accuracy
        override_result = BenchmarkResult(
            suite_name="RouterBenchmark",
            model="all_models",
            metadata={"scenario": "override_routing"},
        )

        # Test multimodal override → should route to M3
        mm_decisions: list[str] = []
        for _ in range(self.iterations):
            request = make_tap_request(intent="execute", has_multimodal=True)
            decision = router.route(request)
            mm_decisions.append(decision.selected_compiler)

        m3_mm_rate = sum(1 for d in mm_decisions if d == "minimax_m3") / len(mm_decisions) if mm_decisions else 0
        override_result.add_metric(BenchmarkMetric(
            name="multimodal_to_m3_rate",
            value=m3_mm_rate,
            unit="ratio",
            sample_count=len(mm_decisions),
        ))

        # Test long-horizon override → should route to GLM-5
        lh_decisions: list[str] = []
        for _ in range(self.iterations):
            request = make_tap_request(intent="execute", is_long_horizon=True)
            decision = router.route(request)
            lh_decisions.append(decision.selected_compiler)

        glm_lh_rate = sum(1 for d in lh_decisions if d == "glm_5") / len(lh_decisions) if lh_decisions else 0
        override_result.add_metric(BenchmarkMetric(
            name="long_horizon_to_glm5_rate",
            value=glm_lh_rate,
            unit="ratio",
            sample_count=len(lh_decisions),
        ))

        # Test intent-based routing
        intent_routing_correct = 0
        total_intent_checks = 0
        expected_intent_map = {
            "design": "deepseek_v4",  # V4 Pro for design
            "plan": "glm_5",
            "execute": "glm_5",
            "review": "deepseek_v4",  # V4 Pro for review
            "chat": "deepseek_v4",    # V4 Flash for chat
        }

        for intent, expected_compiler in expected_intent_map.items():
            for _ in range(self.iterations):
                request = make_tap_request(intent=intent)
                decision = router.route(request)
                total_intent_checks += 1
                if expected_compiler in decision.selected_compiler:
                    intent_routing_correct += 1

        intent_accuracy = intent_routing_correct / total_intent_checks if total_intent_checks > 0 else 0
        override_result.add_metric(BenchmarkMetric(
            name="intent_routing_accuracy",
            value=intent_accuracy,
            unit="ratio",
            sample_count=total_intent_checks,
        ))

        results.append(override_result)

        return results


class FaultRecoveryBenchmark(BaseBenchmark):
    """Benchmark: Circuit breaker and degradation chain analysis.

    Tests the fault tolerance and recovery mechanisms including
    circuit breaker triggering, degradation chain execution,
    and recovery timing.

    Scenarios:
      - Consecutive failure circuit breaker trigger timing
      - Degradation chain latency (V4-Pro→V4-Flash→GLM-5)
      - Recovery time measurement
    """

    def run(self) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []

        from teragent.event_bus import EventBus
        from teragent.reliability.circuit_breaker import (
            CircuitBreakerManager,
            ConsecutiveFailureBreaker,
        )

        # Scenario 1: Circuit breaker trigger timing
        cb_result = BenchmarkResult(
            suite_name="FaultRecoveryBenchmark",
            model="all_models",
            metadata={"scenario": "circuit_breaker_trigger"},
        )

        trigger_samples: list[float] = []
        for _ in range(20):  # Fewer iterations for circuit breaker
            bus = EventBus()
            _manager = CircuitBreakerManager(bus=bus)
            breaker = ConsecutiveFailureBreaker(
                max_consecutive=5,
                window_seconds=300.0,
            )

            start = time.perf_counter()
            for i in range(10):
                breaker.record_failure(f"Failure {i}")
                if breaker.is_open:
                    elapsed = (time.perf_counter() - start) * 1000
                    trigger_samples.append(elapsed)
                    break

        if trigger_samples:
            cb_result.add_metric(BenchmarkMetric.from_samples(
                name="circuit_breaker_trigger_latency",
                samples=trigger_samples,
                unit="ms",
            ))

        results.append(cb_result)

        # Scenario 2: Degradation chain latency
        from teragent.router.model_router import ModelRouter, RoutingTable

        degradation_result = BenchmarkResult(
            suite_name="FaultRecoveryBenchmark",
            model="all_models",
            metadata={"scenario": "degradation_chain"},
        )

        routing_table = RoutingTable()
        router = ModelRouter(routing_table=routing_table)

        # Simulate degradation chain: V4-Pro → V4-Flash → GLM-5
        degradation_samples: list[float] = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            # First choice: V4 Pro
            decision1 = router.route(make_tap_request(intent="design"))
            # Simulate V4 Pro failure → fallback
            fallback_driver = routing_table.degradation_map.get(decision1.selected_driver, "")
            if fallback_driver:
                # Second choice: fallback
                _fallback_compiler = routing_table.resolve_compiler(fallback_driver)
                # If fallback also fails, go further
                _second_fallback = routing_table.degradation_map.get(fallback_driver, "")
            elapsed = (time.perf_counter() - start) * 1000
            degradation_samples.append(elapsed)

        degradation_result.add_metric(BenchmarkMetric.from_samples(
            name="degradation_chain_latency",
            samples=degradation_samples,
            unit="ms",
        ))

        results.append(degradation_result)

        return results


# =========================================================================
# Benchmark Runner
# =========================================================================


class BenchmarkRunner:
    """Orchestrate all benchmarks and generate comprehensive reports.

    Runs all benchmark suites in sequence and aggregates results
    into a single BenchmarkReport.

    Usage::

        runner = BenchmarkRunner(iterations=100, seed=42)
        report = runner.run_all()
        print(report.to_text())
        # or
        with open("report.json", "w") as f:
            f.write(report.to_json())
    """

    def __init__(
        self,
        iterations: int = 50,
        seed: int = 42,
        suites: Optional[list[str]] = None,
    ) -> None:
        """Initialize the benchmark runner.

        Args:
            iterations: Number of iterations per benchmark scenario
            seed: Random seed for reproducibility
            suites: Optional list of suite names to run.
                    If None, all suites are run.
                    Available: "compilation", "latency", "context",
                               "multimodal", "long_horizon", "cost",
                               "router", "fault_recovery"
        """
        self.iterations = iterations
        self.seed = seed
        self.suites = suites

        self._suite_map: dict[str, type[BaseBenchmark]] = {
            "compilation": CompilationBenchmark,
            "latency": LatencyBenchmark,
            "context": ContextManagementBenchmark,
            "multimodal": MultimodalBenchmark,
            "long_horizon": LongHorizonBenchmark,
            "cost": CostEfficiencyBenchmark,
            "router": RouterBenchmark,
            "fault_recovery": FaultRecoveryBenchmark,
        }

    def run_all(self) -> BenchmarkReport:
        """Run all benchmark suites and return a comprehensive report.

        Returns:
            BenchmarkReport containing all results and summary
        """
        report = BenchmarkReport()
        suite_names = self.suites or list(self._suite_map.keys())

        total_start = time.perf_counter()

        for suite_name in suite_names:
            bench_cls = self._suite_map.get(suite_name)
            if bench_cls is None:
                logger.warning(f"Unknown benchmark suite: {suite_name}")
                continue

            logger.info(f"Running benchmark suite: {suite_name}")
            bench = bench_cls(iterations=self.iterations, seed=self.seed)

            try:
                suite_results = bench.run()
                for result in suite_results:
                    report.add_result(result)
            except Exception as e:
                logger.error(f"Benchmark suite {suite_name} failed: {e}")
                report.add_result(BenchmarkResult(
                    suite_name=suite_name,
                    model="error",
                    metadata={"error": str(e)},
                ))

        total_elapsed = time.perf_counter() - total_start

        # Generate summary
        report.summary = self._generate_summary(report, total_elapsed)

        return report

    def _generate_summary(self, report: BenchmarkReport, total_elapsed: float) -> dict[str, Any]:
        """Generate a high-level summary from all benchmark results.

        Args:
            report: The BenchmarkReport with all results
            total_elapsed: Total wall-clock time for the entire benchmark run

        Returns:
            Summary dictionary
        """
        summary: dict[str, Any] = {
            "total_benchmarks": len(report.results),
            "total_elapsed_seconds": round(total_elapsed, 3),
            "iterations_per_scenario": self.iterations,
            "seed": self.seed,
        }

        # Extract key metrics for quick overview
        for result in report.results:
            for metric in result.metrics:
                key = f"{result.suite_name}.{result.model}.{metric.name}"
                summary[key] = round(metric.value, 4) if isinstance(metric.value, float) else metric.value

        return summary

    def run_suite(self, suite_name: str) -> list[BenchmarkResult]:
        """Run a single benchmark suite.

        Args:
            suite_name: Name of the suite to run

        Returns:
            List of BenchmarkResult instances from the suite
        """
        bench_cls = self._suite_map.get(suite_name)
        if bench_cls is None:
            raise ValueError(
                f"Unknown benchmark suite: {suite_name}. "
                f"Available: {list(self._suite_map.keys())}"
            )

        bench = bench_cls(iterations=self.iterations, seed=self.seed)
        return bench.run()
