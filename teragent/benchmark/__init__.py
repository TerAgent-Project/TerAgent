"""teragent.benchmark — Performance benchmark framework for three-model evaluation

Provides deterministic benchmarking for the DeepSeek V4, MiniMax M3, and GLM-5
deep adaptation layer using MockAdapter (no real API calls).

Benchmark suites:
  1. CompilationBenchmark: Measure TAPRequest→CompiledPrompt compilation latency
  2. LatencyBenchmark: Measure end-to-end first-token and total latency (with MockAdapter)
  3. ContextManagementBenchmark: Test 1M context (V4/M3) and 200K compression (GLM-5)
  4. MultimodalBenchmark: Test image/video processing latency with M3
  5. LongHorizonBenchmark: Simulate long-horizon task stability
  6. CostEfficiencyBenchmark: Measure token consumption and cost metrics

Orchestration:
  BenchmarkRunner: Run all benchmarks and generate comprehensive reports

Usage::

    from teragent.benchmark import BenchmarkRunner

    runner = BenchmarkRunner(iterations=100)
    report = runner.run_all()
    print(report.to_text())
"""

from teragent.benchmark.benchmark import (
    # Core data classes
    BenchmarkMetric,
    BenchmarkResult,
    BenchmarkReport,
    # Individual benchmarks
    CompilationBenchmark,
    LatencyBenchmark,
    ContextManagementBenchmark,
    MultimodalBenchmark,
    LongHorizonBenchmark,
    CostEfficiencyBenchmark,
    # Router benchmark
    RouterBenchmark,
    # Fault recovery benchmark
    FaultRecoveryBenchmark,
    # Runner
    BenchmarkRunner,
)

__all__ = [
    "BenchmarkMetric",
    "BenchmarkResult",
    "BenchmarkReport",
    "CompilationBenchmark",
    "LatencyBenchmark",
    "ContextManagementBenchmark",
    "MultimodalBenchmark",
    "LongHorizonBenchmark",
    "CostEfficiencyBenchmark",
    "RouterBenchmark",
    "FaultRecoveryBenchmark",
    "BenchmarkRunner",
]
