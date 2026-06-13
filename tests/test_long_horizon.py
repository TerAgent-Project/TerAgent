"""tests/test_long_horizon.py — long_horizon 模块单元测试

覆盖:
  - SubGoal / PhaseResult / LongHorizonResult 数据类
  - Checkpoint 序列化/反序列化
  - CheckpointStore 保存/加载/列表/清理
  - ProgressTracker 进度追踪
  - LongHorizonTaskManager 目标分解解析、拓扑排序、停滞检测
  - SelfEvaluator 自评估解析和触发条件
  - StrategySwitcher 策略切换检测和记录
"""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from teragent.long_horizon.types import SubGoal, PhaseResult, LongHorizonResult
from teragent.long_horizon.checkpoint import Checkpoint, CheckpointStore
from teragent.long_horizon.progress import ProgressTracker, ProgressReport
from teragent.long_horizon.task_manager import LongHorizonTaskManager
from teragent.core.tap import LongHorizonConfig


# ===== 数据类型测试 =====

class TestSubGoal:
    """SubGoal 数据类测试"""

    def test_default_values(self):
        sg = SubGoal(id="sg_1", description="设计", completion_criteria="完成", estimated_steps=5)
        assert sg.status == "pending"
        assert sg.dependencies == []

    def test_with_dependencies(self):
        sg = SubGoal(
            id="sg_2", description="实现", completion_criteria="代码完成",
            estimated_steps=10, dependencies=["sg_1"],
        )
        assert sg.dependencies == ["sg_1"]


class TestPhaseResult:
    """PhaseResult 数据类测试"""

    def test_default_values(self):
        pr = PhaseResult(sub_goal_id="sg_1", success=True, result_text="Done", steps_taken=3)
        assert pr.files_created == []
        assert pr.files_modified == []
        assert pr.errors == []

    def test_with_files_and_errors(self):
        pr = PhaseResult(
            sub_goal_id="sg_1", success=False, result_text="Error",
            steps_taken=1, files_created=["a.py"], files_modified=["b.py"],
            errors=["timeout"],
        )
        assert pr.files_created == ["a.py"]
        assert pr.errors == ["timeout"]


class TestLongHorizonResult:
    """LongHorizonResult 数据类测试"""

    def test_creation(self):
        lr = LongHorizonResult(
            task_id="t1", goal="实现系统", success=True,
            total_steps=10, total_elapsed_minutes=30.0,
            completed_sub_goals=2, total_sub_goals=3,
            strategy_switches=0, phase_results=[], final_summary="OK",
            checkpoints_saved=2,
        )
        assert lr.task_id == "t1"
        assert lr.total_sub_goals == 3


# ===== Checkpoint 测试 =====

class TestCheckpoint:
    """Checkpoint 数据类测试"""

    def test_to_dict_and_from_dict(self):
        cp = Checkpoint(
            checkpoint_id="cp_1", task_id="t1",
            timestamp="2024-01-01T00:00:00+00:00", phase="executing",
            completed_sub_goals=["sg_1"], current_sub_goal="sg_2",
            steps_completed=5, elapsed_minutes=15.0,
            strategy_switches=0, state_data={"key": "value"},
        )
        d = cp.to_dict()
        assert d["checkpoint_id"] == "cp_1"
        assert d["state_data"]["key"] == "value"

        cp2 = Checkpoint.from_dict(d)
        assert cp2.checkpoint_id == cp.checkpoint_id
        assert cp2.state_data == cp.state_data

    def test_from_dict_missing_fields(self):
        """from_dict 对缺少字段的容错处理"""
        cp = Checkpoint.from_dict({"checkpoint_id": "cp_1"})
        assert cp.task_id == ""
        assert cp.phase == "planning"
        assert cp.completed_sub_goals == []


class TestCheckpointStore:
    """CheckpointStore 测试"""

    @pytest.fixture
    def store(self, tmp_path):
        return CheckpointStore(base_dir=str(tmp_path / "checkpoints"))

    @pytest.fixture
    def sample_checkpoint(self):
        return Checkpoint(
            checkpoint_id="cp_1", task_id="t1",
            timestamp="2024-01-01T00:00:00+00:00", phase="planning",
            completed_sub_goals=[], current_sub_goal="sg_1",
            steps_completed=0, elapsed_minutes=0.0,
            strategy_switches=0, state_data={},
        )

    @pytest.mark.asyncio
    async def test_save_and_load_latest(self, store, sample_checkpoint):
        await store.save(sample_checkpoint)
        latest = await store.load_latest("t1")
        assert latest is not None
        assert latest.checkpoint_id == "cp_1"

    @pytest.mark.asyncio
    async def test_load_by_id(self, store, sample_checkpoint):
        await store.save(sample_checkpoint)
        loaded = await store.load("cp_1")
        assert loaded is not None
        assert loaded.task_id == "t1"

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, store):
        loaded = await store.load("nonexistent")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, store):
        for i in range(3):
            cp = Checkpoint(
                checkpoint_id=f"cp_{i}", task_id="t1",
                timestamp=f"2024-01-01T0{i}:00:00+00:00", phase="executing",
                completed_sub_goals=[], current_sub_goal="sg_1",
                steps_completed=i, elapsed_minutes=float(i),
                strategy_switches=0, state_data={},
            )
            await store.save(cp)

        cps = await store.list_checkpoints("t1")
        assert len(cps) == 3
        # 应按时间戳升序
        assert cps[0].checkpoint_id == "cp_0"

    @pytest.mark.asyncio
    async def test_cleanup(self, store):
        for i in range(5):
            cp = Checkpoint(
                checkpoint_id=f"cp_{i}", task_id="t1",
                timestamp=f"2024-01-01T0{i}:00:00+00:00", phase="executing",
                completed_sub_goals=[], current_sub_goal="sg_1",
                steps_completed=i, elapsed_minutes=float(i),
                strategy_switches=0, state_data={},
            )
            await store.save(cp)

        deleted = await store.cleanup("t1", keep_last=2)
        assert deleted == 3

        remaining = await store.list_checkpoints("t1")
        assert len(remaining) == 2

    @pytest.mark.asyncio
    async def test_empty_list(self, store):
        cps = await store.list_checkpoints("nonexistent")
        assert cps == []


# ===== ProgressTracker 测试 =====

class TestProgressTracker:
    """ProgressTracker 测试"""

    def test_initial_state(self):
        tracker = ProgressTracker(task_id="t1", goal="测试")
        report = tracker.get_report()
        assert report.task_id == "t1"
        assert report.completed_sub_goals == 0
        assert report.current_phase == "planning"

    def test_register_and_complete_sub_goal(self):
        tracker = ProgressTracker(task_id="t1", goal="测试")
        tracker.register_sub_goal("sg_1", "设计数据库")
        tracker.start_sub_goal("sg_1", "设计数据库")
        tracker.record_step("创建表")
        tracker.complete_sub_goal("sg_1", "表已创建")

        report = tracker.get_report()
        assert report.completed_sub_goals == 1
        assert report.steps_completed == 1
        assert report.total_sub_goals == 1

    def test_fail_sub_goal(self):
        tracker = ProgressTracker(task_id="t1", goal="测试")
        tracker.register_sub_goal("sg_1", "设计")
        tracker.start_sub_goal("sg_1", "设计")
        tracker.fail_sub_goal("sg_1", "超时")

        report = tracker.get_report()
        assert report.completed_sub_goals == 0
        sg = report.sub_goal_statuses[0]
        assert sg["status"] == "failed"

    def test_strategy_switch(self):
        tracker = ProgressTracker(task_id="t1", goal="测试")
        tracker.record_strategy_switch("连续3次相同结果")

        report = tracker.get_report()
        assert report.strategy_switches == 1
        assert report.current_phase == "stagnant"

    def test_elapsed_minutes(self):
        tracker = ProgressTracker(task_id="t1", goal="测试")
        elapsed = tracker.get_elapsed_minutes()
        assert elapsed >= 0.0

    def test_estimated_remaining(self):
        tracker = ProgressTracker(task_id="t1", goal="测试")
        tracker.register_sub_goal("sg_1", "设计")
        tracker.register_sub_goal("sg_2", "实现")

        # 完成一个子目标后，应有预估剩余时间
        tracker.start_sub_goal("sg_1", "设计")
        tracker.complete_sub_goal("sg_1", "完成")

        report = tracker.get_report()
        # 至少有一个子目标未完成，预估时间应 >= 0
        assert report.estimated_remaining_minutes >= 0.0

    def test_set_phase_and_checkpoint(self):
        tracker = ProgressTracker(task_id="t1", goal="测试")
        tracker.set_phase("evaluating")
        assert tracker.current_phase == "evaluating"

        tracker.set_last_checkpoint("cp_123")
        report = tracker.get_report()
        assert report.last_checkpoint == "cp_123"


# ===== LongHorizonTaskManager 解析辅助方法测试 =====

class TestLongHorizonTaskManagerParsing:
    """LongHorizonTaskManager 解析方法测试"""

    def test_parse_sub_goals_json(self):
        content = json.dumps([
            {"id": "sg_1", "description": "设计", "completion_criteria": "完成", "estimated_steps": 3},
            {"id": "sg_2", "description": "实现", "completion_criteria": "代码完成", "estimated_steps": 5, "dependencies": ["sg_1"]},
        ])
        result = LongHorizonTaskManager._parse_sub_goals_json(content)
        assert len(result) == 2
        assert result[0].id == "sg_1"
        assert result[1].dependencies == ["sg_1"]

    def test_parse_sub_goals_json_with_markdown(self):
        content = "```json\n" + json.dumps([
            {"id": "sg_1", "description": "测试", "completion_criteria": "通过", "estimated_steps": 2},
        ]) + "\n```"
        result = LongHorizonTaskManager._parse_sub_goals_json(content)
        assert len(result) == 1

    def test_parse_sub_goals_json_invalid(self):
        result = LongHorizonTaskManager._parse_sub_goals_json("not json")
        assert result == []

    def test_parse_sub_goals_json_empty_array(self):
        result = LongHorizonTaskManager._parse_sub_goals_json("[]")
        assert result == []

    def test_estimate_steps_from_response(self):
        content = "步骤1: 创建文件\n步骤2: 编写代码\n步骤3: 测试"
        steps = LongHorizonTaskManager._estimate_steps_from_response(content)
        assert steps >= 1

    def test_estimate_steps_empty(self):
        steps = LongHorizonTaskManager._estimate_steps_from_response("")
        assert steps == 1  # 最少1步

    def test_extract_files_from_response(self):
        content = '<file path="main.py">code</file>'
        result = LongHorizonTaskManager._extract_files_from_response(content)
        assert "main.py" in result["created"]

    def test_parse_evaluation_json(self):
        content = json.dumps({"assessment": "on_track", "summary": "进展顺利", "recommendation": "继续"})
        result = LongHorizonTaskManager._parse_evaluation_json(content)
        assert result["assessment"] == "on_track"

    def test_parse_evaluation_json_invalid(self):
        result = LongHorizonTaskManager._parse_evaluation_json("not json")
        assert result["assessment"] == "on_track"  # 默认值

    def test_format_progress_context(self):
        tracker = ProgressTracker(task_id="t1", goal="测试")
        tracker.register_sub_goal("sg_1", "设计")
        tracker.start_sub_goal("sg_1", "设计")
        tracker.complete_sub_goal("sg_1", "完成")
        report = tracker.get_report()
        context = LongHorizonTaskManager._format_progress_context(report)
        assert "1/1" in context


class TestLongHorizonTaskManagerTopology:
    """LongHorizonTaskManager 拓扑排序测试"""

    def test_simple_chain(self):
        from teragent.core.adapters.mock import MockAdapter
        from teragent.core.compilers.default import DefaultCompiler
        from teragent.core.provider import ModelProvider

        compiler = DefaultCompiler()
        adapter = MockAdapter()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="test")

        manager = LongHorizonTaskManager(goal="测试", model_provider=provider)
        manager._sub_goals = [
            SubGoal(id="sg_1", description="第一步", completion_criteria="完成", estimated_steps=1),
            SubGoal(id="sg_2", description="第二步", completion_criteria="完成", estimated_steps=1, dependencies=["sg_1"]),
            SubGoal(id="sg_3", description="第三步", completion_criteria="完成", estimated_steps=1, dependencies=["sg_2"]),
        ]

        order = manager._topological_sort()
        ids = [sg.id for sg in order]
        assert ids.index("sg_1") < ids.index("sg_2")
        assert ids.index("sg_2") < ids.index("sg_3")

    def test_parallel_sub_goals(self):
        from teragent.core.adapters.mock import MockAdapter
        from teragent.core.compilers.default import DefaultCompiler
        from teragent.core.provider import ModelProvider

        compiler = DefaultCompiler()
        adapter = MockAdapter()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="test")

        manager = LongHorizonTaskManager(goal="测试", model_provider=provider)
        manager._sub_goals = [
            SubGoal(id="sg_1", description="基础", completion_criteria="完成", estimated_steps=1),
            SubGoal(id="sg_2", description="分支A", completion_criteria="完成", estimated_steps=1, dependencies=["sg_1"]),
            SubGoal(id="sg_3", description="分支B", completion_criteria="完成", estimated_steps=1, dependencies=["sg_1"]),
        ]

        order = manager._topological_sort()
        ids = [sg.id for sg in order]
        assert ids.index("sg_1") < ids.index("sg_2")
        assert ids.index("sg_1") < ids.index("sg_3")

    def test_no_sub_goals(self):
        from teragent.core.adapters.mock import MockAdapter
        from teragent.core.compilers.default import DefaultCompiler
        from teragent.core.provider import ModelProvider

        compiler = DefaultCompiler()
        adapter = MockAdapter()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="test")

        manager = LongHorizonTaskManager(goal="测试", model_provider=provider)
        assert manager._topological_sort() == []


class TestLongHorizonTaskManagerStagnation:
    """LongHorizonTaskManager 停滞检测测试"""

    def test_no_stagnation_initial(self):
        from teragent.core.adapters.mock import MockAdapter
        from teragent.core.compilers.default import DefaultCompiler
        from teragent.core.provider import ModelProvider

        compiler = DefaultCompiler()
        adapter = MockAdapter()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="test")

        manager = LongHorizonTaskManager(goal="测试", model_provider=provider)
        pr = PhaseResult(sub_goal_id="sg_1", success=True, result_text="OK", steps_taken=1)
        assert not manager._detect_stagnation(pr)

    def test_stagnation_identical_results(self):
        from teragent.core.adapters.mock import MockAdapter
        from teragent.core.compilers.default import DefaultCompiler
        from teragent.core.provider import ModelProvider

        compiler = DefaultCompiler()
        adapter = MockAdapter()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="test")

        config = LongHorizonConfig(stagnation_threshold=3)
        manager = LongHorizonTaskManager(goal="测试", model_provider=provider, config=config)

        # 模拟连续3次相同结果
        manager._recent_result_summaries = ["same result", "same result", "same result"]
        pr = PhaseResult(sub_goal_id="sg_1", success=True, result_text="same result", steps_taken=1)
        assert manager._detect_stagnation(pr)

    def test_stagnation_very_short_results(self):
        from teragent.core.adapters.mock import MockAdapter
        from teragent.core.compilers.default import DefaultCompiler
        from teragent.core.provider import ModelProvider

        compiler = DefaultCompiler()
        adapter = MockAdapter()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="test")

        config = LongHorizonConfig(stagnation_threshold=3)
        manager = LongHorizonTaskManager(goal="测试", model_provider=provider, config=config)

        # 模拟连续3次极短结果
        manager._recent_result_summaries = ["a", "b", "c"]
        pr = PhaseResult(sub_goal_id="sg_1", success=True, result_text="d", steps_taken=1)
        assert manager._detect_stagnation(pr)

    def test_manager_has_self_evaluator_and_strategy_switcher(self):
        """LongHorizonTaskManager 默认创建 SelfEvaluator 和 StrategySwitcher"""
        from teragent.core.adapters.mock import MockAdapter
        from teragent.core.compilers.default import DefaultCompiler
        from teragent.core.provider import ModelProvider
        from teragent.long_horizon.self_evaluation import SelfEvaluator
        from teragent.long_horizon.strategy_switch import StrategySwitcher

        compiler = DefaultCompiler()
        adapter = MockAdapter()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="test")

        manager = LongHorizonTaskManager(goal="测试", model_provider=provider)
        assert isinstance(manager.self_evaluator, SelfEvaluator)
        assert isinstance(manager.strategy_switcher, StrategySwitcher)

    def test_manager_custom_evaluator_and_switcher(self):
        """LongHorizonTaskManager 可接受自定义 SelfEvaluator 和 StrategySwitcher"""
        from teragent.core.adapters.mock import MockAdapter
        from teragent.core.compilers.default import DefaultCompiler
        from teragent.core.provider import ModelProvider
        from teragent.long_horizon.self_evaluation import SelfEvaluator
        from teragent.long_horizon.strategy_switch import StrategySwitcher

        compiler = DefaultCompiler()
        adapter = MockAdapter()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="test")

        evaluator = SelfEvaluator(model_provider=provider, evaluation_interval_steps=5)
        switcher = StrategySwitcher(model_provider=provider, stagnation_threshold=2)

        manager = LongHorizonTaskManager(
            goal="测试",
            model_provider=provider,
            self_evaluator=evaluator,
            strategy_switcher=switcher,
        )
        assert manager.self_evaluator is evaluator
        assert manager.strategy_switcher is switcher


# ===== SelfEvaluator 测试 =====

class TestSelfEvaluationResult:
    """SelfEvaluationResult 数据类测试"""

    def test_creation(self):
        from teragent.long_horizon.self_evaluation import SelfEvaluationResult

        result = SelfEvaluationResult(
            goal_alignment=4,
            output_quality=3,
            bottleneck_identified="缺少测试覆盖",
            strategy_review="当前策略基本有效",
            next_step_plan="增加测试用例",
            overall_score=3.6,
            should_switch_strategy=False,
            raw_response="test",
        )
        assert result.goal_alignment == 4
        assert result.output_quality == 3
        assert not result.should_switch_strategy

    def test_switch_strategy_flag(self):
        from teragent.long_horizon.self_evaluation import SelfEvaluationResult

        result = SelfEvaluationResult(
            goal_alignment=1,
            output_quality=2,
            bottleneck_identified="严重偏移",
            strategy_review="策略失效",
            next_step_plan="重新规划",
            overall_score=1.4,
            should_switch_strategy=True,
            raw_response="test",
        )
        assert result.should_switch_strategy


class TestSelfEvaluator:
    """SelfEvaluator 测试"""

    @pytest.fixture
    def provider(self):
        from teragent.core.adapters.mock import MockAdapter
        from teragent.core.compilers.default import DefaultCompiler
        from teragent.core.provider import ModelProvider

        compiler = DefaultCompiler()
        adapter = MockAdapter()
        return ModelProvider(compiler=compiler, adapter=adapter, model="test")

    def test_should_evaluate_by_steps(self, provider):
        from teragent.long_horizon.self_evaluation import SelfEvaluator

        evaluator = SelfEvaluator(
            model_provider=provider,
            evaluation_interval_steps=10,
            evaluation_interval_minutes=30.0,
        )
        assert not evaluator.should_evaluate(steps_since_last=5, minutes_since_last=10.0)
        assert evaluator.should_evaluate(steps_since_last=10, minutes_since_last=0.0)
        assert evaluator.should_evaluate(steps_since_last=15, minutes_since_last=0.0)

    def test_should_evaluate_by_time(self, provider):
        from teragent.long_horizon.self_evaluation import SelfEvaluator

        evaluator = SelfEvaluator(
            model_provider=provider,
            evaluation_interval_steps=100,
            evaluation_interval_minutes=30.0,
        )
        assert not evaluator.should_evaluate(steps_since_last=5, minutes_since_last=20.0)
        assert evaluator.should_evaluate(steps_since_last=0, minutes_since_last=30.0)
        assert evaluator.should_evaluate(steps_since_last=0, minutes_since_last=60.0)

    def test_build_evaluation_prompt(self, provider):
        from teragent.long_horizon.self_evaluation import SelfEvaluator

        evaluator = SelfEvaluator(model_provider=provider)
        tracker = ProgressTracker(task_id="t1", goal="实现系统")
        tracker.register_sub_goal("sg_1", "设计")
        tracker.start_sub_goal("sg_1", "设计")
        tracker.complete_sub_goal("sg_1", "完成")
        report = tracker.get_report()

        prompt = evaluator._build_evaluation_prompt(
            goal="实现系统", progress_report=report, recent_results=[]
        )
        assert "实现系统" in prompt
        assert "目标对齐度" in prompt or "goal_alignment" in prompt

    def test_parse_json_response(self, provider):
        from teragent.long_horizon.self_evaluation import SelfEvaluator

        evaluator = SelfEvaluator(model_provider=provider)
        response = json.dumps({
            "goal_alignment": 4,
            "output_quality": 3,
            "bottleneck_identified": "缺少文档",
            "strategy_review": "策略有效",
            "next_step_plan": "补充文档",
            "should_switch_strategy": False,
        })
        result = evaluator._parse_evaluation_response(response)
        assert result.goal_alignment == 4
        assert result.output_quality == 3
        assert not result.should_switch_strategy

    def test_parse_json_with_markdown(self, provider):
        from teragent.long_horizon.self_evaluation import SelfEvaluator

        evaluator = SelfEvaluator(model_provider=provider)
        data = {
            "goal_alignment": 2,
            "output_quality": 1,
            "bottleneck_identified": "严重偏离",
            "strategy_review": "策略失效",
            "next_step_plan": "重新规划",
            "should_switch_strategy": True,
        }
        response = "```json\n" + json.dumps(data) + "\n```"
        result = evaluator._parse_evaluation_response(response)
        assert result.goal_alignment == 2
        assert result.should_switch_strategy

    def test_parse_auto_switch_when_low_score(self, provider):
        """评分低时自动触发策略切换建议"""
        from teragent.long_horizon.self_evaluation import SelfEvaluator

        evaluator = SelfEvaluator(model_provider=provider)
        response = json.dumps({
            "goal_alignment": 1,
            "output_quality": 2,
            "bottleneck_identified": "方向错误",
            "strategy_review": "无效",
            "next_step_plan": "重新开始",
            "should_switch_strategy": False,  # 即使模型说不切换
        })
        result = evaluator._parse_evaluation_response(response)
        assert result.should_switch_strategy  # 低分自动触发

    def test_parse_empty_response(self, provider):
        from teragent.long_horizon.self_evaluation import SelfEvaluator

        evaluator = SelfEvaluator(model_provider=provider)
        result = evaluator._parse_evaluation_response("")
        assert result.goal_alignment == 3  # 默认值
        assert not result.should_switch_strategy

    def test_heuristic_parse(self, provider):
        """启发式解析测试"""
        from teragent.long_horizon.self_evaluation import SelfEvaluator

        evaluator = SelfEvaluator(model_provider=provider)
        response = (
            "目标对齐度：3\n"
            "产出质量：2\n"
            "瓶颈：缺少单元测试\n"
            "策略审查：需要调整策略\n"
            "下一步规划：添加测试\n"
            "建议切换策略"
        )
        result = evaluator._parse_evaluation_response(response)
        assert result.goal_alignment == 3
        assert result.output_quality == 2
        assert result.should_switch_strategy

    def test_clamp_score(self, provider):
        from teragent.long_horizon.self_evaluation import SelfEvaluator

        evaluator = SelfEvaluator(model_provider=provider)
        assert evaluator._clamp_score(0) == 1
        assert evaluator._clamp_score(6) == 5
        assert evaluator._clamp_score(3) == 3


# ===== StrategySwitcher 测试 =====

class TestStrategySwitchRecord:
    """StrategySwitchRecord 数据类测试"""

    def test_creation(self):
        from teragent.long_horizon.strategy_switch import StrategySwitchRecord

        record = StrategySwitchRecord(
            timestamp="2024-01-01T00:00:00+00:00",
            reason="连续3次相似结果",
            previous_strategy="初始策略",
            new_strategy="分解细化",
            risk_assessment="低风险",
        )
        assert record.effectiveness == ""  # 初始为空
        record.effectiveness = "有效"
        assert record.effectiveness == "有效"


class TestStrategySwitcher:
    """StrategySwitcher 测试"""

    @pytest.fixture
    def provider(self):
        from teragent.core.adapters.mock import MockAdapter
        from teragent.core.compilers.default import DefaultCompiler
        from teragent.core.provider import ModelProvider

        compiler = DefaultCompiler()
        adapter = MockAdapter()
        return ModelProvider(compiler=compiler, adapter=adapter, model="test")

    def test_detect_no_stagnation(self, provider):
        from teragent.long_horizon.strategy_switch import StrategySwitcher

        switcher = StrategySwitcher(model_provider=provider, stagnation_threshold=3)
        results = [
            PhaseResult(sub_goal_id="sg_1", success=True, result_text="完成设计", steps_taken=1),
            PhaseResult(sub_goal_id="sg_2", success=True, result_text="完成编码", steps_taken=2),
        ]
        is_stagnant, reason = switcher.detect_stagnation(results, ["步骤1", "步骤2"])
        assert not is_stagnant

    def test_detect_stagnation_similar_results(self, provider):
        """连续N次结果高度相似时检测到停滞"""
        from teragent.long_horizon.strategy_switch import StrategySwitcher

        switcher = StrategySwitcher(
            model_provider=provider,
            stagnation_threshold=3,
            similarity_threshold=0.5,
        )
        # 创建高度相似的结果
        same_text = "同样的输出结果没有任何变化"
        results = [
            PhaseResult(sub_goal_id=f"sg_{i}", success=True, result_text=same_text, steps_taken=1)
            for i in range(4)
        ]
        is_stagnant, reason = switcher.detect_stagnation(results, [])
        assert is_stagnant
        assert "相似" in reason

    def test_detect_stagnation_no_files(self, provider):
        """连续M步无新文件产出时检测到停滞"""
        from teragent.long_horizon.strategy_switch import StrategySwitcher

        switcher = StrategySwitcher(
            model_provider=provider,
            no_progress_threshold=3,
        )
        results = [
            PhaseResult(sub_goal_id=f"sg_{i}", success=True, result_text=f"不同的结果{i}", steps_taken=1)
            for i in range(4)
        ]
        is_stagnant, reason = switcher.detect_stagnation(results, [])
        assert is_stagnant
        assert "文件" in reason

    def test_detect_stagnation_consecutive_failures(self, provider):
        """连续N次执行失败时检测到停滞"""
        from teragent.long_horizon.strategy_switch import StrategySwitcher

        switcher = StrategySwitcher(model_provider=provider, stagnation_threshold=3)
        results = [
            PhaseResult(sub_goal_id=f"sg_{i}", success=False, result_text="", steps_taken=1, errors=["失败"])
            for i in range(4)
        ]
        is_stagnant, reason = switcher.detect_stagnation(results, [])
        assert is_stagnant
        assert "失败" in reason

    def test_calculate_similarity_identical(self, provider):
        from teragent.long_horizon.strategy_switch import StrategySwitcher

        switcher = StrategySwitcher(model_provider=provider)
        sim = switcher._calculate_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_calculate_similarity_empty(self, provider):
        from teragent.long_horizon.strategy_switch import StrategySwitcher

        switcher = StrategySwitcher(model_provider=provider)
        sim = switcher._calculate_similarity("", "")
        assert sim == 1.0

    def test_calculate_similarity_different(self, provider):
        from teragent.long_horizon.strategy_switch import StrategySwitcher

        switcher = StrategySwitcher(model_provider=provider)
        sim = switcher._calculate_similarity("alpha beta", "gamma delta")
        assert sim < 0.5

    def test_calculate_similarity_chinese(self, provider):
        """中文文本相似度计算"""
        from teragent.long_horizon.strategy_switch import StrategySwitcher

        switcher = StrategySwitcher(model_provider=provider)
        sim = switcher._calculate_similarity("创建用户表", "创建用户表")
        assert sim == 1.0

        sim2 = switcher._calculate_similarity("创建用户表", "删除订单表")
        assert 0.0 < sim2 < 1.0  # 部分相似（"创建" vs "删除"，"表"相同）

    def test_get_switch_history_empty(self, provider):
        from teragent.long_horizon.strategy_switch import StrategySwitcher

        switcher = StrategySwitcher(model_provider=provider)
        assert switcher.get_switch_history() == []

    def test_assess_switch_effectiveness_valid_index(self, provider):
        from teragent.long_horizon.strategy_switch import StrategySwitcher, StrategySwitchRecord

        switcher = StrategySwitcher(model_provider=provider)
        switcher._switch_history = [
            StrategySwitchRecord(
                timestamp="2024-01-01T00:00:00+00:00",
                reason="停滞",
                previous_strategy="旧策略",
                new_strategy="新策略",
                risk_assessment="低",
            )
        ]
        subsequent = [
            PhaseResult(sub_goal_id="sg_1", success=True, result_text="OK", steps_taken=1,
                        files_created=["a.py"]),
            PhaseResult(sub_goal_id="sg_2", success=True, result_text="Done", steps_taken=1,
                        files_created=["b.py"]),
        ]
        effectiveness = switcher.assess_switch_effectiveness(0, subsequent)
        assert "有效" in effectiveness
        assert switcher._switch_history[0].effectiveness != ""

    def test_assess_switch_effectiveness_invalid_index(self, provider):
        from teragent.long_horizon.strategy_switch import StrategySwitcher

        switcher = StrategySwitcher(model_provider=provider)
        effectiveness = switcher.assess_switch_effectiveness(99, [])
        assert "无效" in effectiveness

    def test_assess_switch_effectiveness_no_results(self, provider):
        from teragent.long_horizon.strategy_switch import StrategySwitcher, StrategySwitchRecord

        switcher = StrategySwitcher(model_provider=provider)
        switcher._switch_history = [
            StrategySwitchRecord(
                timestamp="2024-01-01T00:00:00+00:00",
                reason="停滞",
                previous_strategy="旧",
                new_strategy="新",
                risk_assessment="低",
            )
        ]
        effectiveness = switcher.assess_switch_effectiveness(0, [])
        assert "无" in effectiveness

    def test_current_strategy_property(self, provider):
        from teragent.long_horizon.strategy_switch import StrategySwitcher

        switcher = StrategySwitcher(model_provider=provider)
        assert switcher.current_strategy == "初始策略"

    def test_parse_strategy_response_json(self, provider):
        from teragent.long_horizon.strategy_switch import StrategySwitcher

        switcher = StrategySwitcher(model_provider=provider)
        response = json.dumps({
            "new_strategy": "分解细化策略",
            "risk_assessment": "低风险：已完成的进度可保留",
            "rationale": "当前子目标过大，需要拆分",
        })
        new_strategy, risk = switcher._parse_strategy_response(response, "旧策略", "停滞")
        assert "分解细化" in new_strategy
        assert "低风险" in risk

    def test_parse_strategy_response_heuristic(self, provider):
        from teragent.long_horizon.strategy_switch import StrategySwitcher

        switcher = StrategySwitcher(model_provider=provider)
        response = "新策略：增量验证方法\n风险：中等，可能需要回退部分工作"
        new_strategy, risk = switcher._parse_strategy_response(response, "旧策略", "停滞")
        assert "增量验证" in new_strategy
        assert "中等" in risk

    def test_parse_strategy_response_empty(self, provider):
        from teragent.long_horizon.strategy_switch import StrategySwitcher

        switcher = StrategySwitcher(model_provider=provider)
        new_strategy, risk = switcher._parse_strategy_response("", "旧策略", "停滞")
        assert "调整执行策略" in new_strategy
