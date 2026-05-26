# tests/test_pipeline_tracing.py
"""Pipeline 追踪模块单元测试

测试 teragent.pipeline.tracing 模块:
  - TraceRecord 记录
  - TAPTracer 录制请求/响应/检查单
  - DPO 配对生成（含 allow_partial）
  - DPOPair.validate() 验证
  - TraceStats 统计
  - DataConstitution 数据宪法
"""
import asyncio
import json
import os
import tempfile
import pytest

from teragent.pipeline.tracing import (
    DPOPair,
    DataConstitution,
    TAPTracer,
    TraceRecord,
    TraceStats,
)
from teragent.core.tap import TAPRequest, TAPResponse


# ===== TraceRecord 测试 =====


class TestTraceRecord:
    """TraceRecord 数据类测试"""

    def test_to_dict(self):
        """to_dict 序列化所有字段"""
        record = TraceRecord(
            trace_id="t1",
            timestamp=1000.0,
            record_type="tap_request",
            task_id="1.1",
            intent="code_generation",
            data={"key": "value"},
        )
        d = record.to_dict()
        assert d["trace_id"] == "t1"
        assert d["record_type"] == "tap_request"
        assert d["data"]["key"] == "value"

    def test_from_dict(self):
        """from_dict 反序列化"""
        d = {
            "trace_id": "t2",
            "timestamp": 2000.0,
            "record_type": "tap_response",
            "task_id": "2.1",
            "intent": "design",
            "data": {},
        }
        record = TraceRecord.from_dict(d)
        assert record.trace_id == "t2"
        assert record.record_type == "tap_response"

    def test_roundtrip(self):
        """to_dict → from_dict 往返一致"""
        original = TraceRecord(
            trace_id="t3",
            timestamp=3000.0,
            record_type="checklist_result",
            task_id="3.1",
            intent="review",
            data={"fail_count": 0},
        )
        restored = TraceRecord.from_dict(original.to_dict())
        assert restored.trace_id == original.trace_id
        assert restored.data == original.data


# ===== DPOPair 测试 =====


class TestDPOPairValidation:
    """DPOPair.validate() 测试"""

    def test_valid_full_pair(self):
        """完整的 DPO 对通过验证"""
        pair = DPOPair(
            prompt="写一个函数",
            chosen="def foo(): pass",
            rejected="# TODO",
            task_id="1.1",
            intent="code_generation",
        )
        errors = pair.validate()
        assert errors == []

    def test_missing_prompt(self):
        """缺少 prompt 报错"""
        pair = DPOPair(
            prompt="",
            chosen="good",
            rejected="bad",
            task_id="1.1",
        )
        errors = pair.validate()
        assert any("prompt" in e for e in errors)

    def test_missing_task_id(self):
        """缺少 task_id 报错"""
        pair = DPOPair(
            prompt="写代码",
            chosen="good",
            rejected="bad",
            task_id="",
        )
        errors = pair.validate()
        assert any("task_id" in e for e in errors)

    def test_empty_chosen_no_partial(self):
        """非 partial 模式下 chosen 为空报错"""
        pair = DPOPair(
            prompt="写代码",
            chosen="",
            rejected="bad",
            task_id="1.1",
        )
        errors = pair.validate(allow_partial=False)
        assert any("chosen" in e for e in errors)

    def test_empty_chosen_with_partial(self):
        """partial 模式下 chosen 为空通过"""
        pair = DPOPair(
            prompt="写代码",
            chosen="",
            rejected="bad",
            task_id="1.1",
        )
        errors = pair.validate(allow_partial=True)
        assert errors == []

    def test_both_empty_fails_even_partial(self):
        """partial 模式下 chosen 和 rejected 都为空仍报错"""
        pair = DPOPair(
            prompt="写代码",
            chosen="",
            rejected="",
            task_id="1.1",
        )
        errors = pair.validate(allow_partial=True)
        assert len(errors) > 0

    def test_identical_chosen_rejected(self):
        """chosen 和 rejected 相同报错（无偏好信号）"""
        pair = DPOPair(
            prompt="写代码",
            chosen="same",
            rejected="same",
            task_id="1.1",
        )
        errors = pair.validate()
        assert any("identical" in e.lower() for e in errors)

    def test_partial_only_chosen(self):
        """partial 模式下只有 chosen 通过验证"""
        pair = DPOPair(
            prompt="写代码",
            chosen="good code",
            rejected="",
            task_id="1.1",
        )
        errors = pair.validate(allow_partial=True)
        assert errors == []

    def test_to_dict(self):
        """to_dict 包含所有字段"""
        pair = DPOPair(
            prompt="写代码",
            chosen="good",
            rejected="bad",
            task_id="1.1",
            intent="code_generation",
        )
        d = pair.to_dict()
        assert d["prompt"] == "写代码"
        assert d["source"] == "deterministic_check"


# ===== DataConstitution 测试 =====


class TestDataConstitution:
    """DataConstitution 数据宪法测试"""

    def test_default_principles(self):
        """默认有 3 条原则"""
        constitution = DataConstitution()
        assert len(constitution.principles) == 3
        assert constitution.upload_policy == "never"

    def test_to_dict(self):
        """to_dict 包含所有字段"""
        constitution = DataConstitution()
        d = constitution.to_dict()
        assert "version" in d
        assert "principles" in d
        assert d["data_ownership"] == "user"


# ===== TraceStats 测试 =====


class TestTraceStats:
    """TraceStats 统计测试"""

    def test_default_values(self):
        """默认值为零"""
        stats = TraceStats()
        assert stats.total_records == 0
        assert stats.request_count == 0
        assert stats.pass_count == 0
        assert stats.fail_count == 0

    def test_to_dict(self):
        """to_dict 将 set 转换为 sorted list"""
        stats = TraceStats(
            total_records=5,
            task_ids={"2.1", "1.1"},
        )
        d = stats.to_dict()
        assert d["total_records"] == 5
        assert d["task_ids"] == ["1.1", "2.1"]  # sorted


# ===== TAPTracer 测试 =====


class TestTAPTracerRecording:
    """TAPTracer 录制测试"""

    @pytest.mark.asyncio
    async def test_record_request(self):
        """录制 TAP 请求"""
        tracer = TAPTracer(trace_dir=tempfile.mkdtemp(), enabled=True)
        request = TAPRequest(
            meta={"task_id": "1.1", "intent": "code_generation"},
            instruction="写一个函数",
        )
        trace_id = await tracer.record_request(request)
        assert trace_id != ""
        assert len(tracer) == 1

    @pytest.mark.asyncio
    async def test_record_response(self):
        """录制 TAP 响应"""
        tracer = TAPTracer(trace_dir=tempfile.mkdtemp(), enabled=True)
        response = TAPResponse(raw_text="def foo(): pass", usage={"prompt_tokens": 10})
        await tracer.record_response(response, task_id="1.1")
        assert len(tracer) == 1

    @pytest.mark.asyncio
    async def test_record_checklist(self):
        """录制检查单结果"""
        tracer = TAPTracer(trace_dir=tempfile.mkdtemp(), enabled=True)
        await tracer.record_checklist("1.1", {
            "fail_count": 0,
            "warn_count": 1,
            "ok_count": 4,
            "has_critical_warn": False,
            "needs_repair": False,
        })
        assert len(tracer) == 1

    @pytest.mark.asyncio
    async def test_disabled_tracer_no_records(self):
        """禁用的 tracer 不记录"""
        tracer = TAPTracer(enabled=False)
        request = TAPRequest(meta={"task_id": "1.1"}, instruction="test")
        trace_id = await tracer.record_request(request)
        assert len(tracer) == 0


class TestTAPTracerDPOPairs:
    """TAPTracer DPO 配对测试"""

    @pytest.mark.asyncio
    async def test_full_dpo_pair(self):
        """完整 DPO 配对：同一 task 有 PASS 和 FAIL"""
        tracer = TAPTracer(trace_dir=tempfile.mkdtemp(), enabled=True)

        # PASS 尝试
        req1 = TAPRequest(meta={"task_id": "1.1", "intent": "code_generation"}, instruction="写代码")
        tid1 = await tracer.record_request(req1)
        resp1 = TAPResponse(raw_text="def foo(): pass")
        await tracer.record_response(resp1, task_id="1.1", trace_id=tid1, intent="code_generation")
        await tracer.record_checklist("1.1", {
            "fail_count": 0, "warn_count": 0, "ok_count": 5,
            "has_critical_warn": False, "needs_repair": False,
        }, trace_id=tid1, intent="code_generation")

        # FAIL 尝试
        req2 = TAPRequest(meta={"task_id": "1.1", "intent": "code_generation"}, instruction="写代码")
        tid2 = await tracer.record_request(req2)
        resp2 = TAPResponse(raw_text="# TODO")
        await tracer.record_response(resp2, task_id="1.1", trace_id=tid2, intent="code_generation")
        await tracer.record_checklist("1.1", {
            "fail_count": 3, "warn_count": 0, "ok_count": 0,
            "has_critical_warn": True, "needs_repair": True,
        }, trace_id=tid2, intent="code_generation")

        pairs = tracer.export_dpo_pairs()
        assert len(pairs) >= 1
        pair = pairs[0]
        assert pair["chosen"] == "def foo(): pass"
        assert pair["rejected"] == "# TODO"

    @pytest.mark.asyncio
    async def test_cross_task_dpo_pair(self):
        """跨 task DPO 配对：不同 task_id 同一 intent"""
        tracer = TAPTracer(trace_dir=tempfile.mkdtemp(), enabled=True)

        # PASS task
        req1 = TAPRequest(meta={"task_id": "1.1", "intent": "code_generation"}, instruction="写A")
        await tracer.record_request(req1)
        resp1 = TAPResponse(raw_text="good code A")
        await tracer.record_response(resp1, task_id="1.1", intent="code_generation")
        await tracer.record_checklist("1.1", {
            "fail_count": 0, "has_critical_warn": False,
        }, intent="code_generation")

        # FAIL task
        req2 = TAPRequest(meta={"task_id": "1.2", "intent": "code_generation"}, instruction="写B")
        await tracer.record_request(req2)
        resp2 = TAPResponse(raw_text="bad code B")
        await tracer.record_response(resp2, task_id="1.2", intent="code_generation")
        await tracer.record_checklist("1.2", {
            "fail_count": 2, "has_critical_warn": True,
        }, intent="code_generation")

        pairs = tracer.export_dpo_pairs()
        assert len(pairs) >= 1
        # 应有跨 task 配对
        cross_pairs = [p for p in pairs if p.get("metadata", {}).get("pairing_strategy") == "cross_task"]
        assert len(cross_pairs) >= 1

    @pytest.mark.asyncio
    async def test_partial_dpo_pairs(self):
        """partial DPO 配对：include_partial 模式"""
        tracer = TAPTracer(trace_dir=tempfile.mkdtemp(), enabled=True)

        # 只有 PASS，没有 FAIL
        req = TAPRequest(meta={"task_id": "2.1", "intent": "design"}, instruction="设计")
        await tracer.record_request(req)
        resp = TAPResponse(raw_text="good design")
        await tracer.record_response(resp, task_id="2.1", intent="design")
        await tracer.record_checklist("2.1", {
            "fail_count": 0, "has_critical_warn": False,
        }, intent="design")

        # 不含 partial 不生成
        pairs_no_partial = tracer.export_dpo_pairs(include_partial=False)
        # 含 partial 生成
        pairs_with_partial = tracer.export_dpo_pairs(include_partial=True)
        assert len(pairs_with_partial) >= len(pairs_no_partial)


class TestTAPTracerExport:
    """TAPTracer 导出测试"""

    @pytest.mark.asyncio
    async def test_export_traces(self):
        """导出所有 trace 记录"""
        tracer = TAPTracer(trace_dir=tempfile.mkdtemp(), enabled=True)
        req = TAPRequest(meta={"task_id": "1.1"}, instruction="test")
        await tracer.record_request(req)

        traces = tracer.export_traces()
        assert len(traces) == 1
        assert traces[0]["record_type"] == "tap_request"

    @pytest.mark.asyncio
    async def test_export_traces_jsonl(self):
        """导出 JSONL 文件"""
        tracer = TAPTracer(trace_dir=tempfile.mkdtemp(), enabled=True)
        req = TAPRequest(meta={"task_id": "1.1"}, instruction="test")
        await tracer.record_request(req)

        output_path = tracer.export_traces_jsonl()
        assert os.path.isfile(output_path)
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = [l for l in f.readlines() if l.strip()]
        assert len(lines) >= 2  # constitution header + 1 record

    @pytest.mark.asyncio
    async def test_clear_records(self):
        """clear() 清空内存记录"""
        tracer = TAPTracer(trace_dir=tempfile.mkdtemp(), enabled=True)
        req = TAPRequest(meta={"task_id": "1.1"}, instruction="test")
        await tracer.record_request(req)
        assert len(tracer) == 1
        tracer.clear()
        assert len(tracer) == 0


class TestTAPTracerStats:
    """TAPTracer 统计测试"""

    @pytest.mark.asyncio
    async def test_trace_stats(self):
        """get_trace_stats 返回正确统计"""
        tracer = TAPTracer(trace_dir=tempfile.mkdtemp(), enabled=True)

        req = TAPRequest(meta={"task_id": "1.1", "intent": "code_generation"}, instruction="test")
        await tracer.record_request(req)

        resp = TAPResponse(raw_text="code")
        await tracer.record_response(resp, task_id="1.1")

        await tracer.record_checklist("1.1", {
            "fail_count": 0, "has_critical_warn": False,
        })

        stats = tracer.get_trace_stats()
        assert stats.total_records == 3
        assert stats.request_count == 1
        assert stats.response_count == 1
        assert stats.checklist_count == 1
        assert stats.pass_count == 1

    @pytest.mark.asyncio
    async def test_session_id(self):
        """session_id 非空"""
        tracer = TAPTracer(enabled=False)
        assert tracer.session_id != ""

    @pytest.mark.asyncio
    async def test_is_enabled(self):
        """is_enabled 属性"""
        tracer_on = TAPTracer(enabled=True)
        tracer_off = TAPTracer(enabled=False)
        assert tracer_on.is_enabled is True
        assert tracer_off.is_enabled is False


class TestTAPTracerLoadFromFile:
    """TAPTracer 文件加载测试"""

    @pytest.mark.asyncio
    async def test_load_from_file(self):
        """从 JSONL 文件加载 trace 记录"""
        tracer = TAPTracer(trace_dir=tempfile.mkdtemp(), enabled=True)
        req = TAPRequest(meta={"task_id": "1.1"}, instruction="test")
        await tracer.record_request(req)

        trace_file = tracer.trace_file
        new_tracer = TAPTracer(trace_dir=tempfile.mkdtemp(), enabled=False)
        count = new_tracer.load_from_file(trace_file)
        assert count == 1
        assert len(new_tracer) == 1

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self):
        """加载不存在的文件返回 0"""
        tracer = TAPTracer(enabled=False)
        count = tracer.load_from_file("/nonexistent/path/file.jsonl")
        assert count == 0
