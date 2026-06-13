# tests/test_utils_tracing.py
"""分布式追踪工具单元测试

测试 @experimental 装饰器、span 生命周期、request_id 管理等。
"""
import warnings

from teragent.utils.tracing import (
    Span,
    end_span,
    experimental,
    get_all_spans,
    get_request_id,
    get_span_summary,
    reset_request_id,
    set_request_id,
    start_span,
)

# ===== @experimental 装饰器 =====

class TestExperimentalDecorator:
    """@experimental 装饰器：首次调用警告，后续不警告"""

    def test_warns_on_first_call(self):
        """首次调用发出 UserWarning"""
        @experimental
        def my_func():
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = my_func()
            assert result == 42
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "experimental" in str(w[0].message)
            assert "my_func" in str(w[0].message)

    def test_no_warn_on_subsequent_calls(self):
        """后续调用不再发出警告"""
        call_count = 0

        @experimental
        def my_func2():
            nonlocal call_count
            call_count += 1
            return call_count

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            my_func2()  # 第一次：警告
            my_func2()  # 第二次：无警告
            my_func2()  # 第三次：无警告
            assert len(w) == 1  # 只有第一次的警告

    def test_preserves_function_name(self):
        """保留原函数名"""
        @experimental
        def named_func():
            """原始文档"""
            pass

        assert named_func.__name__ == "named_func"
        assert named_func.__doc__ == "原始文档"

    def test_separate_functions_warn_independently(self):
        """不同函数各自独立警告"""
        @experimental
        def func_a():
            return "a"

        @experimental
        def func_b():
            return "b"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            func_a()
            func_b()
            assert len(w) == 2  # 每个函数各警告一次


# ===== Span 生命周期 =====

class TestSpanLifecycle:
    """start_span / end_span 生命周期"""

    def test_start_span_creates_span(self):
        """start_span 创建 Span 对象"""
        set_request_id("test-req")
        span = start_span("operation_1")
        assert isinstance(span, Span)
        assert span.operation == "operation_1"
        assert span.start_time > 0
        assert span.end_time == 0.0
        assert span.request_id == "test-req"
        reset_request_id()

    def test_end_span_sets_end_time(self):
        """end_span 记录结束时间和状态"""
        set_request_id("test-req")
        span = start_span("op")
        end_span(span, status="ok")
        assert span.end_time > 0
        assert span.status == "ok"
        assert span.duration_ms > 0
        reset_request_id()

    def test_end_span_with_error_status(self):
        """end_span 可设置错误状态"""
        set_request_id("test-req")
        span = start_span("failing_op")
        end_span(span, status="error")
        assert span.status == "error"
        reset_request_id()

    def test_span_parent_child(self):
        """子 span 关联父 span"""
        set_request_id("test-req")
        parent = start_span("parent_op")
        child = start_span("child_op")
        assert child.parent_id == parent.span_id
        end_span(child)
        end_span(parent)
        reset_request_id()

    def test_span_attributes(self):
        """span 可携带属性"""
        set_request_id("test-req")
        span = start_span("with_attrs", attributes={"key": "value", "count": 42})
        assert span.attributes["key"] == "value"
        assert span.attributes["count"] == 42
        reset_request_id()


# ===== get_all_spans =====

class TestGetAllSpans:
    """get_all_spans 返回正确数据"""

    def test_returns_all_spans_as_dicts(self):
        """返回所有 span 的字典形式"""
        set_request_id("test-req")
        s1 = start_span("op1")
        s2 = start_span("op2")
        end_span(s2)
        end_span(s1)

        all_spans = get_all_spans()
        assert len(all_spans) == 2
        # 每个 span 是 dict
        for s_dict in all_spans:
            assert "span_id" in s_dict
            assert "operation" in s_dict
            assert "duration_ms" in s_dict

        reset_request_id()

    def test_empty_when_no_spans(self):
        """无 span 时返回空列表"""
        set_request_id("test-req-empty")
        assert get_all_spans() == []
        reset_request_id()


# ===== get_span_summary =====

class TestGetSpanSummary:
    """get_span_summary 返回正确摘要"""

    def test_summary_counts(self):
        """摘要统计正确"""
        set_request_id("test-req-summary")
        s1 = start_span("op1")
        s2 = start_span("op2")
        end_span(s1, status="ok")
        end_span(s2, status="error")

        summary = get_span_summary()
        assert summary["total_spans"] == 2
        assert summary["error_count"] == 1
        assert summary["total_duration_ms"] > 0
        assert summary["request_id"] == "test-req-summary"

        reset_request_id()


# ===== request_id 管理 =====

class TestRequestId:
    """get_request_id / set_request_id / reset_request_id"""

    def test_initial_request_id_empty(self):
        """初始 request_id 为空"""
        reset_request_id()
        assert get_request_id() == ""

    def test_set_request_id_with_value(self):
        """手动设置 request_id"""
        rid = set_request_id("my-request-123")
        assert rid == "my-request-123"
        assert get_request_id() == "my-request-123"
        reset_request_id()

    def test_set_request_id_auto_generate(self):
        """自动生成 request_id"""
        rid = set_request_id()
        assert rid != ""
        assert len(rid) > 0
        assert get_request_id() == rid
        reset_request_id()

    def test_reset_request_id(self):
        """重置 request_id 为空"""
        set_request_id("to-be-reset")
        reset_request_id()
        assert get_request_id() == ""


# ===== Span.to_dict =====

class TestSpanToDict:
    """Span.to_dict 方法"""

    def test_to_dict_keys(self):
        """to_dict 包含所有字段"""
        span = Span(
            span_id="abc",
            operation="test",
            start_time=1.0,
            end_time=2.0,
            parent_id="",
            request_id="req1",
            attributes={"k": "v"},
            status="ok",
        )
        d = span.to_dict()
        assert d["span_id"] == "abc"
        assert d["operation"] == "test"
        assert d["duration_ms"] == 1000.0
        assert d["status"] == "ok"
        assert d["attributes"] == {"k": "v"}

    def test_duration_ms_ongoing_span(self):
        """未结束的 span duration_ms 为 0"""
        span = Span(span_id="x", operation="y", start_time=1.0)
        assert span.duration_ms == 0.0
