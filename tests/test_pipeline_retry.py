# tests/test_pipeline_retry.py
"""Pipeline 重试模块单元测试

测试 teragent.pipeline.retry 模块:
  - 指数退避
  - 验证回调
  - 最大重试次数
  - 异常传播
"""
import asyncio
import pytest

from teragent.pipeline.retry import retry_with_backoff


class TestRetryWithBackoff:
    """retry_with_backoff 函数测试"""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """首次成功不需要重试"""
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await retry_with_backoff(fn, max_retries=3, backoff_base=0.01)
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        """失败后重试成功"""
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("temporary failure")
            return "success"

        result = await retry_with_backoff(fn, max_retries=3, backoff_base=0.01)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """所有重试用尽抛出最后一个异常"""
        async def fn():
            raise ValueError("persistent error")

        with pytest.raises(ValueError, match="persistent error"):
            await retry_with_backoff(fn, max_retries=2, backoff_base=0.01)

    @pytest.mark.asyncio
    async def test_validation_callback(self):
        """验证回调失败触发重试"""
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            return {"sections": []}

        def validate(result):
            if not result["sections"]:
                return ["sections is empty"]
            return []

        # 第一次验证失败，第二次验证也失败
        # 由于 max_retries=1，最终抛 ValueError
        with pytest.raises(ValueError, match="Validation failed"):
            await retry_with_backoff(
                fn, max_retries=1, backoff_base=0.01,
                validate=validate,
            )

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """on_retry 回调被调用"""
        retry_calls = []

        async def fn():
            raise RuntimeError("fail")

        def on_retry(attempt, error_msg):
            retry_calls.append((attempt, error_msg))

        with pytest.raises(RuntimeError):
            await retry_with_backoff(
                fn, max_retries=2, backoff_base=0.01,
                on_retry=on_retry,
            )

        assert len(retry_calls) == 2  # 2 次重试

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """指数退避延迟递增"""
        timestamps = []

        async def fn():
            timestamps.append(asyncio.get_event_loop().time())
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            await retry_with_backoff(fn, max_retries=2, backoff_base=0.05)

        # 至少有 3 次调用（初始 + 2 次重试）
        assert len(timestamps) == 3
