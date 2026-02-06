from __future__ import annotations

import time

import pytest

from knowledge3d.knowledgeverse.resilience import (
    CircuitBreakerOpen,
    SelfHealingWrapper,
)


def test_retry_succeeds_on_second_attempt():
    attempts = [0]

    @SelfHealingWrapper.with_retry(max_attempts=3, backoff_base=0.001)
    def flaky_function():
        attempts[0] += 1
        if attempts[0] < 2:
            raise ValueError("Transient failure")
        return "success"

    result = flaky_function()
    assert result == "success"
    assert attempts[0] == 2


def test_retry_exhausts_after_max_attempts():
    attempts = [0]

    @SelfHealingWrapper.with_retry(max_attempts=3, backoff_base=0.001)
    def broken_function():
        attempts[0] += 1
        raise ValueError("Permanent failure")

    with pytest.raises(ValueError, match="Permanent failure"):
        broken_function()

    assert attempts[0] == 3


def test_circuit_breaker_opens_after_threshold():
    call_count = [0]

    @SelfHealingWrapper.circuit_breaker(failure_threshold=3, timeout=60.0)
    def broken_function():
        call_count[0] += 1
        raise ValueError("Always fails")

    for _ in range(3):
        with pytest.raises(ValueError):
            broken_function()

    with pytest.raises(CircuitBreakerOpen):
        broken_function()

    assert call_count[0] == 3


def test_circuit_breaker_recovers_after_timeout():
    call_count = [0]

    @SelfHealingWrapper.circuit_breaker(failure_threshold=2, timeout=0.1)
    def recovering_function():
        call_count[0] += 1
        if call_count[0] <= 2:
            raise ValueError("Initial failures")
        return "recovered"

    for _ in range(2):
        with pytest.raises(ValueError):
            recovering_function()

    with pytest.raises(CircuitBreakerOpen):
        recovering_function()

    time.sleep(0.15)
    result = recovering_function()
    assert result == "recovered"


def test_fallback_uses_cache():
    call_count = [0]

    def fallback():
        return "fallback_result"

    @SelfHealingWrapper.with_fallback(
        fallback_func=fallback,
        cache_duration=1.0,
    )
    def unreliable_function():
        call_count[0] += 1
        if call_count[0] == 1:
            return "cached_result"
        raise ValueError("Failed")

    result1 = unreliable_function()
    assert result1 == "cached_result"

    result2 = unreliable_function()
    assert result2 == "cached_result"


def test_fallback_expires_after_cache_duration():
    call_count = [0]

    def fallback():
        return "fallback_result"

    @SelfHealingWrapper.with_fallback(
        fallback_func=fallback,
        cache_duration=0.1,
    )
    def unreliable_function():
        call_count[0] += 1
        if call_count[0] == 1:
            return "cached_result"
        raise ValueError("Failed")

    result1 = unreliable_function()
    assert result1 == "cached_result"

    time.sleep(0.15)
    result2 = unreliable_function()
    assert result2 == "fallback_result"


def test_combined_patterns():
    attempts = [0]

    @SelfHealingWrapper.circuit_breaker(failure_threshold=2, timeout=60.0)
    @SelfHealingWrapper.with_retry(max_attempts=2, backoff_base=0.001)
    def combined_function():
        attempts[0] += 1
        raise ValueError("Failing")

    with pytest.raises(ValueError, match="Failing"):
        combined_function()

    with pytest.raises(ValueError, match="Failing"):
        combined_function()

    with pytest.raises(CircuitBreakerOpen):
        combined_function()

    # two calls reached function body with two retries each; third call fail-fast
    assert attempts[0] == 4

