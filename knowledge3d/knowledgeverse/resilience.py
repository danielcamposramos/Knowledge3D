"""Self-healing wrappers for Knowledgeverse critical operations."""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is OPEN and fails fast."""


@dataclass
class _CircuitState:
    status: str = "CLOSED"  # CLOSED -> OPEN -> HALF_OPEN -> CLOSED
    failures: int = 0
    opened_at: Optional[float] = None


class SelfHealingWrapper:
    """Decorator collection for runtime resilience."""

    @staticmethod
    def with_retry(
        max_attempts: int = 3,
        backoff_base: float = 2.0,
        exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Retry with exponential backoff.

        Delay schedule:
        `delay = backoff_base * (2 ** attempt)` for retry attempt index starting at 0.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if backoff_base < 0:
            raise ValueError("backoff_base must be >= 0")

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                last_exception: Exception | None = None
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as exc:  # type: ignore[misc]
                        last_exception = exc
                        if attempt >= max_attempts - 1:
                            break
                        delay = backoff_base * (2**attempt)
                        print(
                            f"[SelfHealing] Retry {attempt + 1}/{max_attempts} "
                            f"after {delay:.3f}s"
                        )
                        if delay > 0:
                            time.sleep(delay)

                assert last_exception is not None
                raise last_exception

            return wrapper

        return decorator

    @staticmethod
    def circuit_breaker(
        failure_threshold: int = 5,
        timeout: float = 60.0,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Circuit breaker with OPEN/HALF_OPEN/CLOSED states."""
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if timeout < 0:
            raise ValueError("timeout must be >= 0")

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            state = _CircuitState()

            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                now = time.time()

                if state.status == "OPEN":
                    assert state.opened_at is not None
                    if now - state.opened_at > timeout:
                        state.status = "HALF_OPEN"
                        print("[SelfHealing] Circuit HALF_OPEN (testing recovery)")
                    else:
                        raise CircuitBreakerOpen("Circuit breaker OPEN (fail-fast)")

                try:
                    result = func(*args, **kwargs)
                except Exception:
                    state.failures += 1
                    if state.failures >= failure_threshold:
                        state.status = "OPEN"
                        state.opened_at = time.time()
                        print(
                            f"[SelfHealing] Circuit OPEN "
                            f"(threshold: {failure_threshold})"
                        )
                    raise

                # Success path
                if state.status == "HALF_OPEN":
                    print("[SelfHealing] Circuit CLOSED (recovered)")
                state.status = "CLOSED"
                state.failures = 0
                state.opened_at = None
                return result

            return wrapper

        return decorator

    @staticmethod
    def with_fallback(
        fallback_func: Callable[..., T],
        cache_duration: Optional[float] = None,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Graceful degradation using cache first, then fallback function."""
        if cache_duration is not None and cache_duration < 0:
            raise ValueError("cache_duration must be >= 0 or None")

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            cache: dict[str, object] = {
                "result": None,
                "cached_at": None,
                "has_value": False,
            }

            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                try:
                    result = func(*args, **kwargs)
                    cache["result"] = result
                    cache["cached_at"] = time.time()
                    cache["has_value"] = True
                    return result
                except Exception as exc:
                    print(f"[SelfHealing] Primary failed: {exc}")

                    if bool(cache["has_value"]):
                        if cache_duration is None:
                            print("[SelfHealing] Using cached result (no expiry)")
                            return cache["result"]  # type: ignore[return-value]
                        cached_at = cache["cached_at"]
                        if isinstance(cached_at, (int, float)):
                            age = time.time() - cached_at
                            if age < cache_duration:
                                print(
                                    "[SelfHealing] Using cached result "
                                    f"(age: {age:.1f}s)"
                                )
                                return cache["result"]  # type: ignore[return-value]

                    print("[SelfHealing] Using fallback function")
                    return fallback_func(*args, **kwargs)

            return wrapper

        return decorator

