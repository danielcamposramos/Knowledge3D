"""Dynamic Math Core pool with GPU-aware capacity limits and reuse semantics."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MathCore:
    """Lightweight record for a Math Core allocation."""

    instance_id: int
    tier: int
    stack_depth: int = 69
    gpu_id: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def reset(self, *, tier: Optional[int] = None) -> None:
        if tier is not None:
            self.tier = tier
        self.last_used = time.time()

    def cleanup(self) -> None:
        # Placeholder for future GPU teardown hooks.
        self.last_used = time.time()


class MathCorePool:
    """Dynamic Math Core instantiation and lifecycle management."""

    FALLBACK_MAX_CORES = 18  # Legacy static limit (Tesla 3-6-9 heritage)

    def __init__(self, gpu_id: int = 0, idle_timeout: float = 60.0) -> None:
        self.gpu_id = gpu_id
        self.idle_timeout = float(idle_timeout)
        self.active_cores: Dict[int, MathCore] = {}
        self.idle_pool: List[MathCore] = []
        self._lock = threading.Lock()
        self.max_cores = self._query_gpu_capacity()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def spawn_core(self, tier: int, reuse: bool = True) -> int:
        """Spawn or reuse a Math Core."""
        with self._lock:
            self._cleanup_idle_cores_locked()

            if reuse and self.idle_pool:
                core = self.idle_pool.pop()
                core.reset(tier=tier)
                self.active_cores[core.instance_id] = core
                return core.instance_id

            if len(self.active_cores) >= self.max_cores:
                raise RuntimeError(
                    f"GPU {self.gpu_id} at capacity "
                    f"({len(self.active_cores)}/{self.max_cores} cores)"
                )

            instance_id = self._allocate_id_locked()
            core = MathCore(
                instance_id=instance_id,
                tier=tier,
                gpu_id=self.gpu_id,
            )
            self.active_cores[instance_id] = core
            return instance_id

    def release_core(self, instance_id: int, pool: bool = True) -> None:
        """Release a Math Core back to pool or deallocate."""
        with self._lock:
            core = self.active_cores.pop(instance_id, None)
            if core is None:
                # If already in idle pool, drop it.
                self.idle_pool = [c for c in self.idle_pool if c.instance_id != instance_id]
                return

            core.last_used = time.time()
            pool_cap = max(1, int(self.max_cores * 0.1))
            if pool and len(self.idle_pool) < pool_cap:
                core.reset()
                self.idle_pool.append(core)
            else:
                core.cleanup()

    def retier_core(self, instance_id: int, tier: int) -> None:
        """Update an active or pooled core to a new tier."""
        with self._lock:
            core = self.active_cores.get(instance_id)
            if core is not None:
                core.reset(tier=tier)
                return
            for pooled in self.idle_pool:
                if pooled.instance_id == instance_id:
                    pooled.reset(tier=tier)
                    return

    def touch(self, instance_id: int) -> None:
        """Update last_used timestamp to keep core warm."""
        with self._lock:
            core = self.active_cores.get(instance_id)
            if core:
                core.last_used = time.time()

    def describe_tier(self, tier: int) -> str:
        """Return the canonical math-core role name for a tier."""
        return {
            1: "worker_worker",
            2: "worker",
            3: "master",
        }.get(int(tier), "unknown")

    def snapshot(self) -> dict:
        """Expose a lightweight runtime snapshot for orchestration/telemetry."""
        with self._lock:
            self._cleanup_idle_cores_locked()
            active_tiers = {1: 0, 2: 0, 3: 0}
            idle_tiers = {1: 0, 2: 0, 3: 0}
            for core in self.active_cores.values():
                active_tiers[int(core.tier)] = active_tiers.get(int(core.tier), 0) + 1
            for core in self.idle_pool:
                idle_tiers[int(core.tier)] = idle_tiers.get(int(core.tier), 0) + 1
            return {
                "gpu_id": int(self.gpu_id),
                "max_cores": int(self.max_cores),
                "active": len(self.active_cores),
                "idle": len(self.idle_pool),
                "idle_timeout": float(self.idle_timeout),
                "active_tiers": active_tiers,
                "idle_tiers": idle_tiers,
                "spawn_policy": "adaptive_reuse",
                "pool_role": "dynamic_math_core_pool",
            }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _allocate_id_locked(self) -> int:
        used_ids = set(self.active_cores.keys()) | {c.instance_id for c in self.idle_pool}
        for candidate in range(self.max_cores):
            if candidate not in used_ids:
                return candidate
        raise RuntimeError("No available instance IDs")

    def _cleanup_idle_cores_locked(self) -> None:
        if self.idle_timeout <= 0:
            return
        cutoff = time.time() - self.idle_timeout
        keep: List[MathCore] = []
        for core in self.idle_pool:
            if core.last_used >= cutoff:
                keep.append(core)
            else:
                core.cleanup()
        self.idle_pool = keep

    def _query_gpu_capacity(self) -> int:
        """Query GPU SM count + VRAM to determine max concurrent cores.

        Uses sovereign ctypes approach (libcuda.so directly) to avoid cupy dependency.
        """
        try:
            import ctypes

            # Load CUDA Driver API
            try:
                nvcuda = ctypes.CDLL("libcuda.so.1")
            except OSError:
                nvcuda = ctypes.CDLL("libcuda.so")

            # Initialize CUDA
            nvcuda.cuInit(0)

            # Get device handle
            device = ctypes.c_int()
            nvcuda.cuDeviceGet(ctypes.byref(device), self.gpu_id)

            # Query SM count (multiProcessorCount = attribute 16)
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
            sm_count = ctypes.c_int()
            nvcuda.cuDeviceGetAttribute(
                ctypes.byref(sm_count),
                CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                device
            )
            hardware_limit = max(self.FALLBACK_MAX_CORES, sm_count.value * 10)

            # Query VRAM (cuMemGetInfo_v2 or cuMemGetInfo)
            try:
                cuMemGetInfo = getattr(nvcuda, "cuMemGetInfo_v2")
            except AttributeError:
                cuMemGetInfo = getattr(nvcuda, "cuMemGetInfo")

            free_bytes = ctypes.c_size_t()
            total_bytes = ctypes.c_size_t()
            cuMemGetInfo(ctypes.byref(free_bytes), ctypes.byref(total_bytes))

            free_mb = free_bytes.value / (1024 ** 2)
            # 1% of free VRAM, 2 KB per core
            vram_limit = int((free_mb * 0.01) / 0.002)

            capacity = min(hardware_limit, vram_limit)
            return max(self.FALLBACK_MAX_CORES, capacity)
        except Exception:
            # Fall back to static limit if GPU query fails
            return self.FALLBACK_MAX_CORES


_GLOBAL_POOL: MathCorePool | None = None


def get_global_math_core_pool() -> MathCorePool:
    """Lazy-create a shared global pool for callers that do not pass one."""
    global _GLOBAL_POOL
    if _GLOBAL_POOL is None:
        _GLOBAL_POOL = MathCorePool()
    return _GLOBAL_POOL


__all__ = ["MathCore", "MathCorePool", "get_global_math_core_pool"]
