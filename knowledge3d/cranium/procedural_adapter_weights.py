"""Procedural specialist adapter weights encoded as small RPN-style programs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from knowledge3d.cranium.ptx_runtime.rpn_math_core import DeviceTensor, HostTensorF32, RPNMathCore
from knowledge3d.cranium.ptx_runtime import rpn_opcodes as ropc


@dataclass(frozen=True)
class ProceduralOp:
    opcode: int
    name: str
    params: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "opcode": int(self.opcode),
            "name": str(self.name),
            "params": dict(self.params),
        }

    @staticmethod
    def from_payload(payload: dict[str, Any]) -> "ProceduralOp":
        return ProceduralOp(
            opcode=int(payload.get("opcode", 0)),
            name=str(payload.get("name") or ""),
            params=dict(payload.get("params") or {}),
        )


class ProceduralAdapterWeights:
    """Low-rank adapter deltas as a compact procedural program over base weights."""

    STORAGE_VERSION = 1
    STORAGE_KIND = "procedural_adapter_v1"

    def __init__(
        self,
        *,
        base_weights_ref: str,
        shape: tuple[int, int],
        rank: int,
        alpha: float,
        primary_a: HostTensorF32,
        primary_b: HostTensorF32,
        shadow_a: HostTensorF32 | None = None,
        shadow_b: HostTensorF32 | None = None,
        ternary_mask: Sequence[int] | None = None,
    ) -> None:
        self.base_weights_ref = str(base_weights_ref)
        self.shape = (int(shape[0]), int(shape[1]))
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.primary_a = HostTensorF32.from_array_like(primary_a)
        self.primary_b = HostTensorF32.from_array_like(primary_b)
        self.shadow_a = HostTensorF32.from_array_like(shadow_a or primary_a)
        self.shadow_b = HostTensorF32.from_array_like(shadow_b or primary_b)
        self.ternary_mask = [int(value) for value in list(ternary_mask or [1] * self.shape[0])]
        if len(self.ternary_mask) < self.shape[0]:
            self.ternary_mask.extend([1] * (self.shape[0] - len(self.ternary_mask)))

    @classmethod
    def from_legacy_factors(
        cls,
        *,
        base_weights_ref: str,
        shape: tuple[int, int],
        rank: int,
        alpha: float,
        primary_a: Any,
        primary_b: Any,
        shadow_a: Any | None = None,
        shadow_b: Any | None = None,
        ternary_mask: Sequence[int] | None = None,
    ) -> "ProceduralAdapterWeights":
        return cls(
            base_weights_ref=base_weights_ref,
            shape=shape,
            rank=rank,
            alpha=alpha,
            primary_a=HostTensorF32.from_array_like(primary_a),
            primary_b=HostTensorF32.from_array_like(primary_b),
            shadow_a=HostTensorF32.from_array_like(shadow_a or primary_a),
            shadow_b=HostTensorF32.from_array_like(shadow_b or primary_b),
            ternary_mask=ternary_mask,
        )

    @classmethod
    def from_legacy_adapter(
        cls,
        adapter: Any,
        *,
        base_weights_ref: str | None = None,
        ternary_mask: Sequence[int] | None = None,
    ) -> "ProceduralAdapterWeights":
        specialist_name = str(getattr(adapter, "specialist_name", "specialist")).strip() or "specialist"
        return cls.from_legacy_factors(
            base_weights_ref=base_weights_ref or f"{specialist_name}:base",
            shape=tuple(int(value) for value in tuple(getattr(adapter, "shape", (0, 0)))),
            rank=int(getattr(adapter, "rank", 0)),
            alpha=float(getattr(adapter, "alpha", 1.0)),
            primary_a=getattr(adapter, "A"),
            primary_b=getattr(adapter, "B"),
            shadow_a=getattr(adapter, "A_shadow", None),
            shadow_b=getattr(adapter, "B_shadow", None),
            ternary_mask=ternary_mask,
        )

    def primary_program(self) -> list[ProceduralOp]:
        return [
            ProceduralOp(ropc.OP_LORA_LOAD_BASE, "LORA_LOAD_BASE", {"base_weights_ref": self.base_weights_ref}),
            ProceduralOp(ropc.OP_LORA_LOW_RANK_ADD, "LORA_LOW_RANK_ADD", {"rank": int(self.rank), "target": "primary"}),
            ProceduralOp(ropc.OP_LORA_SCALE, "LORA_SCALE", {"alpha": float(self.alpha)}),
            ProceduralOp(ropc.OP_LORA_TERNARY_MASK, "LORA_TERNARY_MASK", {"mask_len": len(self.ternary_mask)}),
        ]

    def shadow_program(self) -> list[ProceduralOp]:
        return self.primary_program() + [
            ProceduralOp(ropc.OP_LORA_SHADOW_ABSORB, "LORA_SHADOW_ABSORB", {"target": "shadow"}),
        ]

    def to_payload(self) -> dict[str, Any]:
        return {
            "storage_kind": self.STORAGE_KIND,
            "version": self.STORAGE_VERSION,
            "base_weights_ref": self.base_weights_ref,
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "rank": int(self.rank),
            "alpha": float(self.alpha),
            "ternary_mask": list(self.ternary_mask),
            "primary_program": [op.to_payload() for op in self.primary_program()],
            "shadow_program": [op.to_payload() for op in self.shadow_program()],
            "primary_a_shape": [int(self.primary_a.rows), int(self.primary_a.cols)],
            "primary_b_shape": [int(self.primary_b.rows), int(self.primary_b.cols)],
            "shadow_a_shape": [int(self.shadow_a.rows), int(self.shadow_a.cols)],
            "shadow_b_shape": [int(self.shadow_b.rows), int(self.shadow_b.cols)],
        }

    @staticmethod
    def archive_entries() -> dict[str, str]:
        return {
            "payload": "procedural_adapter.json",
            "primary_a": "procedural_primary_A.bin",
            "primary_b": "procedural_primary_B.bin",
            "shadow_a": "procedural_shadow_A.bin",
            "shadow_b": "procedural_shadow_B.bin",
        }

    def save_archive(self, archive: Any) -> None:
        names = self.archive_entries()
        archive.writestr(names["payload"], json.dumps(self.to_payload(), separators=(",", ":"), sort_keys=True))
        archive.writestr(names["primary_a"], self.primary_a.to_bytes())
        archive.writestr(names["primary_b"], self.primary_b.to_bytes())
        archive.writestr(names["shadow_a"], self.shadow_a.to_bytes())
        archive.writestr(names["shadow_b"], self.shadow_b.to_bytes())

    @classmethod
    def from_archive(cls, archive: Any) -> "ProceduralAdapterWeights":
        names = cls.archive_entries()
        payload = json.loads(archive.read(names["payload"]).decode("utf-8"))
        primary_a = HostTensorF32.zeros(int(payload["primary_a_shape"][0]), int(payload["primary_a_shape"][1]))
        primary_b = HostTensorF32.zeros(int(payload["primary_b_shape"][0]), int(payload["primary_b_shape"][1]))
        shadow_a = HostTensorF32.zeros(int(payload["shadow_a_shape"][0]), int(payload["shadow_a_shape"][1]))
        shadow_b = HostTensorF32.zeros(int(payload["shadow_b_shape"][0]), int(payload["shadow_b_shape"][1]))
        primary_a.load_bytes(archive.read(names["primary_a"]))
        primary_b.load_bytes(archive.read(names["primary_b"]))
        shadow_a.load_bytes(archive.read(names["shadow_a"]))
        shadow_b.load_bytes(archive.read(names["shadow_b"]))
        return cls(
            base_weights_ref=str(payload.get("base_weights_ref") or ""),
            shape=(int(payload["shape"][0]), int(payload["shape"][1])),
            rank=int(payload["rank"]),
            alpha=float(payload["alpha"]),
            primary_a=primary_a,
            primary_b=primary_b,
            shadow_a=shadow_a,
            shadow_b=shadow_b,
            ternary_mask=list(payload.get("ternary_mask") or []),
        )

    def sync_from_legacy_adapter(self, adapter: Any) -> None:
        self.primary_a.copy_from(getattr(adapter, "A"))
        self.primary_b.copy_from(getattr(adapter, "B"))
        if hasattr(adapter, "A_shadow") and hasattr(adapter, "B_shadow"):
            self.shadow_a.copy_from(getattr(adapter, "A_shadow"))
            self.shadow_b.copy_from(getattr(adapter, "B_shadow"))

    def fork_shadow_from_primary(self) -> None:
        self.shadow_a.copy_from(self.primary_a)
        self.shadow_b.copy_from(self.primary_b)

    def _apply_ternary_mask(self, tensor: HostTensorF32) -> HostTensorF32:
        masked = HostTensorF32.from_array_like(tensor)
        for row in range(masked.rows):
            gate = int(self.ternary_mask[row] if row < len(self.ternary_mask) else 1)
            if gate == 0:
                for col in range(masked.cols):
                    masked._buffer[row * masked.cols + col] = 0.0
            elif gate < 0:
                for col in range(masked.cols):
                    index = row * masked.cols + col
                    masked._buffer[index] = -float(masked._buffer[index])
        masked._notify_mutation()
        return masked

    def _materialize(self, left: HostTensorF32, right: HostTensorF32) -> HostTensorF32:
        math_core = RPNMathCore()
        delta = math_core.matmul_host(left, right)
        if self.alpha != 1.0:
            delta.scale_inplace(self.alpha)
        return self._apply_ternary_mask(delta)

    def materialize_host(self, *, shadow: bool = False) -> HostTensorF32:
        return self._materialize(self.shadow_a, self.shadow_b) if shadow else self._materialize(self.primary_a, self.primary_b)

    def materialize_to_vram(self, *, shadow: bool = False) -> DeviceTensor:
        host = self.materialize_host(shadow=shadow)
        math_core = RPNMathCore()
        ptr = math_core.to_device(host)
        return DeviceTensor(ptr, host.rows, host.cols)

    def absorb_contrast(self, gradient: Any, *, lr: float = 0.001, shadow: bool = True) -> float:
        gradient_host = HostTensorF32.from_array_like(gradient, rows=self.shape[0], cols=self.shape[1])
        left = self.shadow_a if shadow else self.primary_a
        right = self.shadow_b if shadow else self.primary_b

        grad_norm_sq = 0.0
        for value in gradient_host.to_flat_list():
            grad_norm_sq += float(value) * float(value)

        out_dim, in_dim = self.shape
        rank = self.rank
        grad_a = HostTensorF32.zeros(out_dim, rank)
        grad_b = HostTensorF32.zeros(rank, in_dim)
        right_t = right.transpose()
        left_t = left.transpose()

        left_norm = sum(float(value) * float(value) for value in left.to_flat_list()) ** 0.5
        right_norm = sum(float(value) * float(value) for value in right.to_flat_list()) ** 0.5
        if left_norm <= 1e-12 and right_norm <= 1e-12:
            # Cold-start bootstrap: seed a deterministic rank-limited factorization
            # directly from the observed gradient so shadow weights can begin moving.
            for col in range(rank):
                pivot = col % max(1, in_dim)
                for row in range(out_dim):
                    grad_a._buffer[row * rank + col] = float(gradient_host[row, pivot])
                for inner in range(in_dim):
                    grad_b._buffer[col * in_dim + inner] = 1.0 if inner == pivot else 0.0
        else:
            for row in range(out_dim):
                for col in range(rank):
                    total = 0.0
                    for inner in range(in_dim):
                        total += float(gradient_host[row, inner]) * float(right_t[inner, col])
                    grad_a._buffer[row * rank + col] = total

            for row in range(rank):
                for col in range(in_dim):
                    total = 0.0
                    for inner in range(out_dim):
                        total += float(left_t[row, inner]) * float(gradient_host[inner, col])
                    grad_b._buffer[row * in_dim + col] = total

        scale = -float(lr)
        for idx in range(left.size):
            left._buffer[idx] = float(left._buffer[idx]) + (float(grad_a._buffer[idx]) * scale)
        for idx in range(right.size):
            right._buffer[idx] = float(right._buffer[idx]) + (float(grad_b._buffer[idx]) * scale)
        left._notify_mutation()
        right._notify_mutation()
        return float(grad_norm_sq ** 0.5)
