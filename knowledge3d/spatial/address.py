from __future__ import annotations

from dataclasses import dataclass
from math import floor, sqrt
from typing import Dict, Optional, Tuple


Cell = Tuple[int, int, int]


@dataclass(frozen=True)
class SpatialAddress:
    """
    Spatial address for knowledge nodes, inspired by OSI addressing.

    - region: integer grid cell (coarse) for routing domains
    - port: virtual port inside the region (service endpoint)
    - xyz: precise coordinates (physical layer)
    - label: optional human/AI label

    URI form: k3d://rx,ry,rz:port@x,y,z[?label=...]
    """

    region: Cell
    port: int
    xyz: Tuple[float, float, float]
    label: Optional[str] = None

    @staticmethod
    def partition(xyz: Tuple[float, float, float], cell_size: float = 1.0) -> Cell:
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        x, y, z = xyz
        return (floor(x / cell_size), floor(y / cell_size), floor(z / cell_size))

    @classmethod
    def from_vector(
        cls,
        xyz: Tuple[float, float, float],
        *,
        port: int = 0,
        label: Optional[str] = None,
        cell_size: float = 1.0,
    ) -> "SpatialAddress":
        return cls(region=cls.partition(xyz, cell_size), port=port, xyz=xyz, label=label)

    def distance_to(self, other: "SpatialAddress") -> float:
        ax, ay, az = self.xyz
        bx, by, bz = other.xyz
        return sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)

    def to_uri(self) -> str:
        rx, ry, rz = self.region
        x, y, z = self.xyz
        s = f"k3d://{rx},{ry},{rz}:{self.port}@{x:.6f},{y:.6f},{z:.6f}"
        if self.label:
            s += f"?label={self.label}"
        return s

    @classmethod
    def from_uri(cls, uri: str) -> "SpatialAddress":
        # very small parser for the form defined above
        if not uri.startswith("k3d://"):
            raise ValueError("Invalid scheme for SpatialAddress")
        rest = uri[len("k3d://") :]
        path, *q = rest.split("?")
        region_part, at_part = path.split("@", 1)
        region_str, port_str = region_part.split(":", 1)
        rx, ry, rz = (int(t) for t in region_str.split(","))
        x_str, y_str, z_str = at_part.split(",", 2)
        x, y, z = float(x_str), float(y_str), float(z_str)
        label: Optional[str] = None
        if q:
            for kv in q[0].split("&"):
                if kv.startswith("label="):
                    label = kv[len("label=") :]
                    break
        return cls(region=(rx, ry, rz), port=int(port_str), xyz=(x, y, z), label=label)

    def to_dict(self) -> Dict[str, object]:
        return {"region": list(self.region), "port": self.port, "xyz": list(self.xyz), "label": self.label}

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "SpatialAddress":
        region = tuple(int(v) for v in (d.get("region") or [0, 0, 0]))  # type: ignore[arg-type]
        port = int(d.get("port") or 0)
        xyz_t = tuple(float(v) for v in (d.get("xyz") or [0.0, 0.0, 0.0]))  # type: ignore[arg-type]
        label = d.get("label")
        return cls(region=region, port=port, xyz=xyz_t, label=str(label) if label is not None else None)

