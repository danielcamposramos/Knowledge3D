from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .address import SpatialAddress


@dataclass(frozen=True)
class Door:
    label: str
    address: SpatialAddress


class Physical3D:
    """Physical layer for 3D positions and embedding buffers (conceptual)."""

    @staticmethod
    def address_of(xyz: Iterable[float], label: Optional[str] = None, cell: float = 1.0) -> SpatialAddress:
        v = tuple(float(x) for x in xyz)  # type: ignore[assignment]
        return SpatialAddress.from_vector(v, port=0, label=label, cell_size=cell)


class DataLink3D:
    """Data-link layer: immediate neighbors and proximity checks."""

    @staticmethod
    def neighbors_of(node_id: str, neighbor_map: Dict[str, List[str]]) -> List[str]:
        return list(neighbor_map.get(node_id, []))


class Network3D:
    """Network layer: routing across knowledge regions using neighbor graph."""

    @staticmethod
    def route(ids: List[str], neighbors: List[List[str]], start_id: str, target_id: str) -> Optional[List[str]]:
        id_set = set(ids)
        if start_id not in id_set or target_id not in id_set:
            return None
        # Build map for quick lookup
        neigh_map: Dict[str, List[str]] = {ids[i]: list(neighbors[i]) for i in range(len(ids))}
        if start_id == target_id:
            return [start_id]
        from collections import deque
        q = deque([[start_id]])
        visited = {start_id}
        while q:
            path = q.popleft()
            last = path[-1]
            for n in neigh_map.get(last, []):
                if n in visited:
                    continue
                visited.add(n)
                nxt = path + [n]
                if n == target_id:
                    return nxt
                q.append(nxt)
        return None


class Transport3D:
    """Transport layer: message envelope definition (lightweight)."""

    @dataclass
    class Message:
        src: SpatialAddress
        dst: SpatialAddress
        seq: int
        payload: bytes


class Session3D:
    """Session layer: minimal per-channel state (current label, doors)."""
    _labels: Dict[str, str] = {}
    _doors: Dict[str, Dict[str, str]] = {}

    @classmethod
    def set_label(cls, channel: str, label: str) -> None:
        cls._labels[channel] = label

    @classmethod
    def get_label(cls, channel: str) -> Optional[str]:
        return cls._labels.get(channel)

    @classmethod
    def set_doors(cls, channel: str, doors: Dict[str, str]) -> None:
        cls._doors[channel] = dict(doors)

    @classmethod
    def get_doors(cls, channel: str) -> Dict[str, str]:
        return dict(cls._doors.get(channel, {}))


class Presentation3D:
    """Presentation layer: simple helpers to format payloads."""

    @staticmethod
    def format_open_payload(label: str, address: str, path: List[str]) -> dict:
        return {"label": label, "address": address, "path": path}


class Application3D:
    """Application layer: helpers for door registration and routing."""

    @staticmethod
    def register_doors(channel: str, doors: Dict[str, str]) -> None:
        Session3D.set_doors(channel, doors)

    @staticmethod
    def current_label(channel: str) -> Optional[str]:
        return Session3D.get_label(channel)

