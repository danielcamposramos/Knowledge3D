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
    """Transport layer: reliable delivery (placeholder)."""

    @dataclass
    class Message:
        src: SpatialAddress
        dst: SpatialAddress
        seq: int
        payload: bytes


class Session3D:
    """Session layer: maintains conversational/session state (placeholder)."""
    pass


class Presentation3D:
    """Presentation layer: transforms data for human/AI clients (placeholder)."""
    pass


class Application3D:
    """Application layer: high-level services like search, edit, insight (placeholder)."""
    pass

