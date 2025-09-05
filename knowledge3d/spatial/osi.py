from __future__ import annotations

from dataclasses import dataclass
import os
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
        """Default routing (BFS). Honors K3D_ROUTER=astar|dijkstra|bfs (default)."""
        router = (os.getenv("K3D_ROUTER", "bfs") or "bfs").lower().strip()
        if router == "dijkstra":
            return Network3D.route_dijkstra(ids, neighbors, start_id, target_id)
        # A* requires positions; use BFS unless positions are provided via route_astar_ex
        return Network3D.route_bfs(ids, neighbors, start_id, target_id)

    @staticmethod
    def route_bfs(ids: List[str], neighbors: List[List[str]], start_id: str, target_id: str) -> Optional[List[str]]:
        id_set = set(ids)
        if start_id not in id_set or target_id not in id_set:
            return None
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

    @staticmethod
    def route_dijkstra(ids: List[str], neighbors: List[List[str]], start_id: str, target_id: str) -> Optional[List[str]]:
        """Dijkstra on unweighted graph (edge cost=1) — identical to BFS in path length but uses classic API.

        Provided to enable easy extension to weighted edges later (e.g., 1-cosine or geometric length).
        """
        id_to_idx = {ids[i]: i for i in range(len(ids))}
        si = id_to_idx.get(start_id)
        ti = id_to_idx.get(target_id)
        if si is None or ti is None:
            return None
        import heapq
        dist = {si: 0.0}
        prev: Dict[int, int | None] = {si: None}
        pq = [(0.0, si)]
        visited = set()
        while pq:
            d, i = heapq.heappop(pq)
            if i in visited:
                continue
            visited.add(i)
            if i == ti:
                # reconstruct
                path_idx = []
                cur: Optional[int] = i
                while cur is not None:
                    path_idx.append(cur)
                    cur = prev.get(cur)
                path_idx.reverse()
                return [ids[k] for k in path_idx]
            for nid in neighbors[i]:
                j = id_to_idx.get(nid)
                if j is None or j in visited:
                    continue
                alt = d + 1.0
                if alt < dist.get(j, 1e18):
                    dist[j] = alt
                    prev[j] = i
                    heapq.heappush(pq, (alt, j))
        return None

    @staticmethod
    def route_astar_ex(
        ids: List[str],
        neighbors: List[List[str]],
        positions: List[tuple[float, float, float]] | None,
        start_id: str,
        target_id: str,
    ) -> Optional[List[str]]:
        """A* pathfinding using Euclidean 3D heuristic if positions are provided.

        Falls back to BFS if positions is None.
        """
        if positions is None:
            return Network3D.route_bfs(ids, neighbors, start_id, target_id)
        import heapq
        id_to_idx = {ids[i]: i for i in range(len(ids))}
        si = id_to_idx.get(start_id)
        ti = id_to_idx.get(target_id)
        if si is None or ti is None:
            return None
        def dist3(a: int, b: int) -> float:
            pa = positions[a]; pb = positions[b]
            return ((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2 + (pa[2]-pb[2])**2) ** 0.5
        # min observed edge length approximation
        min_edge = float('inf')
        for i in range(len(ids)):
            for nb in neighbors[i]:
                j = id_to_idx.get(nb)
                if j is None: continue
                d = dist3(i, j)
                if 0 < d < min_edge:
                    min_edge = d
        if not (min_edge > 0 and min_edge < float('inf')):
            min_edge = 1.0
        def h(i: int, j: int) -> float:
            return dist3(i, j) / min_edge
        openq = []
        heapq.heappush(openq, (h(si, ti), 0.0, si, None))
        came: Dict[int, Optional[int]] = {}
        gscore: Dict[int, float] = {si: 0.0}
        visited = set()
        while openq:
            f, g, i, parent = heapq.heappop(openq)
            if i in visited:
                continue
            visited.add(i)
            came[i] = parent
            if i == ti:
                path_idx = []
                cur = i
                while cur is not None:
                    path_idx.append(cur)
                    cur = came.get(cur)
                path_idx.reverse()
                return [ids[k] for k in path_idx]
            for nb in neighbors[i]:
                j = id_to_idx.get(nb)
                if j is None or j in visited:
                    continue
                ng = g + 1.0
                if ng < gscore.get(j, 1e18):
                    gscore[j] = ng
                    heapq.heappush(openq, (ng + h(j, ti), ng, j, i))
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
