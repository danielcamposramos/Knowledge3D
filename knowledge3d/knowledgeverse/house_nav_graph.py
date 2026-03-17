"""Build-time House navigation graph from room and door stars."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .house_doors import HOUSE_DOORS
from .house_rooms import HOUSE_ROOMS


@dataclass
class HouseNavNode:
    """A navigable location in the House."""

    star_id: str
    house_room: str
    position: tuple[float, float, float]
    connected_to: list[str] = field(default_factory=list)


@dataclass
class HouseNavEdge:
    """A traversable connection between two nav nodes."""

    door_star_id: str
    from_node: str
    to_node: str
    cost: float = 1.0


@dataclass
class HouseNavGraph:
    """Complete House navigation graph."""

    nodes: dict[str, HouseNavNode]
    edges: list[HouseNavEdge]

    def neighbors(self, star_id: str) -> list[str]:
        node = self.nodes.get(star_id)
        return list(node.connected_to) if node is not None else []

    def shortest_path(self, from_id: str, to_id: str) -> list[str]:
        if from_id not in self.nodes or to_id not in self.nodes:
            return []
        if from_id == to_id:
            return [from_id]
        frontier: deque[str] = deque([from_id])
        came_from: dict[str, str | None] = {from_id: None}
        while frontier:
            current = frontier.popleft()
            if current == to_id:
                break
            for neighbor in self.neighbors(current):
                if neighbor in came_from:
                    continue
                came_from[neighbor] = current
                frontier.append(neighbor)
        if to_id not in came_from:
            return []
        path: list[str] = []
        cursor: str | None = to_id
        while cursor is not None:
            path.append(cursor)
            cursor = came_from[cursor]
        path.reverse()
        return path

    def to_metadata(self) -> dict[str, object]:
        return {
            "nodes": list(self.nodes.keys()),
            "edges": [
                {
                    "door": edge.door_star_id,
                    "from": edge.from_node,
                    "to": edge.to_node,
                    "cost": float(edge.cost),
                }
                for edge in self.edges
            ],
        }


def _door_room_refs(door_behavior_rpn: str, fallback_refs: list[str]) -> tuple[str, str] | None:
    tokens = door_behavior_rpn.split()
    if len(tokens) >= 4 and tokens[0] == "DOOR_TRAVERSE" and tokens[1] == "CONNECT":
        return tokens[2], tokens[3]
    if len(fallback_refs) >= 2:
        return fallback_refs[0], fallback_refs[1]
    return None


def build_house_nav_graph() -> HouseNavGraph:
    """Construct navigation graph from HOUSE_ROOMS and HOUSE_DOORS."""
    room_by_house_room = {room.house_room: room for room in HOUSE_ROOMS}
    nodes = {
        room.star_id: HouseNavNode(
            star_id=room.star_id,
            house_room=room.house_room,
            position=tuple(float(value) for value in room.house_position),
        )
        for room in HOUSE_ROOMS
    }
    edges: list[HouseNavEdge] = []
    for door in HOUSE_DOORS:
        room_refs = _door_room_refs(door.behavior_rpn or "", list(door.taxonomy_refs))
        if room_refs is None:
            continue
        left_room = room_by_house_room.get(room_refs[0])
        right_room = room_by_house_room.get(room_refs[1])
        if left_room is None or right_room is None:
            continue
        edges.append(HouseNavEdge(door.star_id, left_room.star_id, right_room.star_id))
        edges.append(HouseNavEdge(door.star_id, right_room.star_id, left_room.star_id))
        if right_room.star_id not in nodes[left_room.star_id].connected_to:
            nodes[left_room.star_id].connected_to.append(right_room.star_id)
        if left_room.star_id not in nodes[right_room.star_id].connected_to:
            nodes[right_room.star_id].connected_to.append(left_room.star_id)
    for node in nodes.values():
        node.connected_to.sort()
    return HouseNavGraph(nodes=nodes, edges=edges)


__all__ = [
    "HouseNavEdge",
    "HouseNavGraph",
    "HouseNavNode",
    "build_house_nav_graph",
]
