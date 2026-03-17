from __future__ import annotations

from knowledge3d.knowledgeverse.house_nav_graph import build_house_nav_graph


def test_nav_graph_has_all_rooms() -> None:
    graph = build_house_nav_graph()
    assert len(graph.nodes) == 5
    assert "room_library" in graph.nodes
    assert "room_bathtub" in graph.nodes


def test_nav_graph_edges_from_doors() -> None:
    graph = build_house_nav_graph()
    assert len(graph.edges) == 8


def test_nav_graph_neighbors() -> None:
    graph = build_house_nav_graph()
    garden_neighbors = graph.neighbors("room_garden")
    assert "room_library" in garden_neighbors
    assert "room_workshop" in garden_neighbors


def test_nav_graph_shortest_path() -> None:
    graph = build_house_nav_graph()
    path = graph.shortest_path("room_library", "room_bathtub")
    assert path == ["room_library", "room_garden", "room_workshop", "room_gallery", "room_bathtub"]


def test_nav_graph_shortest_path_reverse() -> None:
    graph = build_house_nav_graph()
    path = graph.shortest_path("room_bathtub", "room_library")
    assert path == ["room_bathtub", "room_gallery", "room_workshop", "room_garden", "room_library"]
