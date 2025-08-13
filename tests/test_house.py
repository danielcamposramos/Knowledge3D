import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from k3dgen import house


def test_store_and_retrieve(tmp_path):
    house.DATA_PATH = tmp_path / "store.k3d"

    house.store_embedding("a", [1.0, 0.0, 0.0], {"label": "A"})
    house.store_embedding("b", [0.0, 1.0, 0.0], {"label": "B"})
    house.store_embedding("c", [0.0, 0.0, 1.0], {"label": "C"})

    data = json.loads(house.DATA_PATH.read_text())
    assert len(data) == 3

    neighbors = house.retrieve_neighbors("a", 2)
    assert set(neighbors) == {"b", "c"}
