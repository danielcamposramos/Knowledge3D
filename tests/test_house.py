import json
import sys
from pathlib import Path

# This is no longer best practice, but we'll keep it for now to avoid breaking other tests
sys.path.append(str(Path(__file__).resolve().parents[1]))
from k3dgen.house import House


def test_store_and_retrieve(tmp_path):
    """Verify that storing and retrieving embeddings works with the new House class."""
    # 1. Arrange: Create a House instance pointing to a temporary file
    house_path = tmp_path / "store.k3d"
    house_uri = house_path.as_uri()
    my_house = House(uri=house_uri)

    # 2. Act: Store some embeddings
    my_house.store_embedding("a", [1.0, 0.0, 0.0, 0.0], {"label": "A"})
    my_house.store_embedding("b", [0.0, 1.0, 0.0, 0.0], {"label": "B"})
    my_house.store_embedding("c", [0.0, 1.1, 0.0, 0.0], {"label": "C"}) # 'c' is closer to 'b'
    my_house.store_embedding("d", [1.1, 0.1, 0.0, 0.0], {"label": "D"}) # 'd' is closer to 'a'


    # 3. Assert: Check that the data was written correctly
    data = json.loads(house_path.read_text())
    assert len(data) == 4

    # 4. Act & Assert: Check neighbor retrieval
    # Neighbors of 'a' should be 'd' then 'b'
    neighbors_a = my_house.retrieve_neighbors("a", 2)
    assert neighbors_a == ["d", "b"]

    # Neighbors of 'b' should be 'c' then 'a'
    neighbors_b = my_house.retrieve_neighbors("b", 2)
    assert neighbors_b == ["c", "a"]
