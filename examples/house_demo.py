"""Demonstrate storing embeddings and retrieving neighbours."""
from k3dgen.house import retrieve_neighbors, store_embedding

store_embedding("a", [1.0, 0.0, 0.0], {"label": "A"})
store_embedding("b", [0.0, 1.0, 0.0], {"label": "B"})
store_embedding("c", [0.0, 0.0, 1.0], {"label": "C"})

print("Neighbours of 'a':", retrieve_neighbors("a", 2))
