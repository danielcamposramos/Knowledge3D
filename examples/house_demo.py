"""
Demonstrates the full Condo -> House -> Cranium workflow.

1. Loads the condo configuration.
2. Populates two houses ('math' and 'physics') with sample data.
3. Uses the Cranium to project the embeddings to 3D for each house.
4. Retrieves neighbors for a sample vector to test the API.
"""
import numpy as np
from pathlib import Path

from k3dgen.condo import Condo
from k3dgen.cranium import Cranium

# Define the path to the condo configuration
CONDO_CONFIG_PATH = Path(__file__).parent / "condo.json"

def populate_house(house, expert_name):
    """Adds some sample data to a given house."""
    print(f"Populating '{expert_name}' house...")
    if expert_name == "math":
        # Simple vectors representing points on a line
        house.store_embedding("one", [1, 0, 0, 0, 0], {"label": "one"})
        house.store_embedding("two", [2, 0, 0, 0, 0], {"label": "two"})
        house.store_embedding("three", [3, 0, 0, 0, 0], {"label": "three"})
        house.store_embedding("four", [4, 0, 0, 0, 0], {"label": "four"})
    elif expert_name == "physics":
        # Simple vectors representing points in a plane
        house.store_embedding("force", [1, 1, 0, 0, 0], {"label": "force"})
        house.store_embedding("mass", [1, 2, 0, 0, 0], {"label": "mass"})
        house.store_embedding("acceleration", [2, 1, 0, 0, 0], {"label": "acceleration"})
        house.store_embedding("gravity", [2, 2, 0, 0, 0], {"label": "gravity"})
    print(f"'{expert_name}' house populated.")

def main():
    """Main execution function."""
    print("Initializing Condo...")
    condo = Condo(CONDO_CONFIG_PATH)

    experts = condo.list_experts()
    print(f"Found experts: {experts}")

    # --- Populate and Process Houses ---
    for expert in ["math", "physics"]:
        house = condo.get_house(expert)
        populate_house(house, expert)

        print(f"Initializing Cranium for '{expert}' house...")
        cranium = Cranium(house)

        print("Projecting embeddings to 3D and updating house vectors...")
        cranium.update_house_vectors()
        print(f"'{expert}' house vectors updated.")

    # --- Verify Neighbor Retrieval ---
    print("\nVerifying neighbor retrieval...")
    math_house = condo.get_house("math")
    try:
        neighbors = math_house.retrieve_neighbors("one", k=2)
        print("Neighbors of 'one' in math house:", neighbors)
        # Expected output might be ['two', 'three'] or similar, depending on projection
    except Exception as e:
        print(f"Error retrieving neighbors: {e}")

    print("\nDemo complete. You can now run the viewer to see the results.")

if __name__ == "__main__":
    main()
