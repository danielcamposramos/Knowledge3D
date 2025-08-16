import argparse
import base64
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from pygltflib import (
    GLTF2,
    Accessor,
    Asset,
    Buffer,
    BufferView,
    Mesh,
    Node,
    Primitive,
    Scene,
)
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# --- Constants ---
ARRAY_BUFFER = 34962
FLOAT = 5126
K3D_EXTENSION_NAME = "K3D_nodes"
K3D_IDS_PROPERTY = "extras.k3dIds"


def load_vectors_with_metadata(csv_path: str) -> Tuple[List[str], np.ndarray, List[Dict[str, Any]]]:
    """Load vectors and metadata from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file '{csv_path}': {exc}") from exc

    if "id" not in df.columns:
        raise ValueError("CSV must contain an 'id' column")

    ids: List[str] = df["id"].astype(str).tolist()

    metadata_cols = [col for col in df.columns if col not in ids and not col.startswith('v')]
    metadata = df[metadata_cols].to_dict(orient='records')

    vector_cols = [col for col in df.columns if col.startswith('v')]
    df_vectors = df[vector_cols]

    if df_vectors.empty:
        raise ValueError("CSV must contain at least one vector column (e.g., v1, v2)")

    try:
        for col in df_vectors.columns:
            df_vectors[col] = pd.to_numeric(df_vectors[col], errors="raise")
    except ValueError as exc:
        raise ValueError(f"Non-numeric data found in column '{col}': {exc}") from exc

    if df_vectors.isnull().any().any():
        raise ValueError("CSV contains missing values in vector columns")

    return ids, df_vectors.to_numpy(dtype=float), metadata


def reduce_dimensions(vectors: np.ndarray) -> np.ndarray:
    """Reduce the dimensionality of vectors to 3D using PCA."""
    try:
        pca = PCA(n_components=3)
        return pca.fit_transform(vectors)
    except Exception as exc:
        raise ValueError(f"PCA computation failed: {exc}") from exc


def find_neighbors(vectors: np.ndarray, k: int) -> np.ndarray:
    """Find the k-nearest neighbors for each vector."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if k >= len(vectors):
        raise ValueError("k must be less than the number of vectors")
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(vectors)
    _, indices = nn.kneighbors(vectors)
    return indices[:, 1:]  # Exclude the point itself


def create_k3d_file(
    k3d_path: str,
    ids: List[str],
    points: np.ndarray,
    embeddings: np.ndarray,
    neighbor_indices: np.ndarray,
    metadata: List[Dict[str, Any]],
) -> None:
    """Create the .k3d file."""
    records = []
    for i, (point_id, point, embedding) in enumerate(zip(ids, points, embeddings)):
        neighbor_ids = [ids[j] for j in neighbor_indices[i]]
        records.append(
            {
                "id": point_id,
                "vector": point.tolist(),
                "embedding": embedding.tolist(),
                "metadata": metadata[i],
                "neighbors": neighbor_ids,
            }
        )

    try:
        with open(k3d_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    except OSError as exc:
        raise OSError(f"Failed to write .k3d file '{k3d_path}': {exc}") from exc


def create_gltf_file(
    gltf_path: str, k3d_path: str, ids: List[str], points: np.ndarray
) -> None:
    """Create the .gltf file."""
    # 1. Convert numpy array to glTF binary buffer
    positions = points.astype(np.float32)
    data_bytes = positions.tobytes()
    uri = "data:application/octet-stream;base64," + base64.b64encode(
        data_bytes
    ).decode("ascii")

    # 2. Create glTF structure
    buffer = Buffer(byteLength=len(data_bytes), uri=uri)
    view = BufferView(
        buffer=0, byteOffset=0, byteLength=len(data_bytes), target=ARRAY_BUFFER
    )
    accessor = Accessor(
        bufferView=0,
        byteOffset=0,
        componentType=FLOAT,
        count=len(points),
        type="VEC3",
        max=positions.max(axis=0).tolist(),
        min=positions.min(axis=0).tolist(),
    )
    primitive = Primitive(
        attributes={"POSITION": 0}, mode=0, extras={"k3dIds": ids}
    )
    mesh = Mesh(primitives=[primitive])
    node = Node(mesh=0)
    scene = Scene(nodes=[0])

    # 3. Create K3D extension
    gltf_path = Path(gltf_path)
    k3d_path = Path(k3d_path)
    gltf_dir = gltf_path.resolve().parent

    # Correct the path to the schema
    try:
        spec_dir = Path(__file__).resolve().parent.parent / "spec"
        schema_path = spec_dir / "k3d_node_schema.json"
    except Exception:
        # Fallback for different execution contexts
        schema_path = Path("spec/k3d_node_schema.json")


    def _relative_path(target: Path, base: Path) -> str:
        try:
            return os.path.relpath(target, base)
        except ValueError:
            return str(target.resolve())

    relative_k3d_path = _relative_path(k3d_path, gltf_dir)
    relative_schema_path = _relative_path(schema_path, gltf_dir)

    gltf = GLTF2(
        asset=Asset(generator="my_house_generator"),
        buffers=[buffer],
        bufferViews=[view],
        accessors=[accessor],
        meshes=[mesh],
        nodes=[node],
        scenes=[scene],
        scene=0,
        extensionsUsed=[K3D_EXTENSION_NAME],
        extensions={
            K3D_EXTENSION_NAME: {
                "uri": relative_k3d_path,
                "schema": relative_schema_path,
                "primitiveIdsProperty": K3D_IDS_PROPERTY,
            }
        },
    )

    # 4. Save glTF file
    try:
        gltf.save(gltf_path)
    except OSError as exc:
        raise OSError(f"Failed to write glTF file '{gltf_path}': {exc}") from exc


def generate(csv_path: str, gltf_path: str, k3d_path: str, k: int) -> None:
    """Generate a glTF scene and .k3d metadata from a CSV."""
    # 1. Load the high-dimensional embeddings and metadata from the CSV file.
    ids, embeddings, metadata = load_vectors_with_metadata(csv_path)

    # 2. Reduce the dimensionality of the embeddings to 3D for visualization.
    points = reduce_dimensions(embeddings)

    # 3. Find the k-nearest neighbours for each point
    neighbor_indices = find_neighbors(embeddings, k)

    # 4. Create the .k3d file with the full embeddings and metadata.
    create_k3d_file(k3d_path, ids, points, embeddings, neighbor_indices, metadata)

    # 5. Create the .gltf file with the 3D positions and a link to the .k3d file.
    create_gltf_file(gltf_path, k3d_path, ids, points)


def main() -> None:
    """Command-line entry point for generating K3D assets."""
    parser = argparse.ArgumentParser(description="Generate glTF + .k3d for a house from a CSV.")
    parser.add_argument("csv", help="CSV file with id, metadata, and vector columns")
    parser.add_argument("--gltf", default="my_house.gltf", help="Output glTF path")
    parser.add_argument("--k3d", default="my_house.k3d", help="Output .k3d path")
    parser.add_argument(
        "--k", type=int, default=2, help="Number of nearest neighbors to find"
    )
    args = parser.parse_args()

    # Prepend 'examples/' to paths if not already there
    args.csv = os.path.join('examples', args.csv) if not args.csv.startswith('examples') else args.csv
    args.gltf = os.path.join('examples', args.gltf) if not args.gltf.startswith('examples') else args.gltf
    args.k3d = os.path.join('examples', args.k3d) if not args.k3d.startswith('examples') else args.k3d

    try:
        generate(args.csv, args.gltf, args.k3d, args.k)
        print(f"Successfully generated '{args.gltf}' and '{args.k3d}'")
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
