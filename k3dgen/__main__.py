import argparse
import base64
import json
import os
from typing import List, Tuple

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


def load_vectors(csv_path: str) -> Tuple[List[str], np.ndarray]:
    """Load vectors from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file '{csv_path}': {exc}") from exc

    if "id" in df.columns:
        ids: List[str] = df["id"].astype(str).tolist()
        df_vectors = df.drop(columns=["id"])
    else:
        ids = [str(i) for i in range(len(df))]
        df_vectors = df

    if df_vectors.empty:
        raise ValueError("CSV must contain at least one vector column")

    try:
        for col in df_vectors.columns:
            df_vectors[col] = pd.to_numeric(df_vectors[col], errors="raise")
    except ValueError as exc:
        raise ValueError(f"Non-numeric data found in column '{col}': {exc}") from exc

    if df_vectors.isnull().any().any():
        raise ValueError("CSV contains missing values in vector columns")

    return ids, df_vectors.to_numpy(dtype=float)


def reduce_dimensions(vectors: np.ndarray) -> np.ndarray:
    """Reduce the dimensionality of vectors to 3D using PCA."""
    try:
        pca = PCA(n_components=3)
        return pca.fit_transform(vectors)
    except Exception as exc:
        raise ValueError(f"PCA computation failed: {exc}") from exc


def find_neighbors(vectors: np.ndarray, k: int) -> np.ndarray:
    """Find the k-nearest neighbors for each vector."""
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
                "metadata": {"label": point_id},
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
    gltf_dir = os.path.dirname(gltf_path)
    relative_k3d_path = os.path.relpath(k3d_path, gltf_dir)
    relative_schema_path = os.path.relpath("spec/k3d_node_schema.json", gltf_dir)

    gltf = GLTF2(
        asset=Asset(generator="k3dgen"),
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
    # 1. Load the high-dimensional embeddings from the CSV file.
    ids, embeddings = load_vectors(csv_path)

    # 2. Reduce the dimensionality of the embeddings to 3D for visualization.
    points = reduce_dimensions(embeddings)

    # 3. Find the k-nearest neighbours for each point
    neighbor_indices = find_neighbors(embeddings, k)

    # 4. Create the .k3d file with the full embeddings and metadata.
    create_k3d_file(k3d_path, ids, points, embeddings, neighbor_indices)

    # 5. Create the .gltf file with the 3D positions and a link to the .k3d file.
    create_gltf_file(gltf_path, k3d_path, ids, points)


def main() -> None:
    """Command-line entry point for generating K3D assets."""
    parser = argparse.ArgumentParser(description="Generate glTF + .k3d from vectors")
    parser.add_argument("csv", help="CSV file with id + vector columns")
    parser.add_argument("--gltf", default="output.gltf", help="Output glTF path")
    parser.add_argument("--k3d", default="output.k3d", help="Output .k3d path")
    parser.add_argument(
        "--k", type=int, default=5, help="Number of nearest neighbors to find"
    )
    args = parser.parse_args()
    try:
        generate(args.csv, args.gltf, args.k3d, args.k)
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
