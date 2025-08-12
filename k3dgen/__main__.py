import argparse
import base64
import json
import os
from typing import List

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

# Constants
FLOAT = 5126
VEC3 = "VEC3"


def generate(csv_path: str, gltf_path: str, k3d_path: str) -> None:
    """Generate a glTF scene and accompanying .k3d metadata from a CSV file.

    This function implements the K3D standard by creating two files:
    1. A .k3d file containing the high-dimensional vectors and metadata,
       conforming to the k3d_file.schema.json.
    2. A .gltf file with one node per data point, where each node is
       translated to its 3D position (from PCA) and contains a `k3dId`
       in its `extras` to link back to the .k3d file. The glTF also
       includes the `K3D_nodes` extension.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file. Expected columns are 'id', 'label' (optional),
        and numeric vector components.
    gltf_path : str
        Destination filepath for the generated glTF scene.
    k3d_path : str
        Destination filepath for the generated .k3d metadata file.
    """
    # 1. Read and process the input CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file '{csv_path}': {exc}") from exc

    if "id" in df.columns:
        ids: List[str] = df["id"].astype(str).tolist()
        labels: List[str] = (
            df["label"].astype(str).tolist() if "label" in df.columns else ids
        )
        df_vectors = df.drop(columns=["id", "label"], errors="ignore")
    else:
        ids = [str(i) for i in range(len(df))]
        labels = ids
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

    source_vectors = df_vectors.to_numpy(dtype=float)

    # 2. Perform PCA to get 3D positions
    try:
        pca = PCA(n_components=3)
        points_3d = pca.fit_transform(source_vectors)
    except Exception as exc:
        raise ValueError(f"PCA computation failed: {exc}") from exc

    # 3. Create and write the .k3d file
    k3d_nodes = [
        {
            "id": _id,
            "sourceVector": source_vectors[i].tolist(),
            "metadata": {"label": labels[i]},
            "neighbors": [],
        }
        for i, _id in enumerate(ids)
    ]
    k3d_content = {
        "asset": {"version": "0.1", "generator": "k3dgen"},
        "nodes": k3d_nodes,
    }
    try:
        with open(k3d_path, "w", encoding="utf-8") as f:
            json.dump(k3d_content, f, indent=2)
    except OSError as exc:
        raise OSError(f"Failed to write .k3d file '{k3d_path}': {exc}") from exc

    # 4. Create and write the .gltf file
    # Define a single-point mesh that all nodes will instance
    point_positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    data_bytes = point_positions.tobytes()
    uri = "data:application/octet-stream;base64," + base64.b64encode(
        data_bytes
    ).decode("ascii")

    buffer = Buffer(byteLength=len(data_bytes), uri=uri)
    buffer_view = BufferView(buffer=0, byteOffset=0, byteLength=len(data_bytes))
    accessor = Accessor(
        bufferView=0,
        byteOffset=0,
        componentType=FLOAT,
        count=1,
        type=VEC3,
        max=[0.0, 0.0, 0.0],
        min=[0.0, 0.0, 0.0],
    )
    primitive = Primitive(attributes={"POSITION": 0}, mode=0)  # mode 0 = POINTS
    mesh = Mesh(primitives=[primitive])

    # Create one glTF node for each data point
    gltf_nodes = [
        Node(
            mesh=0,  # All nodes instance the same single-point mesh
            translation=points_3d[i].tolist(),
            extras={"k3dId": _id},
        )
        for i, _id in enumerate(ids)
    ]

    scene = Scene(nodes=list(range(len(gltf_nodes))))

    # Add the K3D extension linking to the .k3d file
    k3d_filename = os.path.basename(k3d_path)
    extensions = {"K3D_nodes": {"uri": k3d_filename, "nodeProperty": "extras.k3dId"}}

    gltf = GLTF2(
        asset=Asset(version="2.0", generator="k3dgen"),
        scenes=[scene],
        scene=0,
        nodes=gltf_nodes,
        meshes=[mesh],
        accessors=[accessor],
        bufferViews=[buffer_view],
        buffers=[buffer],
        extensionsUsed=["K3D_nodes"],
        extensions=extensions,
    )

    try:
        gltf.save(gltf_path)
    except OSError as exc:
        raise OSError(f"Failed to write glTF file '{gltf_path}': {exc}") from exc


def main() -> None:
    """Command-line entry point for generating K3D assets."""
    parser = argparse.ArgumentParser(description="Generate glTF + .k3d from vectors")
    parser.add_argument("csv", help="CSV file with id + vector columns")
    parser.add_argument("--gltf", default="output.gltf", help="Output glTF path")
    parser.add_argument("--k3d", default="output.k3d", help="Output .k3d path")
    args = parser.parse_args()
    try:
        generate(args.csv, args.gltf, args.k3d)
    except Exception as exc:  # pragma: no cover - CLI wrapper
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
