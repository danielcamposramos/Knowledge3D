import argparse
import base64
import json
import os
from pathlib import Path
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


def reduce_dimensions(vectors: np.ndarray, reducer: str = "umap") -> np.ndarray:
    """Reduce dimensionality to 3D using UMAP (default) or PCA."""
    try:
        n_samples = vectors.shape[0]
        if reducer.lower() == "umap":
            # Guard: tiny datasets can fail UMAP spectral step when n_components >= n_samples
            if n_samples <= 3:
                pca = PCA(n_components=min(3, n_samples))
                projected = pca.fit_transform(vectors)
                if projected.shape[1] < 3:
                    pad = np.zeros((projected.shape[0], 3))
                    pad[:, : projected.shape[1]] = projected
                    return pad
                return projected
            try:
                import umap  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise ValueError(
                    "UMAP not available. Install umap-learn or use --reducer pca"
                ) from exc
            um = umap.UMAP(n_components=3, n_neighbors=min(15, max(2, len(vectors) - 1)))
            return um.fit_transform(vectors)
        else:
            pca = PCA(n_components=3)
            return pca.fit_transform(vectors)
    except Exception as exc:
        raise ValueError(f"Dimensionality reduction failed: {exc}") from exc


def embed_texts(text_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Tuple[List[str], np.ndarray, List[str]]:
    """Compute embeddings for lines of text using sentence-transformers.

    Returns ids, embedding matrix, and labels (trimmed text snippets).
    """
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError as exc:
        raise ValueError(f"Failed to read text file '{text_path}': {exc}") from exc
    if not lines:
        raise ValueError("Text file is empty")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ValueError(
            "sentence-transformers not available. Install it or omit --text"
        ) from exc
    model = SentenceTransformer(model_name)
    embeddings = np.asarray(model.encode(lines, convert_to_numpy=True), dtype=float)
    ids = [str(i) for i in range(len(lines))]
    # labels: first 24 chars of line
    labels = [ln if len(ln) <= 24 else (ln[:21] + "...") for ln in lines]
    return ids, embeddings, labels


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


def create_k3d_file(*args, **kwargs) -> None:  # pragma: no cover - removed functionality
    raise RuntimeError(".k3d sidecar output is no longer supported. Use embedded glTF.")


def create_gltf_file(
    gltf_path: str,
    ids: List[str],
    points: np.ndarray,
    embeddings: np.ndarray,
    neighbor_indices: np.ndarray,
    labels: List[str] | None = None,
    metadata_texts: List[str] | None = None,
    fmt: str = "gltf",
    emb_precision: str = "f32",
) -> None:
    """Create a glTF/GLB file with positions + embeddings in buffers.

    - Positions are in bufferView 0 with accessor 0 and used by POSITION attribute.
    - Embeddings are in bufferView 1 (no accessor; shape provided in extras.k3d).
    - extras.k3d contains ids, metadata, neighbors, and bufferView indices.
    """
    positions = points.astype(np.float32)
    emb = embeddings.astype(np.float32)

    pos_bytes = positions.tobytes()
    if emb_precision.lower() == "f16":
        emb_bytes = embeddings.astype(np.float16).tobytes()
    else:
        emb_bytes = emb.tobytes()
    data_bytes = pos_bytes + emb_bytes

    # Buffer and bufferViews
    if fmt not in {"gltf", "glb"}:
        raise ValueError("fmt must be 'gltf' or 'glb'")

    if fmt == "gltf":
        uri = "data:application/octet-stream;base64," + base64.b64encode(
            data_bytes
        ).decode("ascii")
        buffer = Buffer(byteLength=len(data_bytes), uri=uri)
    else:
        buffer = Buffer(byteLength=len(data_bytes))

    view_positions = BufferView(
        buffer=0, byteOffset=0, byteLength=len(pos_bytes), target=ARRAY_BUFFER
    )
    view_embeddings = BufferView(
        buffer=0, byteOffset=len(pos_bytes), byteLength=len(emb_bytes)
    )

    accessor_positions = Accessor(
        bufferView=0,
        byteOffset=0,
        componentType=FLOAT,
        count=len(points),
        type="VEC3",
        max=positions.max(axis=0).tolist(),
        min=positions.min(axis=0).tolist(),
    )

    # K3D extras payload
    neighbors: List[List[str]] = []
    for i, _ in enumerate(ids):
        neighbors.append([ids[j] for j in neighbor_indices[i]])

    meta_list = []
    for i in range(len(ids)):
        entry = {"label": (labels[i] if labels else ids[i])}
        if metadata_texts is not None and i < len(metadata_texts):
            entry["text"] = metadata_texts[i]
        meta_list.append(entry)

    k3d_payload = {
        "ids": ids,
        "vectorsView": 0,
        "embeddingsView": 1,
        "embeddingDims": int(embeddings.shape[1]),
        "embeddingPrecision": "f16" if emb_precision.lower() == "f16" else "f32",
        "metadata": meta_list,
        "neighbors": neighbors,
    }

    primitive = Primitive(
        attributes={"POSITION": 0},
        mode=0,
        extras={
            "k3dIds": ids,
            "k3d": k3d_payload,
        },
    )
    mesh = Mesh(primitives=[primitive])
    node = Node(mesh=0)
    scene = Scene(nodes=[0])

    gltf = GLTF2(
        asset=Asset(generator="k3dgen"),
        buffers=[buffer],
        bufferViews=[view_positions, view_embeddings],
        accessors=[accessor_positions],
        meshes=[mesh],
        nodes=[node],
        scenes=[scene],
        scene=0,
    )

    # For GLB embed binary blob
    if fmt == "glb":
        gltf.set_binary_blob(data_bytes)

    try:
        gltf.save(gltf_path)
    except OSError as exc:
        raise OSError(f"Failed to write glTF file '{gltf_path}': {exc}") from exc


def generate(
    csv_path: str | None,
    gltf_path: str,
    k: int,
    reducer: str = "umap",
    text_path: str | None = None,
    model_name: str | None = None,
) -> None:
    """Generate a glTF scene with embedded K3D payload from CSV or text.

    If text_path is provided, CSV is ignored.
    """
    # 1. Load embeddings
    metadata_texts: List[str] | None = None
    if text_path:
        ids, embeddings, labels = embed_texts(text_path, model_name or "sentence-transformers/all-MiniLM-L6-v2")
        # Also capture the raw text lines for metadata
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                metadata_texts = [ln.strip() for ln in f if ln.strip()]
        except OSError:
            metadata_texts = None
    else:
        assert csv_path is not None
        ids, embeddings = load_vectors(csv_path)
        labels = ids

    # Early validation for k
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if k >= len(embeddings):
        raise ValueError("k must be less than the number of vectors")

    # 2. Reduce
    points = reduce_dimensions(embeddings, reducer=reducer)

    # 3. Find the k-nearest neighbours for each point
    neighbor_indices = find_neighbors(embeddings, k)

    # 4. Create the .gltf file with embedded embeddings in primitive.extras.
    # 4. Create the .gltf/.glb with embedded buffers and labels/text metadata.
    fmt = "glb" if str(gltf_path).lower().endswith(".glb") else "gltf"
    emb_precision = _ARGS.emb_precision if '_ARGS' in globals() else 'f32'
    create_gltf_file(
        gltf_path,
        ids,
        points,
        embeddings,
        neighbor_indices,
        labels,
        metadata_texts,
        fmt,
        emb_precision,
    )


def main() -> None:
    """Command-line entry point for generating K3D assets."""
    parser = argparse.ArgumentParser(description="Generate embedded glTF/GLB from vectors or text")
    parser.add_argument("csv", nargs="?", help="CSV file with id + vector columns")
    parser.add_argument("--gltf", default="output.gltf", help="Output glTF path")
    # Sidecar output removed by design
    parser.add_argument(
        "--k", type=int, default=5, help="Number of nearest neighbors to find"
    )
    parser.add_argument(
        "--reducer", choices=["umap", "pca"], default="umap", help="Dimensionality reduction method"
    )
    parser.add_argument(
        "--text", help="Path to a text file; each non-empty line becomes a record"
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformer model for --text mode",
    )
    parser.add_argument(
        "--emb-precision",
        choices=["f32", "f16"],
        default="f32",
        help="Embedding precision in GLTF bufferView (binary). f16 halves storage at some precision cost",
    )
    args = parser.parse_args()
    global _ARGS
    _ARGS = args
    try:
        if args.text:
            generate(None, args.gltf, args.k, reducer=args.reducer, text_path=args.text, model_name=args.model)
        else:
            if not args.csv:
                parser.exit(2, "Error: CSV path required when not using --text\n")
            generate(args.csv, args.gltf, args.k, reducer=args.reducer)
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
