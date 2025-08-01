import argparse
import json
import base64
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pygltflib import GLTF2, Scene, Node, Mesh, Buffer, BufferView, Accessor, Primitive, Asset

ARRAY_BUFFER = 34962
FLOAT = 5126


def generate(csv_path, gltf_path, k3d_path):
    df = pd.read_csv(csv_path)
    if 'id' in df.columns:
        ids = df['id'].astype(str)
        vectors = df.drop(columns=['id']).to_numpy()
    else:
        ids = [str(i) for i in range(len(df))]
        vectors = df.to_numpy()

    pca = PCA(n_components=3)
    points = pca.fit_transform(vectors)

    records = [
        {"id": i, "vector": vec.tolist(), "metadata": {}}
        for i, vec in zip(ids, points)
    ]
    with open(k3d_path, 'w') as f:
        json.dump(records, f, indent=2)

    # glTF with POINTS primitive
    positions = points.astype(np.float32)
    data_bytes = positions.tobytes()
    uri = 'data:application/octet-stream;base64,' + base64.b64encode(data_bytes).decode('ascii')
    buffer = Buffer(byteLength=len(data_bytes), uri=uri)
    view = BufferView(buffer=0, byteOffset=0, byteLength=len(data_bytes), target=ARRAY_BUFFER)
    accessor = Accessor(
        bufferView=0,
        byteOffset=0,
        componentType=FLOAT,
        count=len(points),
        type="VEC3",
        max=positions.max(axis=0).tolist(),
        min=positions.min(axis=0).tolist(),
    )
    primitive = Primitive(attributes={"POSITION": 0}, mode=0)
    mesh = Mesh(primitives=[primitive])
    node = Node(mesh=0)
    scene = Scene(nodes=[0])
    gltf = GLTF2(
        asset=Asset(generator="k3dgen"),
        buffers=[buffer],
        bufferViews=[view],
        accessors=[accessor],
        meshes=[mesh],
        nodes=[node],
        scenes=[scene],
        scene=0,
    )
    gltf.save(gltf_path)


def main():
    parser = argparse.ArgumentParser(description="Generate glTF + .k3d from vectors")
    parser.add_argument('csv', help='CSV file with id + vector columns')
    parser.add_argument('--gltf', default='output.gltf', help='Output glTF path')
    parser.add_argument('--k3d', default='output.k3d', help='Output .k3d path')
    args = parser.parse_args()
    generate(args.csv, args.gltf, args.k3d)


if __name__ == '__main__':
    main()
