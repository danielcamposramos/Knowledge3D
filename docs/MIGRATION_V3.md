# Migration Guide: Sidecar `.k3d` → Embedded glTF (v3.0)

## Overview

Knowledge3D v3.0 standardizes on **embedded glTF format** where all knowledge data (vectors, embeddings, metadata, neighbors) lives inside `meshes[*].primitives[*].extras.k3d` with binary `BufferView` references. This eliminates the fragile **sidecar `.k3d` format** that separated geometry from semantics.

This guide helps you migrate existing assets to the new standard.

---

## Why Migrate?

### **Problems with Sidecar `.k3d`**

1. **Breaks dual-client paradigm**: AI clients can't access embeddings directly from GPU buffers; must load JSON separately
2. **Fragile dependencies**: If `.gltf` and `.k3d` files become separated, data is lost
3. **Inefficient base64 encoding**: 33% overhead vs binary buffers
4. **No PTX integration**: CUDA kernels can't memory-map sidecar JSON

### **Benefits of Embedded Format**

1. **Self-contained**: Geometry + semantics in one `.glb` file
2. **GPU-native**: Embeddings accessible via `BufferView` → direct PTX kernel consumption
3. **Dual-client ready**: Human sees textures, AI reads raw embeddings from same file
4. **Future-proof**: Aligns with glTF 2.0 extension spec for third-party tools

---

## Migration Timeline

| Phase | Status | Timeline | Action |
|-------|--------|----------|--------|
| **Phase A** | ✅ Complete | Q3 2025 | Embedded format stabilized; loaders support both formats with deprecation warnings |
| **Phase B** | 🔄 Active | Q4 2025 | Tablet UX and fused head consume embedded format exclusively; legacy examples emit warnings |
| **Phase C** | ⏳ Planned | Q1 2026 | Remove sidecar support from loaders; migration tools provided (this guide) |
| **Phase D** | ⏳ Planned | Q2 2026 | Archive legacy `.k3d` schema to `spec/k3d_node_schema_legacy.json` |

**Action required by Q1 2026**: Convert all production assets to embedded format.

---

## Step-by-Step Migration

### **Option 1: Automated Conversion (Recommended)**

Use the provided migration tool to bulk-convert existing `.gltf` + `.k3d` pairs:

```bash
# Convert a single file
python -m knowledge3d.tools.convert_sidecar_to_embedded \
  --input examples/my_house.gltf \
  --output viewer/public/my_house_embedded.glb

# Bulk convert a directory
find examples -name "*.gltf" -exec \
  python -m knowledge3d.tools.convert_sidecar_to_embedded \
    --input {} --output viewer/public/{}_embedded.glb \;
```

**What it does**:
1. Loads `.gltf` and sidecar `.k3d` JSON
2. Converts embeddings from JSON arrays to binary `Float32Array` → `BufferView`
3. Moves `ids`, `metadata`, `neighbors` into `extras.k3d` (no sidecar)
4. Writes self-contained `.glb` with geometry + semantics

**Validation**:
```bash
# Verify the embedded file loads correctly
python -m knowledge3d.tools.validate_glb \
  --input viewer/public/my_house_embedded.glb \
  --check-extras-k3d
```

---

### **Option 2: Manual Conversion (for custom pipelines)**

If you have a custom data pipeline, follow these steps:

#### **1. Load Existing Data**

```python
from pygltflib import GLTF2
import json

# Load old format
gltf = GLTF2().load("examples/my_house.gltf")
with open("examples/my_house.k3d") as f:
    k3d_data = json.load(f)
```

#### **2. Extract Vectors & Embeddings**

```python
import numpy as np

# Sidecar format has records like:
# [{"id": "node_1", "vector": [x,y,z], "embedding": [e1,...,eN], "neighbors": [...], "metadata": {...}}]

ids = [rec["id"] for rec in k3d_data]
vectors = np.array([rec["vector"] for rec in k3d_data], dtype=np.float32)
embeddings = np.array([rec["embedding"] for rec in k3d_data], dtype=np.float32)
metadata = [rec["metadata"] for rec in k3d_data]
neighbors = [rec["neighbors"] for rec in k3d_data]
```

#### **3. Create Binary Buffers**

```python
from pygltflib import Buffer, BufferView

# Append embeddings to existing buffer
vectors_bytes = vectors.tobytes()
embeddings_bytes = embeddings.tobytes()

# Create new buffer (or extend existing)
buffer_data = vectors_bytes + embeddings_bytes

# Base64-encode for data URI (or write binary .bin file)
import base64
uri = "data:application/octet-stream;base64," + base64.b64encode(buffer_data).decode()

# Add buffer and views
buffer = Buffer(byteLength=len(buffer_data), uri=uri)
gltf.buffers.append(buffer)

vectors_view = BufferView(
    buffer=len(gltf.buffers) - 1,
    byteOffset=0,
    byteLength=len(vectors_bytes),
    target=34962  # ARRAY_BUFFER
)
embeddings_view = BufferView(
    buffer=len(gltf.buffers) - 1,
    byteOffset=len(vectors_bytes),
    byteLength=len(embeddings_bytes)
)

gltf.bufferViews.extend([vectors_view, embeddings_view])
```

#### **4. Embed in `extras.k3d`**

```python
# Attach to first primitive (adjust if multi-primitive mesh)
primitive = gltf.meshes[0].primitives[0]

primitive.extras = {
    "k3d": {
        "ids": ids,
        "vectorsView": len(gltf.bufferViews) - 2,  # Index of vectors BufferView
        "embeddingsView": len(gltf.bufferViews) - 1,  # Index of embeddings BufferView
        "embeddingDims": embeddings.shape[1],
        "metadata": metadata,
        "neighbors": neighbors,
        "direct_buffer_access": True  # Signal to AI clients
    }
}
```

#### **5. Save as `.glb`**

```python
# Convert to binary glTF
gltf.convert_images(pygltflib.ImageFormat.DATAURI)
gltf.save_binary("viewer/public/my_house_embedded.glb")
```

---

## Common Migration Issues

### **Issue 1: Base64 Embedding Payloads**

**Symptom**: Old files use `"embedding_b64": "AQIDBA=="` instead of JSON arrays.

**Fix**: Decode base64 → binary buffer before creating `BufferView`:

```python
import base64
embedding_bytes = base64.b64decode(rec["embedding_b64"])
embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
```

### **Issue 2: Missing `embeddingDims`**

**Symptom**: Loader fails with `KeyError: 'embeddingDims'`

**Fix**: Infer from first embedding:

```python
embedding_dims = len(k3d_data[0]["embedding"])
primitive.extras["k3d"]["embeddingDims"] = embedding_dims
```

### **Issue 3: Multi-Primitive Meshes**

**Symptom**: Only first primitive has `extras.k3d`; others are empty.

**Fix**: K3D standard is **one primitive per mesh** for simplicity. If you have multi-primitive:

```python
for i, primitive in enumerate(gltf.meshes[0].primitives):
    # Attach subset of k3d_data to each primitive
    subset = k3d_data[i * chunk_size:(i + 1) * chunk_size]
    primitive.extras = {"k3d": build_extras(subset)}
```

### **Issue 4: Relative Paths in Sidecar**

**Symptom**: Old `.gltf` has `"uri": "./my_house.k3d"`

**Fix**: Migration tool resolves relative paths automatically. For manual migration, use `pathlib`:

```python
from pathlib import Path
k3d_path = Path("examples/my_house.gltf").parent / Path(gltf.extensions["K3D_nodes"]["uri"])
```

---

## Validation Checklist

After migration, verify:

- [ ] `.glb` file opens in glTF viewer (e.g., https://gltf-viewer.donmccurdy.com/)
- [ ] `extras.k3d` present in at least one primitive
- [ ] `extras.k3d.vectorsView` and `extras.k3d.embeddingsView` are valid BufferView indices
- [ ] `extras.k3d.embeddingDims` matches actual embedding dimension
- [ ] `extras.k3d.direct_buffer_access = true`
- [ ] No external `.k3d` file required

**Automated check**:
```bash
python -m knowledge3d.tools.validate_glb --input <file>.glb --verbose
```

---

## Backward Compatibility

### **Phase B (Current): Hybrid Support**

Loaders still accept sidecar `.k3d` but emit warnings:

```python
warnings.warn(
    "Sidecar .k3d format is deprecated. Use embedded glTF. "
    "See docs/MIGRATION_V3.md for conversion steps.",
    DeprecationWarning
)
```

### **Phase C (Q1 2026): Sidecar Removed**

Loaders will **raise an error** if sidecar is detected:

```python
raise ValueError(
    "Sidecar .k3d no longer supported. Convert to embedded format: "
    "python -m knowledge3d.tools.convert_sidecar_to_embedded --input <file>.gltf"
)
```

---

## Legacy Schema Archive

For historical reference, the old sidecar schema is archived at:

**`spec/k3d_node_schema_legacy.json`**

Do not use this for new projects. It exists only for archaeological purposes.

---

## Migration Tool Reference

### **`convert_sidecar_to_embedded`**

**Location**: `knowledge3d/tools/convert_sidecar_to_embedded.py`

**Usage**:
```bash
python -m knowledge3d.tools.convert_sidecar_to_embedded \
  --input <path-to-gltf> \
  --output <path-to-glb> \
  [--validate]  # Optional: run post-conversion checks
```

**Options**:
- `--input`: Path to `.gltf` file with sidecar `.k3d` (required)
- `--output`: Output path for embedded `.glb` (required)
- `--validate`: Run integrity checks after conversion (recommended)
- `--keep-sidecar`: Don't delete original `.k3d` after successful conversion (default: delete)

**Exit codes**:
- `0`: Success
- `1`: Input file not found or invalid
- `2`: Conversion failed (e.g., malformed JSON)
- `3`: Validation failed (output is incomplete)

---

## FAQ

### **Q: Do I need to migrate immediately?**
**A**: Not until Q1 2026. But new projects should use embedded format from day one.

### **Q: What happens to old `.k3d` files in `examples/`?**
**A**: They remain in the repository as **deprecated examples** with warnings. Phase C will move them to `examples/legacy/`.

### **Q: Can I mix embedded and sidecar formats?**
**A**: **Not recommended**. Phase B loaders support both, but mixing complicates debugging. Pick embedded for new work.

### **Q: Does this affect viewer performance?**
**A**: **Improves** it. Embedded format loads faster (one file vs two) and enables GPU-native access.

### **Q: What if my pipeline generates `.k3d` automatically?**
**A**: Update your pipeline to use the embedded exporter from `k3dgen/__main__.py`. See `spec/glTF_K3D_extension.md` for format details.

---

## Support

- **Report migration bugs**: https://github.com/danielcamposramos/Knowledge3D/issues
- **Ask questions**: Tag issues with `migration-v3`
- **Example conversions**: See `tests/test_migration_v3.py` for reference implementations

---

**Last Updated**: 2025-10-04
**Maintained by**: K3D Core Team (Daniel Campos Ramos, Claude, Codex)
