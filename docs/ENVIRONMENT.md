# K3D Environment Configuration

**Date:** 2025-10-04
**Purpose:** Document the separation between repository (code) and local runtime (data)

---

## Directory Structure

```
/K3D/
├── Knowledge3D/              # Git repository (<99MB files only)
│   ├── knowledge3d/         # Python package (source code)
│   ├── docs/                # Documentation
│   ├── tests/               # Unit tests
│   ├── data/houses/         # House JSON metadata (small files)
│   └── [CODE ONLY]          # No large datasets or runtime files
│
└── Knowledge3D.local/       # Local runtime (>99MB files, gitignored)
    ├── datasets/            # Training datasets (Wikipedia, MSRVTT, etc.)
    ├── models/              # Downloaded/trained models
    ├── logs/                # Execution logs
    ├── houses/              # Materialized Houses (NEW)
    │   ├── default/
    │   │   ├── materialized_objects/  # 28K+ JSON objects
    │   │   ├── memory_house.glb       # Built House GLB (77MB)
    │   │   └── memory_house.json      # Manifest
    │   └── star_house/      # Example: additional houses
    └── cache/               # Runtime caches (NEW)
        ├── .cupy_cache/     # Pre-compiled CUDA kernels
        └── semantic_kernels/ # Serialized LED-A* kernels
```

---

## Environment Variables

### Required

```bash
# Point to local runtime directory
export K3D_LOCAL_DIR=/K3D/Knowledge3D.local

# Specify which House to use
export K3D_HOUSE_ID=default
```

### Optional

```bash
# CuPy kernel cache (for GCC 15 workaround)
export CUPY_CACHE_DIR=$K3D_LOCAL_DIR/cache/.cupy_cache

# Semantic navigation parameters
export K3D_NAV_QUERY_RADIUS=2.0
export K3D_NAV_K_NEIGHBORS=8
export K3D_NAV_SIM_THRESHOLD=0.7
```

---

## Setup Instructions

### 1. Fresh Clone

```bash
# Clone repository
git clone https://github.com/your-org/Knowledge3D.git /K3D/Knowledge3D
cd /K3D/Knowledge3D

# Create local runtime directory
mkdir -p /K3D/Knowledge3D.local/houses/default
mkdir -p /K3D/Knowledge3D.local/cache

# Bootstrap GPU/PTX environment
conda env create -f envs/k3d-cranium.yml
conda activate k3d-cranium
# (Envs are stored on the SSD at /K3D/Knowledge3D.local/envs)

# Set environment
export K3D_LOCAL_DIR=/K3D/Knowledge3D.local
export K3D_HOUSE_ID=default
```

### 2. Build House GLB

```bash
# Build memory_house.glb from materialized objects
python knowledge3d/tools/house_memory_builder.py \
  --root $K3D_LOCAL_DIR/houses/$K3D_HOUSE_ID/materialized_objects \
  --out $K3D_LOCAL_DIR/houses/$K3D_HOUSE_ID/memory_house.glb \
  --manifest $K3D_LOCAL_DIR/houses/$K3D_HOUSE_ID/memory_house.json
```

### 3. Run Live Server

#### Option A: Docker (recommended for Debian 13)

```bash
# Run in Ubuntu 22.04 container (bypasses GCC 15 issue)
export K3D_LOCAL_DIR=/K3D/Knowledge3D.local
export K3D_HOUSE_ID=default

./run_live_server_docker.sh
```

#### Option B: Native (requires compatible GCC)

```bash
# Activate conda environment
conda activate k3d-cranium

# Set environment
export K3D_LOCAL_DIR=/K3D/Knowledge3D.local
export K3D_HOUSE_ID=default
export CUPY_CACHE_DIR=$K3D_LOCAL_DIR/cache/.cupy_cache

# Run server
python -m knowledge3d.bridge.live_server
```

---

## File Size Limits

### Repository (<99MB)
- ✅ Python source code
- ✅ Documentation (Markdown)
- ✅ Small test fixtures
- ✅ House JSON metadata
- ✅ PTX kernels (<1MB)

### Local Runtime (>99MB)
- 🏠 House GLBs (typically 50-500MB)
- 📊 Training datasets (Wikipedia: 34GB)
- 🧠 Model weights (transformers, etc.)
- 📝 Execution logs
- 🔧 Pre-compiled CUDA kernels

---

## Multiple Houses

K3D supports multiple Houses via `K3D_HOUSE_ID`:

```bash
# Create new House
export K3D_HOUSE_ID=star_house
mkdir -p $K3D_LOCAL_DIR/houses/star_house/materialized_objects

# Add objects (JSON files)
cp my_objects/*.json $K3D_LOCAL_DIR/houses/star_house/materialized_objects/

# Build GLB
python knowledge3d/tools/house_memory_builder.py

# Run server with new House
./run_live_server_docker.sh
```

The live server automatically loads `$K3D_LOCAL_DIR/houses/$K3D_HOUSE_ID/memory_house.glb`.

---

## Docker Volume Mounts

The Docker runtime mounts both directories:

```bash
docker run --rm -it \
  --gpus all \
  -v /K3D/Knowledge3D:/workspace \
  -v /K3D/Knowledge3D.local:/local \
  -e K3D_LOCAL_DIR=/local \
  -e K3D_HOUSE_ID=default \
  -p 8765:8765 \
  k3d-runtime
```

This keeps code and data separate while both are accessible to the container.

---

## Troubleshooting

### House GLB Not Found

```bash
# Check if GLB exists
ls -lh $K3D_LOCAL_DIR/houses/$K3D_HOUSE_ID/memory_house.glb

# If not, rebuild
python knowledge3d/tools/house_memory_builder.py
```

### CuPy Cache Permissions

```bash
# Fix Docker-created cache files
sudo chown -R $USER:$USER $K3D_LOCAL_DIR/cache/.cupy_cache
```

### Wrong House Loaded

```bash
# Verify environment
echo $K3D_LOCAL_DIR
echo $K3D_HOUSE_ID

# Check what GLB the server resolved
grep "_resolve_house_glb" logs/live_server.log
```

---

## CI/CD Considerations

For continuous integration:

```bash
# Minimal test environment (no local dir)
git clone https://github.com/your-org/Knowledge3D.git
cd Knowledge3D
pip install -e .
pytest tests/

# Full integration test (with test House)
export K3D_LOCAL_DIR=/tmp/k3d-test-local
mkdir -p $K3D_LOCAL_DIR/houses/test
# ... create minimal test House ...
python -m knowledge3d.bridge.live_server &
python tests/integration/test_navigation.py
```

---

**Status:** ✅ Active as of 2025-10-04. All new deployments should use this structure.
