# Knowledge3D: getting started

Part of [Knowledge3D](../../README.md), moved here whole from its front page.

## Getting Started

### Prerequisites
- CUDA-capable GPU (RTX 3060 12GB recommended, GTX 1060 6GB minimum)
- CUDA Toolkit 12.x
- Python 3.10+

### Install
```bash
git clone https://github.com/danielcamposramos/Knowledge3D.git
cd Knowledge3D
pip install -e .
```

### Runtime Workspace
```bash
mkdir ../Knowledge3D.local  # Sibling directory for runtime data
```

### Launch
```bash
# Terminal 1: Viewer
python scripts/viewer.py

# Terminal 2: Bridge
python scripts/bridge.py

# Browser: http://localhost:8000
```

**What you'll see**: 3D Galaxy Universe (Drawing, Character, Word, Grammar, Math, Audio stars) navigable in ThreeJS.

### Troubleshooting
- **CUDA Error 222**: See [CUDA_PTX_VERSION_COMPATIBILITY_GUIDE.md](../CUDA_PTX_VERSION_COMPATIBILITY_GUIDE.md)
- **NumPy in hot path**: Intentional sovereignty check — K3D rejects CPU fallbacks
- **Low VRAM**: Reduce `TRM_BATCH_SIZE` in `knowledge3d/config.py`
