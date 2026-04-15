# Knowledge3D Canonical Opcodes & Kernels Registry

## Overview

This document serves as the authoritative canonical source for all Knowledge3D opcodes and GPU kernels. The registry contains **307 opcodes** across **6 functional tiers** and **99 kernel files** (72 CUDA, 27 PTX) mapped to their corresponding operations.

## Registry Statistics

- **Total Opcodes**: 307
- **Total Kernel Files**: 99
- **CUDA Kernels**: 72
- **PTX Kernels**: 27
- **Orphaned Kernels**: 94 (require opcode binding)
- **Orphaned Opcodes**: 237 (require kernel implementation)

## Canonical Address Space Architecture

### Tier-0: Core Mathematical Operations (0x0000-0x00FF)
- **Count**: 206 opcodes
- **Domain**: Basic arithmetic, transcendental functions, comparisons
- **Examples**: `OP_ADD`, `OP_SIN`, `OP_SQRT`, `OP_GT`

### Tier-1: Cooperative Groups (0x0100-0x01FF)  
- **Count**: 48 opcodes
- **Domain**: Warp-level synchronization, shared memory operations
- **Examples**: `OP_BARRIER`, `OP_SHFL_SYNC`, `OP_REDUCTION_TREE`

### Tier-2: Physics Simulation (0x0200-0x02FF)
- **Count**: 53 opcodes
- **Domain**: SPH, FEM, rigid body dynamics
- **Examples**: `OP_PH_BROAD_PHASE`, `OP_SPH_DENSITY`, `OP_FEM_STRESS_TENSOR`

### Tier-3: Computer Algebra System (0x0300-0x03FF)
- **Count**: Reserved for future expansion
- **Domain**: Symbolic mathematics, polynomial operations
- **Examples**: `OP_POLY_COEFF`, `OP_SIMPLIFY`, `OP_SOLVE_LINEAR`

### Tier-4: Drawing & Rendering (0x0400-0x04FF)
- **Count**: Reserved for future expansion
- **Domain**: Procedural drawing, rasterization, shading
- **Examples**: `OP_DRAW_LINE`, `OP_RAY_INTERSECT_TRI`, `OP_PHONG_SHADE`

### Tier-5: Qdrant Vector DB Operations (0x0500-0x05FF)
- **Count**: Reserved for future expansion
- **Domain**: Distance metrics, quantization, indexing
- **Examples**: `OP_L2_DISTANCE`, `OP_HNSW_SEARCH`, `OP_SQ_ENCODE`

### Tier-6+: Extension APIs (0x0600+)
- **Count**: Reserved for future expansion
- **Domain**: Filter operations, payload handling, vendor extensions

## Sovereignty Compliance

All opcodes and kernels in this registry adhere to Knowledge3D sovereignty principles:

- **GPU-First**: No CPU fallbacks in hot paths
- **Deterministic**: Reproducible results across runs
- **Memory Safe**: Defined shared memory contracts
- **Algorithmically Sound**: Provenance tracking for all operations

## Qdrant Integration

The complete registry is available as a Qdrant collection: `k3d_opcodes_kernels_canonical`

- **Total Points**: 406
- **Opcode Points**: 307
- **Kernel Points**: 99
- **Vector Dimensions**: 384
- **Search Capabilities**: Semantic search by functionality, opcode lookup, kernel mapping

## Usage Examples

### Semantic Search
```python
# Search for distance-related operations
query = "distance metric L2 cosine similarity"
results = qdrant.search(
    collection_name="k3d_opcodes_kernels_canonical",
    query_vector=embed(query),
    limit=10
)
```

### Opcode Lookup
```python
# Find specific opcode by name
opcode_info = qdrant.retrieve(
    collection_name="k3d_opcodes_kernels_canonical",
    ids=[hash("OP_L2_DISTANCE") % (10**8)]
)
```

### Kernel Mapping
```python
# Find kernels that implement specific opcodes
kernels = qdrant.search(
    collection_name="k3d_opcodes_kernels_canonical",
    query_filter={
        "must": [
            {"key": "type", "match": {"value": "kernel"}},
            {"key": "opcodes_referenced", "match": {"value": "OP_L2_DISTANCE"}}
        ]
    }
)
```

## Critical Issues & Reconciliation

### High Priority Orphaned Opcodes
These opcodes have registry entries but no corresponding kernel implementations:

1. **OP_FFT_RADIX8 (0x0067)** - References non-existent `fft_radix8.cuh` (should be `fft_radix8_opt.cuh`)
2. **OP_SPH_SURFACE_TENSION (0x0245)** - Missing `sph_surface.cuh` file
3. **OP_MANHATTAN (0x0508)** - Kernel exists but dispatcher missing

### High Priority Orphaned Kernels
These kernels exist but lack opcode bindings:

1. **apply_payload_mask** - Needs `OP_FILTER_PAYLOAD (0x0601)`
2. **filter_by_range_float** - Needs `OP_FILTER_RANGE (0x0602)`
3. **manhattan_distance** - Needs proper opcode assignment

## File Structure

```
docs/
├── opcodes_manifest.json          # Complete opcode inventory (307 entries)
├── kernels_manifest.json          # Complete kernel inventory (99 entries)
├── qdrant_batch.json              # Qdrant upload batch (406 points)
├── qdrant_population_report.json  # Population summary and statistics
└── canonical_opcodes_registry.md  # This document
```

## Maintenance & Updates

This registry is automatically updated via GitHub Actions when:
- New opcodes are added to `rpn_opcodes.py`
- Kernel files are modified in `knowledge3d/cranium/kernels/`
- Sovereignty compliance requirements change

## Validation Scripts

Use the provided validation scripts to verify registry integrity:

```bash
# Validate opcode-kernel mappings
python scripts/validate_mappings.py

# Check sovereignty compliance
python scripts/validate_sovereignty.py

# Audit for orphaned entries
python scripts/audit_orphans.py
```

## Contact & Contributions

For updates, corrections, or additions to this registry:
- Submit issues to the Knowledge3D GitHub repository
- Follow the sovereignty compliance guidelines
- Include proper documentation and test coverage

---

*This registry was generated on 2026-04-13 and represents the complete canonical source of truth for Knowledge3D opcodes and kernels.*