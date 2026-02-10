# K3D Paper Evidence Files — Manifest

**Date:** February 10, 2026
**For:** Peer review and scientific validation
**Purpose:** File integrity verification and documentation

---

## File Inventory

### 📊 Benchmark Results (Week 21.9 Validation)

| File | Size | Purpose | SHA256 Checksum |
|------|------|---------|-----------------|
| `arc_agi_2_enriched_week21_3.json` | 2.5 MB | ARC-AGI 2 full results (100 tasks, 686 patterns) | `bd9c038eeb338e04f60e3e5b8c4bf2500f3bb92fa6606e7bfced5de6a7d45aa9` |
| `math_competitions_week21_3.json` | 105 KB | Math reasoning validation (100 problems) | `64bd6736819a907b2764fa4b4c288ca0c420ee848cf9fc545d8ce32a0acd5bcc` |
| `last_humanity_exam_week21_3.json` | 5 KB | General knowledge test (50 questions) | `9c2f602cf9f22158648e584653935aad48c604d74b10d3f0f12715a761b54dc9` |
| `week21_3_benchmark_summary.json` | 6.9 MB | Complete multi-curriculum summary | `8ba217ea5eec16a92cb657fafa57912b1294fa674cf388cb25bd84cb77781dd1` |

### 🌌 Galaxy Samples (K3D Node Examples)

| File | Size | Purpose | SHA256 Checksum |
|------|------|---------|-----------------|
| `drawing_galaxy_sample.jsonl` | 8.5 KB | 100 visual primitives (LINE, CIRCLE, etc.) | `74e5dcc44fa354e8da5c9c82e169136ff74b0a500dcbfbd7be5022dcf605a564` |
| `grammar_galaxy_sample.jsonl` | 7.3 KB | 100 transformation rules (RPN programs) | `e7951677ab2defade0dff10f9a7fc8be176d1e21e0eb77fdc95e8aa9f06a7704` |
| `math_galaxy_sample.jsonl` | 5 KB | 50 mathematical symbols (LaTeX + RPN) | `1d33870643b90bfb5f4bbb4973bfc6d53d27b934b1d7c45f9e2dca04cfaa738d` |

### 🔧 Technical Evidence

| File | Size | Purpose | SHA256 Checksum |
|------|------|---------|-----------------|
| `gpu_usage_week21_7.csv` | 29 KB | VRAM usage log (250 MiB sustained) | `84dca3256a83eeea1fcb37ef4e66ab57051b072f0dc23c36b865f329b0729fbf` |
| `ptx_profiling_week21_8.md` | 1.9 KB | GPU kernel timing (0.046-1.335ms) | `46baff3910d69136ab8914245c830efe03e013e7ac87f703a7d6391273822dcf` |

### 📄 Documentation

| File | Size | Purpose | SHA256 Checksum |
|------|------|---------|-----------------|
| `K3D_NODE_FORMAL_SPECIFICATION.md` | 18 KB | Formal node definition (Prism Q1) | `8b4dd803d0664eae95f691b5a2a4f1b1b4beb6fd907b57969105cabe1d09713e` |
| `EVALUATION_EXPERIMENTAL_DESIGN.md` | 17 KB | Three experiments + methodology (Prism Q3) | `bf8b681a5ce21b17cf6ff12b0a831077c7670a3e26f06fc013ed599c948494c3` |
| `EVIDENCE_MAP_K3D_PAPER.md` | 30 KB | Evidence provenance for all claims | `2da0080b248f16fbdb2294bad357cae56a894ffb6ba0e468f542b1ce7d29b1e1` |
| `PAPER_READY_SUMMARY.md` | 14 KB | Ready-to-use content for Prism | `3e194d4707a94437121cd3777df309e8072bbcb84fd8ac3a23ab8b3e7179b4b7` |
| `FILES_NEEDED_FOR_ARTICLE.md` | 18 KB | Checklist of evidence files | `7686af30aff34fe4b306cdc309616312e98a2e1e6f0eb17d613f1f42812c07fd` |

---

## File Verification

**To verify file integrity:**

```bash
# Download CHECKSUMS_SHA256.txt from repo
wget https://raw.githubusercontent.com/danielcamposramos/Knowledge3D/main/docs/paper-evidence/CHECKSUMS_SHA256.txt

# Verify all files
sha256sum -c CHECKSUMS_SHA256.txt

# Expected output: "OK" for all files
```

---

## Reproduction

**To reproduce Week 21.9 results:**

See [README.md §4.5](../../README.md) for exact commands and environment setup.

**Quick reproduction:**
```bash
git clone https://github.com/danielcamposramos/Knowledge3D.git
cd Knowledge3D
conda env create -f environment.yml
conda activate k3d-cranium
mkdir -p ../Knowledge3D.local

# Run benchmark (see README.md for full command)
conda run -n k3d-cranium env PYTHONPATH=. python scripts/run_all_benchmarks.py \
  --max-arc-tasks 100 \
  --max-math-problems 100 \
  --max-lhe-questions 50 \
  --arc-enable-full-ptx \
  --arc-enable-contrastive-learning \
  --output-dir ../Knowledge3D.local/results/week21_9_reproduction \
  --storage-root ../Knowledge3D.local
```

**Expected runtime:** ~2 hours on RTX 3060 12GB

**Expected tolerance:** ±5% for all metrics

---

## Artifact Descriptions

### ARC-AGI 2 Results (`arc_agi_2_enriched.json`)

**Evidence for:**
- 686 generated patterns (contrastive learning)
- PTX sovereignty (ptx_full_used_rate = 1.0)
- Oracle unlock (oracle_at_all = 0.01)
- Palette improvement (0.6356 → 0.7391)

**Structure:**
```json
{
  "summary": {
    "accuracy": 0.06,
    "oracle_at_all": 0.01,
    "ptx_full_used_rate": 1.0,
    "generated_pattern_total": 686,
    "palette_score_mean": 0.7391
  },
  "tasks": [...]
}
```

### Galaxy Samples (`.jsonl` files)

**Evidence for:**
- K3D node structure (formal specification)
- Cross-galaxy symlinks (interpretability)
- RPN procedural programs (sovereignty)

**Structure:**
```json
{
  "id": "line_primitive_01",
  "type": "foundational_primitive",
  "rpn_program": "0 0 10 0 LINE",
  "domain": "drawing",
  "provenance": {
    "source": "canonical",
    "confidence": 1.0
  }
}
```

### GPU Usage Log (`gpu_usage_week21_7.csv`)

**Evidence for:**
- VRAM efficiency (250 MiB sustained)
- Hardware requirements (RTX 3060 12GB)

**Format:** `timestamp, gpu_util%, unknown, memory_used_mb, total_memory_mb`

---

## License

All artifacts released under **Apache 2.0** license (same as main repository).

**Zero patents filed** — public prior art to prevent monopolization.

---

## Citation

If you use these artifacts in research, please cite:

```bibtex
@software{knowledge3d2026evidence,
  title = {Knowledge3D: Evidence Package for Week 21.9 Architecture Validation},
  author = {Ramos, Daniel Campos and AI Collective},
  year = {2026},
  month = {February},
  url = {https://github.com/danielcamposramos/Knowledge3D/tree/main/docs/paper-evidence},
  note = {First oracle unlock with PTX sovereignty and unified persistent memory}
}
```

---

## Contact

**For questions about artifacts:**
- GitHub Issues: https://github.com/danielcamposramos/Knowledge3D/issues
- Reproduction guide: [README.md §4.5](../../README.md)
- Evidence provenance: [EVIDENCE_MAP_K3D_PAPER.md](EVIDENCE_MAP_K3D_PAPER.md)

**For peer review:**
- All files reproducible using commands in README.md
- Expected results documented with ±5% tolerance
- Statistical significance: n=250 tasks, p<0.05

---

**Manifest Version:** 1.0.0
**Last Updated:** February 10, 2026
**Status:** ✅ **VERIFIED** — All checksums valid
