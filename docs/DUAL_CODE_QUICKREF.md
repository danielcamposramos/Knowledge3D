# Dual-Code Quick Reference

**TL;DR**: HR (Human-Readable) for development, MR (Machine-Runtime) for production/multi-instance.

---

## 🚦 **Decision Tree: Do I Need MR?**

```
Are you running multiple Python instances? (4+ trainers, workers)
  ├─ YES → Use MR (20-30% memory savings)
  └─ NO
      ├─ Is this edge deployment? (Raspberry Pi, mobile, <4GB RAM)
      │   ├─ YES → Use MR (critical for memory budget)
      │   └─ NO → Stick with HR (better debugging)
```

---

## ⚡ **Quick Commands**

### **Compile MR (Tier 1: Hot-Path)**
```bash
make compile-mr-core
export PYTHONPATH=../Knowledge3D.local/mr:.
python -m knowledge3d.bridge.live_server
```

### **Compile MR (Tier 2: Trainers)**
```bash
make compile-mr-trainers
PYTHONPATH=../Knowledge3D.local/mr:. python -m knowledge3d.tools.phase25.rpn_policy_trainer
```

### **Compile MR (Tier 3: Full Repo)**
```bash
make compile-mr-all  # Edge deployment only
```

### **Check Savings**
```bash
make mr-report
```

### **Clean MR**
```bash
make clean-mr
```

---

## 📊 **Memory Savings (Estimated)**

| Scenario | Instances | HR Memory | MR Memory | Saved |
|----------|-----------|-----------|-----------|-------|
| Single dev | 1 | 257KB | 180KB | 77KB (not worth it) |
| Parallel trainers | 10 | 2.5MB | 1.8MB | 700KB ✅ |
| Edge cluster | 20 | 5.1MB | 3.6MB | 1.5MB ✅ |

**Rule of thumb**: Savings matter when **instances × module_size > 1MB**.

---

## 🎯 **What Gets Optimized**

### **Removed by MR**
- ✅ Comments (`# ...`)
- ✅ Docstrings (`"""..."""`)
- ✅ Blank lines (compressed to max 1)
- ✅ Trailing whitespace

### **Preserved by MR**
- ✅ All code logic (100% semantic equivalence)
- ✅ Indentation (Python requires it)
- ✅ String literals (including multiline)
- ✅ Import order

**Guarantee**: MR compiles and imports successfully. If not, it's a `codeopt` bug.

---

## 🔧 **Tier-Based Strategy**

| Tier | Modules | When to Use |
|------|---------|-------------|
| **Tier 1** (core) | `fused_head.py`, `ptx/*`, `skills/*`, `live_server.py` | Multi-instance servers, edge fog nodes |
| **Tier 2** (trainers) | `phase18/*`, `phase25/*`, `*_trainer.py` | Parallel training runs (4+ workers) |
| **Tier 3** (all) | Entire repo (k3dgen, knowledge3d, viewer) | Production Docker, aggressive optimization |

**Default**: Use **Tier 1** for most cases. Tier 3 is overkill unless memory-constrained.

---

## 🐛 **Troubleshooting**

### **Q: MR import fails**
```python
ImportError: cannot import name 'MortonOctree' from 'knowledge3d.spatial.morton_octree'
```
**A**: MR path not in `PYTHONPATH`. Set it:
```bash
export PYTHONPATH=../Knowledge3D.local/mr:.
```

### **Q: MR code unreadable in stack trace**
```
File "../Knowledge3D.local/mr/knowledge3d/cranium/fused_head.py", line 1523
    pass
```
**A**: Cross-reference HR source. MR line numbers may not match (blank lines removed).

### **Q: MR slower than HR**
```
MR import: 250ms
HR import: 200ms
```
**A**: First import compiles `.py` → `.pyc`. Subsequent imports are faster. Measure after warm-up.

---

## 📝 **Best Practices**

### **1. Always Develop in HR**
```bash
# DEV: Edit HR sources
vim knowledge3d/cranium/fused_head.py

# PROD: Recompile MR
make compile-mr-core
```

### **2. Never Edit MR Directly**
```bash
# ❌ BAD
vim ../Knowledge3D.local/mr/knowledge3d/cranium/fused_head.py

# ✅ GOOD
vim knowledge3d/cranium/fused_head.py
make compile-mr-core
```

### **3. Document MR Usage in Scripts**
```bash
#!/bin/bash
# REQUIRES: make compile-mr-core
export PYTHONPATH=../Knowledge3D.local/mr:.
python -m knowledge3d.bridge.live_server
```

### **4. Test Both HR and MR**
```bash
# HR tests
pytest tests/ -v

# MR tests (with MR in path)
PYTHONPATH=../Knowledge3D.local/mr:. pytest tests/ -v
```

---

## 🔬 **Example: Multi-Instance Training**

### **Without MR**
```bash
# 10 trainers, each loads 257KB fused_head.py
for i in {1..10}; do
  python -m knowledge3d.tools.phase25.rpn_policy_trainer --batch $i &
done
# Total memory: 10 × 257KB = 2.57MB
```

### **With MR**
```bash
# Compile once
make compile-mr-trainers

# 10 trainers, each loads 180KB MR fused_head.py
export PYTHONPATH=../Knowledge3D.local/mr:.
for i in {1..10}; do
  python -m knowledge3d.tools.phase25.rpn_policy_trainer --batch $i &
done
# Total memory: 10 × 180KB = 1.8MB (700KB saved)
```

---

## 📦 **MR Directory Structure**

```
../Knowledge3D.local/mr/
├── knowledge3d/
│   ├── cranium/
│   │   ├── fused_head.py      # MR: no comments/docstrings
│   │   └── ptx/
│   │       └── ptx_ops.py     # MR
│   ├── skills/
│   │   ├── vision.py          # MR
│   │   └── audio.py           # MR
│   └── ...
├── k3dgen/
│   └── ...
└── viewer/  (if compile-mr-all)
    └── ...
```

**Important**: `.gitignore` excludes `../Knowledge3D.local/mr/`. Never commit MR.

---

## 🎓 **Key Takeaways**

1. **MR is for machines** (production, multi-instance, edge)
2. **HR is for humans** (development, debugging, collaboration)
3. **Savings scale with instances** (1 instance ≈ no benefit, 10+ instances ≈ significant)
4. **Tier 1 (core) is sweet spot** for most use cases
5. **Always develop in HR**, recompile MR as needed

---

**Full docs**: [`docs/DUAL_CODE_STRATEGY.md`](DUAL_CODE_STRATEGY.md)
