#!/bin/bash
# Prepare evidence files for K3D scientific paper
# Run from repository root: bash scripts/prepare_paper_evidence.sh

set -e  # Exit on error

echo "=== Preparing K3D Paper Evidence ==="
echo ""

# Create evidence directory if needed
mkdir -p docs/paper-evidence/rdf_exports

# 1. Copy benchmark results
echo "[1/6] Copying benchmark results..."
if [ -f "/K3D/Knowledge3D.local/results/week21_3_architecture_fixed_full/math_competitions_enriched.json" ]; then
    cp "/K3D/Knowledge3D.local/results/week21_3_architecture_fixed_full/math_competitions_enriched.json" \
       "docs/paper-evidence/math_competitions_week21_3.json"
    echo "  ✓ Math competitions results copied"
else
    echo "  ⚠ Math competitions results not found"
fi

if [ -f "/K3D/Knowledge3D.local/results/week21_3_architecture_fixed_full/last_humanity_exam_enriched.json" ]; then
    cp "/K3D/Knowledge3D.local/results/week21_3_architecture_fixed_full/last_humanity_exam_enriched.json" \
       "docs/paper-evidence/last_humanity_exam_week21_3.json"
    echo "  ✓ Last Humanity Exam results copied"
else
    echo "  ⚠ Last Humanity Exam results not found"
fi

# 2. Copy PTX profiling report
echo ""
echo "[2/6] Copying PTX profiling report..."
if [ -f "TEMP/CODEX_WEEK21_8_PTX_PROFILE_FINDINGS_02.10.2026.md" ]; then
    cp "TEMP/CODEX_WEEK21_8_PTX_PROFILE_FINDINGS_02.10.2026.md" \
       "docs/paper-evidence/ptx_profiling_week21_8.md"
    echo "  ✓ PTX profiling report copied"
else
    echo "  ⚠ PTX profiling report not found"
fi

# 3. Create galaxy samples
echo ""
echo "[3/6] Creating galaxy samples..."
if [ -f "/K3D/Knowledge3D.local/galaxies/Drawing.jsonl" ]; then
    head -100 /K3D/Knowledge3D.local/galaxies/Drawing.jsonl > \
        docs/paper-evidence/drawing_galaxy_sample.jsonl
    echo "  ✓ Drawing Galaxy sample (100 entries)"
else
    echo "  ⚠ Drawing Galaxy not found"
fi

if [ -f "/K3D/Knowledge3D.local/galaxies/Grammar.jsonl" ]; then
    head -100 /K3D/Knowledge3D.local/galaxies/Grammar.jsonl > \
        docs/paper-evidence/grammar_galaxy_sample.jsonl
    echo "  ✓ Grammar Galaxy sample (100 entries)"
else
    echo "  ⚠ Grammar Galaxy not found"
fi

if [ -f "/K3D/Knowledge3D.local/galaxies/Math.jsonl" ]; then
    head -50 /K3D/Knowledge3D.local/galaxies/Math.jsonl > \
        docs/paper-evidence/math_galaxy_sample.jsonl
    echo "  ✓ Math Galaxy sample (50 entries)"
else
    echo "  ⚠ Math Galaxy not yet populated (expected for current phase)"
fi

# 4. Check for test logs (MVP Phase 1)
echo ""
echo "[4/6] Checking for test logs..."
TEST_LOGS=$(find /K3D/Knowledge3D.local -name "*test*" -type f 2>/dev/null | grep -E "mvp|28.*test" | head -5 || true)
if [ -n "$TEST_LOGS" ]; then
    echo "  ✓ Found test logs:"
    echo "$TEST_LOGS" | while read log; do
        echo "    - $log"
    done
    echo "  → Consider copying relevant logs to docs/paper-evidence/"
else
    echo "  ⚠ No MVP test logs found (marked as HISTORICAL in evidence map)"
fi

# 5. Check for GPU logs
echo ""
echo "[5/6] Checking for GPU utilization logs..."
GPU_LOGS=$(find /K3D/Knowledge3D.local -name "*gpu*" -o -name "*nvidia*" 2>/dev/null | head -5 || true)
if [ -n "$GPU_LOGS" ]; then
    echo "  ✓ Found GPU logs:"
    echo "$GPU_LOGS" | while read log; do
        echo "    - $log"
    done
    echo "  → Consider copying to docs/paper-evidence/"
else
    echo "  ⚠ No GPU logs found (can generate during next benchmark run)"
fi

# 6. Summary
echo ""
echo "[6/6] Summary of evidence files:"
echo ""
ls -lh docs/paper-evidence/*.json 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
ls -lh docs/paper-evidence/*.jsonl 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
ls -lh docs/paper-evidence/*.md 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'

echo ""
echo "=== Done! ==="
echo ""
echo "Next steps:"
echo "  1. Review docs/paper-evidence/FILES_NEEDED_FOR_ARTICLE.md"
echo "  2. Optionally run: python scripts/export_knowledgeverse_to_rdf.py"
echo "  3. Optionally run: python scripts/validate_knowledgeverse_schema.py"
echo "  4. Commit evidence files to repo for ChatGPT access"
echo ""
