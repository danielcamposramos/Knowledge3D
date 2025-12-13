#!/bin/bash
# cleanup_arc_agi.sh — Organize ARC-AGI folder

set -euo pipefail

ARC_DIR="knowledge3d/training/arc_agi"
OLD_DIR="$ARC_DIR/Old_Attempts"
LOGS_DIR="$ARC_DIR/logs"

mkdir -p "$OLD_DIR" "$LOGS_DIR"

# Canonical files to keep in arc_agi root
CANONICAL=(
    "__init__.py"
    "candidate_generator.py"
    "drawing_galaxy.py"
    "grammar_galaxy.py"
    "multimodal_parser.py"
    "parallel_candidate_generator.py"
    "progressive_scorer.py"
    "rpn_executor.py"
    "size_pattern_encoder.py"
    "sovereign_pipeline.py"
    "specialist_registry.py"
    "trm_swarm_coordinator.py"
    "unified_dataset_loader.py"
    "session_reporter.py"
    "compositional_generator.py"
    "dual_shadow_copy.py"
    "embedders"
    "grammar_drawing"
    "grammar_executor.py"
    "grammar_languages"
    "grammar_math"
    "grammar_normalizer.py"
    "grid_processor.py"
    "hybrid_generator.py"
    "language_grammar_rules.py"
    "math_grammar_rules.py"
    "pattern_quality.py"
    "program_composer.py"
    "quality_scorer.py"
    "semantic_compiler.py"
    "semantic_context.py"
    "semantic_parser.py"
    "semantic_primitives.py"
    "semantic_signature.py"
    "sequential_refiner.py"
    "sleeptime_consolidator.py"
    "sovereign_trm_router.py"
    "sovereign_utils.py"
    "dual_shadow_copy.py"
    "parallel_generator.py"
)

is_canonical() {
    local name="$1"
    for c in "${CANONICAL[@]}"; do
        if [[ "$name" == "$c" ]]; then
            return 0
        fi
    done
    return 1
}

shopt -s nullglob
for f in "$ARC_DIR"/*.py; do
    base="$(basename "$f")"
    if ! is_canonical "$base"; then
        echo "Moving $base to Old_Attempts/"
        mv "$f" "$OLD_DIR/"
    fi
done

echo "ARC-AGI folder cleanup complete."
