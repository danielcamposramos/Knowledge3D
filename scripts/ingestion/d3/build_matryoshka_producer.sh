#!/usr/bin/env bash
# Build the standalone sovereign matryoshka_embeddings.bin producer.
#
# Zero Python, zero bulk libraries, zero numpy/cupy/scipy/sympy/torch.
# Links matryoshka_bin_producer.cu with procedural_basis.cu and
# matryoshka_accumulator.cu as separate translation units.
#
# Output: scripts/ingestion/d3/build/matryoshka_bin_producer

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

build_dir="scripts/ingestion/d3/build"
mkdir -p "$build_dir"

nvcc -O3 -std=c++17 \
  -o "$build_dir/matryoshka_bin_producer" \
  scripts/ingestion/d3/matryoshka_bin_producer.cu \
  scripts/ingestion/d3/procedural_basis.cu \
  scripts/ingestion/d3/matryoshka_accumulator.cu

echo "built: $build_dir/matryoshka_bin_producer"
