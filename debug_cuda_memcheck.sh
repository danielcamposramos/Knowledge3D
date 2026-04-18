#!/bin/bash
# Run kernel with CUDA memcheck to find exact error location
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
conda run -n k3d-cranium compute-sanitizer --tool memcheck python debug_kernel_exec.py
