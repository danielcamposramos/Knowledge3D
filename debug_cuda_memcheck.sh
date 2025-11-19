#!/bin/bash
# Run kernel with CUDA memcheck to find exact error location
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
conda run -n k3d-cranium compute-sanitizer --tool memcheck python debug_kernel_exec.py
