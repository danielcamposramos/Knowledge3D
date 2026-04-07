#!/usr/bin/env python3
"""Check PTX version being generated"""

def main() -> int:
    import os
    import subprocess

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from knowledge3d.cranium.sovereign import loader

    loader._ensure_init()

    from knowledge3d.cranium.codecs.ptx_bindings.ternary_mdct_binding import (
        MDCT_KERNEL_SRC,
        _load_cuda,
    )

    cuda, nvrtc = _load_cuda()

    # Get device
    err, ctx = cuda.cuCtxGetCurrent()
    err, dev = cuda.cuCtxGetDevice()

    # Get compute capability
    maj_attr = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    min_attr = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    err, maj = cuda.cuDeviceGetAttribute(maj_attr, dev)
    err2, minu = cuda.cuDeviceGetAttribute(min_attr, dev)
    print(f"GPU Compute Capability: {maj}.{minu} (sm_{maj}{minu})")
    print()

    # Compile PTX
    res, prog = nvrtc.nvrtcCreateProgram(
        MDCT_KERNEL_SRC.encode("utf-8"), b"mdct.cu", 0, [], []
    )

    arch = f"--gpu-architecture=compute_{maj}{minu}".encode("utf-8")
    opts = [arch, b"--fmad=false"]
    print(f"Compile options: {[o.decode() for o in opts]}")

    (res,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)

    res, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
    buf = bytearray(ptx_size)
    (res,) = nvrtc.nvrtcGetPTX(prog, buf)

    # Parse PTX header
    ptx_str = buf.decode("utf-8")
    lines = ptx_str.split("\n")
    print("\nPTX Header:")
    for line in lines[:20]:
        print(f"  {line}")
        if ".target" in line:
            print(f"  ^^^^ Target architecture: {line.strip()}")
        if ".version" in line:
            print(f"  ^^^^ PTX version: {line.strip()}")

    # Check driver capabilities
    print("\nDriver info:")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version,compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"  {result.stdout.strip()}")

    # Try loading with explicit JIT target
    print("\nAttempting cuModuleLoadData...")
    err, module = cuda.cuModuleLoadData(bytes(buf))
    print(f"  Result: error={err}")

    if err != 0:
        print("\n  Error 222 = CUDA_ERROR_ILLEGAL_INSTRUCTION")
        print("  This typically means:")
        print("    - PTX contains instructions not supported by the driver")
        print("    - PTX version (.version 8.7) > Driver's max supported PTX version")
        print("    - Target architecture (.target sm_86) not fully supported")
        nvrtc.nvrtcDestroyProgram(prog)
        return 1

    nvrtc.nvrtcDestroyProgram(prog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
