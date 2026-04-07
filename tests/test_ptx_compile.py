#!/usr/bin/env python3
"""Test PTX compilation"""

def main() -> int:
    import os
    import ctypes

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
    print(f"Context: {ctx}, error={err}")

    err, dev = cuda.cuCtxGetDevice()
    print(f"Device: {dev}, error={err}")

    # Get compute capability
    maj_attr = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    min_attr = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    err, maj = cuda.cuDeviceGetAttribute(maj_attr, dev)
    err2, minu = cuda.cuDeviceGetAttribute(min_attr, dev)
    print(f"Compute capability: {maj}.{minu}")

    # Compile PTX
    res, prog = nvrtc.nvrtcCreateProgram(
        MDCT_KERNEL_SRC.encode("utf-8"), b"mdct.cu", 0, [], []
    )
    print(f"nvrtcCreateProgram: {res}")

    arch = f"--gpu-architecture=compute_{maj}{minu}".encode("utf-8")
    opts = [arch, b"--fmad=false"]
    print(f"Compile options: {[o.decode() for o in opts]}")

    (res,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    print(f"nvrtcCompileProgram: {res}")

    if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        log_size_res, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        if log_size_res == nvrtc.nvrtcResult.NVRTC_SUCCESS and log_size > 1:
            buf = bytearray(log_size)
            nvrtc.nvrtcGetProgramLog(prog, buf)
            print(f"Compilation log:\n{buf.decode('utf-8', errors='replace')}")

    res, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
    print(f"PTX size: {ptx_size} bytes")

    buf = bytearray(ptx_size)
    (res,) = nvrtc.nvrtcGetPTX(prog, buf)
    nvrtc.nvrtcDestroyProgram(prog)

    print(f"First 500 chars of PTX:\n{buf[:500].decode('utf-8', errors='replace')}")

    print("\nTrying to load PTX into module...")
    err, module = cuda.cuModuleLoadData(bytes(buf))
    print(f"cuModuleLoadData: error={err}")

    if err != cuda.CUresult.CUDA_SUCCESS:
        # Try to get error name
        err_str = ctypes.c_char_p()
        if cuda.cuGetErrorString(err, ctypes.byref(err_str)) == 0:
            print(f"Error: {err_str.value.decode()}")
        return 1

    print("✓ PTX loaded successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
