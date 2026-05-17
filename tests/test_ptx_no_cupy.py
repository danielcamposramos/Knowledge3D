#!/usr/bin/env python3
"""Test PTX loading without CuPy interference"""

def main() -> int:
    import os
    import ctypes

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from knowledge3d.cranium.codecs.ptx_bindings.ternary_mdct_binding import (
        MDCT_KERNEL_SRC,
        _load_cuda,
    )
    from knowledge3d.cranium.sovereign import loader

    cuda, nvrtc = _load_cuda()

    print("Step 1: Ensure CUDA context via sovereign loader")
    loader.ensure_init()
    err, ctx = cuda.cuCtxGetCurrent()
    print(f"  cuCtxGetCurrent: {err}, ctx={ctx}")
    if err != cuda.CUresult.CUDA_SUCCESS or ctx is None or int(ctx) == 0:
        raise RuntimeError("No CUDA context after loader.ensure_init()")

    print("\nStep 2: Get device")
    err, dev = cuda.cuDeviceGet(0)
    print(f"  cuDeviceGet: {err}, dev={dev}")

    print("\nStep 3: Get compute capability")
    maj_attr = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    min_attr = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    err, maj = cuda.cuDeviceGetAttribute(maj_attr, dev)
    err2, minu = cuda.cuDeviceGetAttribute(min_attr, dev)
    print(f"  Compute capability: {maj}.{minu}")

    print("\nStep 4: Compile PTX")
    res, prog = nvrtc.nvrtcCreateProgram(
        MDCT_KERNEL_SRC.encode("utf-8"), b"mdct.cu", 0, [], []
    )
    print(f"  nvrtcCreateProgram: {res}")

    arch = f"--gpu-architecture=compute_{maj}{minu}".encode("utf-8")
    opts = [arch, b"--fmad=false"]
    print(f"  Compile options: {[o.decode() for o in opts]}")

    (res,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    print(f"  nvrtcCompileProgram: {res}")

    if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        log_size_res, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        if log_size_res == nvrtc.nvrtcResult.NVRTC_SUCCESS and log_size > 1:
            buf = bytearray(log_size)
            nvrtc.nvrtcGetProgramLog(prog, buf)
            print(f"  Compilation log:\n{buf.decode('utf-8', errors='replace')}")
        raise RuntimeError("Compilation failed")

    res, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
    print(f"  PTX size: {ptx_size} bytes")

    buf = bytearray(ptx_size)
    (res,) = nvrtc.nvrtcGetPTX(prog, buf)
    nvrtc.nvrtcDestroyProgram(prog)

    print("\nStep 5: Load PTX into module")
    err, module = cuda.cuModuleLoadData(bytes(buf))
    print(f"  cuModuleLoadData: error={err}")

    if err != cuda.CUresult.CUDA_SUCCESS:
        # Get error name
        err_str = ctypes.c_char_p()
        if cuda.cuGetErrorString(err, ctypes.byref(err_str)) == 0:
            print(f"  Error string: {err_str.value.decode()}")
        else:
            print(f"  Error code: {err}")
        return 1

    print("  ✓ PTX loaded successfully!")

    print("\nStep 6: Get function")
    err, func = cuda.cuModuleGetFunction(module, b"ternary_mdct_forward")
    print(f"  cuModuleGetFunction: error={err}")
    if err == cuda.CUresult.CUDA_SUCCESS:
        print(f"  ✓ Function retrieved: {func}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
