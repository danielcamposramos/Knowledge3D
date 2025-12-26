#!/usr/bin/env python3
"""Test with simplest possible PTX kernel"""

def main() -> int:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from knowledge3d.cranium.codecs.ptx_bindings.ternary_mdct_binding import _load_cuda

    # Simplest possible CUDA kernel
    simple_kernel = r"""
extern "C" __global__ void add_one(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + 1.0f;
    }
}
"""

    cuda, nvrtc = _load_cuda()

    print("Initializing CUDA...")
    cuda.cuInit(0)
    err, dev = cuda.cuDeviceGet(0)
    err, ctx = cuda.cuCtxCreate(0, dev)
    print(f"Context created: {ctx}")

    err, maj = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev
    )
    err, minu = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev
    )
    print(f"Compute capability: {maj}.{minu}")

    print("\nCompiling simple kernel...")
    res, prog = nvrtc.nvrtcCreateProgram(simple_kernel.encode("utf-8"), b"simple.cu", 0, [], [])

    # Try different compile options
    options_to_try = [
        [f"--gpu-architecture=compute_{maj}{minu}".encode()],
        [b"--gpu-architecture=sm_86"],
        [b"--gpu-architecture=compute_80"],  # Lower arch
    ]

    for i, opts in enumerate(options_to_try):
        print(f"\nAttempt {i+1}: {[o.decode() for o in opts]}")

        res, prog = nvrtc.nvrtcCreateProgram(
            simple_kernel.encode("utf-8"), b"simple.cu", 0, [], []
        )
        (res,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)

        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            print(f"  Compilation failed: {res}")
            continue

        res, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
        buf = bytearray(ptx_size)
        (res,) = nvrtc.nvrtcGetPTX(prog, buf)
        nvrtc.nvrtcDestroyProgram(prog)

        print(f"  PTX first line: {buf[:100].decode('utf-8', errors='replace')}")

        err, module = cuda.cuModuleLoadData(bytes(buf))
        print(f"  cuModuleLoadData: error={err}")

        if err == cuda.CUresult.CUDA_SUCCESS:
            print(f"  ✓ SUCCESS! Module loaded with options: {[o.decode() for o in opts]}")
            return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
