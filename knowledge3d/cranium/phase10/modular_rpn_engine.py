"""GPU-resident modular RPN engine using NVRTC-compiled CUDA kernels.

This engine keeps all computation inside the cranium: no CPU fallbacks are
provided. If a CUDA device or the cuda-python bindings are unavailable, an
exception is raised at construction time.
"""
from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np  # type: ignore

try:
    from cuda import cuda, nvrtc  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "cuda-python bindings are required for ModularRPNEngine; install `cuda-python` and ensure a CUDA device is available"
    ) from exc

CUDA_SOURCE = r"""
extern "C" {

struct RPNValue {
    float4 data;
};

struct RPNInstance {
    float4 stack[64];
    int head;
    int size;
    int error;
    int reserved;
};

__device__ inline void push(RPNInstance& inst, const float4& v) {
    const int mask = 63; // 64-1
    const int max_size = 64;
    int idx = (inst.head + inst.size) & mask;
    inst.stack[idx] = v;
    if (inst.size < max_size) {
        inst.size += 1;
    } else {
        inst.head = (inst.head + 1) & mask;
    }
}

__device__ inline bool pop(RPNInstance& inst, float4& out) {
    if (inst.size <= 0) {
        return false;
    }
    const int mask = 63;
    inst.size -= 1;
    int idx = (inst.head + inst.size) & mask;
    out = inst.stack[idx];
    if (inst.size == 0) {
        inst.head = 0;
    }
    return true;
}

__device__ inline float clamp_min(float value, float eps) {
    return (value < eps) ? eps : value;
}

__global__ void modular_rpn_geometric_kernel(
    int instance_id,
    const unsigned short* op_codes,
    const float* scalars,
    const float3* vectors,
    RPNInstance* instances,
    int token_count
) {
    const int MAX_INSTANCES = 15;
    if (instance_id < 0) instance_id = 0;
    if (instance_id >= MAX_INSTANCES) instance_id = MAX_INSTANCES - 1;

    RPNInstance& inst = instances[instance_id];
    for (int i = 0; i < token_count; ++i) {
        unsigned short opcode = op_codes[i];
        const float scalar = scalars[i];
        const float3 vec = vectors[i];
        float4 a, b, c;
        switch (opcode) {
            case 0: { // scalar literal
                float4 v = make_float4(scalar, 0.f, 0.f, 0.f);
                push(inst, v);
                break;
            }
            case 1: { // vector literal
                float4 v = make_float4(vec.x, vec.y, vec.z, 0.f);
                push(inst, v);
                break;
            }
            case 10: { // add
                if (!pop(inst, b) || !pop(inst, a)) { inst.error = 1002; return; }
                float4 out = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
                push(inst, out);
                break;
            }
            case 11: { // sub
                if (!pop(inst, b) || !pop(inst, a)) { inst.error = 1002; return; }
                float4 out = make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
                push(inst, out);
                break;
            }
            case 12: { // mul
                if (!pop(inst, b) || !pop(inst, a)) { inst.error = 1002; return; }
                float4 out = make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
                push(inst, out);
                break;
            }
            case 13: { // div
                if (!pop(inst, b) || !pop(inst, a)) { inst.error = 1002; return; }
                if (b.x == 0.f || b.y == 0.f || b.z == 0.f) { inst.error = 1003; return; }
                float4 out = make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
                push(inst, out);
                break;
            }
            case 14: { // pow (scalar only)
                if (!pop(inst, b) || !pop(inst, a)) { inst.error = 1002; return; }
                float base = clamp_min(fabsf(a.x), 1e-6f);
                float val = __exp2f(b.x * __log2f(base));
                push(inst, make_float4(val, 0.f, 0.f, 0.f));
                break;
            }
            case 15: { // neg
                if (!pop(inst, a)) { inst.error = 1002; return; }
                push(inst, make_float4(-a.x, -a.y, -a.z, -a.w));
                break;
            }
            case 20: { // sqrt
                if (!pop(inst, a)) { inst.error = 1002; return; }
                float val = sqrtf(fmaxf(a.x, 0.f));
                push(inst, make_float4(val, 0.f, 0.f, 0.f));
                break;
            }
            case 21: { // exp
                if (!pop(inst, a)) { inst.error = 1002; return; }
                float val = __expf(a.x);
                push(inst, make_float4(val, 0.f, 0.f, 0.f));
                break;
            }
            case 22: { // log
                if (!pop(inst, a)) { inst.error = 1002; return; }
                float val = __logf(clamp_min(a.x, 1e-6f));
                push(inst, make_float4(val, 0.f, 0.f, 0.f));
                break;
            }
            case 24: { // sin
                if (!pop(inst, a)) { inst.error = 1002; return; }
                push(inst, make_float4(__sinf(a.x), 0.f, 0.f, 0.f));
                break;
            }
            case 25: { // cos
                if (!pop(inst, a)) { inst.error = 1002; return; }
                push(inst, make_float4(__cosf(a.x), 0.f, 0.f, 0.f));
                break;
            }
            case 26: { // tan
                if (!pop(inst, a)) { inst.error = 1002; return; }
                push(inst, make_float4(__tanf(a.x), 0.f, 0.f, 0.f));
                break;
            }
            case 40: { // gt
                if (!pop(inst, b) || !pop(inst, a)) { inst.error = 1002; return; }
                float val = (a.x > b.x) ? 1.f : 0.f;
                push(inst, make_float4(val, 0.f, 0.f, 0.f));
                break;
            }
            case 42: { // lt
                if (!pop(inst, b) || !pop(inst, a)) { inst.error = 1002; return; }
                float val = (a.x < b.x) ? 1.f : 0.f;
                push(inst, make_float4(val, 0.f, 0.f, 0.f));
                break;
            }
            case 44: { // eq
                if (!pop(inst, b) || !pop(inst, a)) { inst.error = 1002; return; }
                float val = (a.x == b.x) ? 1.f : 0.f;
                push(inst, make_float4(val, 0.f, 0.f, 0.f));
                break;
            }
            case 46: { // max
                if (!pop(inst, b) || !pop(inst, a)) { inst.error = 1002; return; }
                push(inst, make_float4(fmaxf(a.x, b.x), 0.f, 0.f, 0.f));
                break;
            }
            case 47: { // min
                if (!pop(inst, b) || !pop(inst, a)) { inst.error = 1002; return; }
                push(inst, make_float4(fminf(a.x, b.x), 0.f, 0.f, 0.f));
                break;
            }
            case 50: { // dup
                if (!pop(inst, a)) { inst.error = 1002; return; }
                push(inst, a);
                push(inst, a);
                break;
            }
            case 51: { // swap
                if (!pop(inst, a) || !pop(inst, b)) { inst.error = 1002; return; }
                push(inst, a);
                push(inst, b);
                break;
            }
            case 52: { // drop
                if (!pop(inst, a)) { inst.error = 1002; return; }
                break;
            }
            case 53: { // over
                if (!pop(inst, a) || !pop(inst, b)) { inst.error = 1002; return; }
                push(inst, b);
                push(inst, a);
                push(inst, b);
                break;
            }
            case 55: { // clear
                inst.head = 0;
                inst.size = 0;
                break;
            }
            case 60: { // dot
                if (!pop(inst, a) || !pop(inst, b)) { inst.error = 1002; return; }
                float val = b.x * a.x + b.y * a.y + b.z * a.z;
                push(inst, make_float4(val, 0.f, 0.f, 0.f));
                break;
            }
            case 61: { // cross
                if (!pop(inst, a) || !pop(inst, b)) { inst.error = 1002; return; }
                float4 out;
                out.x = b.y * a.z - b.z * a.y;
                out.y = b.z * a.x - b.x * a.z;
                out.z = b.x * a.y - b.y * a.x;
                out.w = 0.f;
                push(inst, out);
                break;
            }
            case 62: { // magnitude
                if (!pop(inst, a)) { inst.error = 1002; return; }
                float val = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
                push(inst, make_float4(val, 0.f, 0.f, 0.f));
                break;
            }
            case 63: { // normalize
                if (!pop(inst, a)) { inst.error = 1002; return; }
                float mag = clamp_min(sqrtf(a.x * a.x + a.y * a.y + a.z * a.z), 1e-6f);
                push(inst, make_float4(a.x / mag, a.y / mag, a.z / mag, 0.f));
                break;
            }
            case 70: { // rotate (axis on top, then angle, then vector)
                float4 axis;
                if (!pop(inst, axis)) { inst.error = 1002; return; }
                float4 angle;
                if (!pop(inst, angle)) { inst.error = 1002; return; }
                float4 vec_val;
                if (!pop(inst, vec_val)) { inst.error = 1002; return; }
                float ax = axis.x, ay = axis.y, az = axis.z;
                float norm = clamp_min(sqrtf(ax*ax + ay*ay + az*az), 1e-6f);
                ax /= norm; ay /= norm; az /= norm;
                float s = sinf(angle.x);
                float c = cosf(angle.x);
                float dot = ax*vec_val.x + ay*vec_val.y + az*vec_val.z;
                float4 cross;
                cross.x = ay * vec_val.z - az * vec_val.y;
                cross.y = az * vec_val.x - ax * vec_val.z;
                cross.z = ax * vec_val.y - ay * vec_val.x;
                float4 out;
                out.x = vec_val.x * c + cross.x * s + ax * dot * (1.f - c);
                out.y = vec_val.y * c + cross.y * s + ay * dot * (1.f - c);
                out.z = vec_val.z * c + cross.z * s + az * dot * (1.f - c);
                out.w = 0.f;
                push(inst, out);
                break;
            }
            case 71: { // scale vector by scalar literal (scalar on top)
                if (!pop(inst, a)) { inst.error = 1002; return; }
                float factor = a.x;
                if (!pop(inst, b)) { inst.error = 1002; return; }
                push(inst, make_float4(b.x * factor, b.y * factor, b.z * factor, 0.f));
                break;
            }
            case 72: { // translate vector by vector
                if (!pop(inst, a) || !pop(inst, b)) { inst.error = 1002; return; }
                push(inst, make_float4(b.x + a.x, b.y + a.y, b.z + a.z, 0.f));
                break;
            }
            case 80: { // ifelse (false, true, condition)
                float4 false_v, true_v, cond_v;
                if (!pop(inst, false_v) || !pop(inst, true_v) || !pop(inst, cond_v)) { inst.error = 1002; return; }
                float4 out = cond_v.x != 0.f ? true_v : false_v;
                push(inst, out);
                break;
            }
            default:
                inst.error = 9001;
                return;
        }
    }
}

} // extern "C"
"""


class ModularRPNEngine:
    """Compile-once GPU RPN engine."""

    _INSTANCE_COUNT = 15
    _STACK_MAX = 64

    OP_LITERAL = 0
    OP_LITERAL_VEC = 1

    OPCODES: Dict[str, int] = {
        "+": 10,
        "-": 11,
        "*": 12,
        "/": 13,
        "^": 14,
        "neg": 15,
        "sqrt": 20,
        "exp": 21,
        "log": 22,
        "sin": 24,
        "cos": 25,
        "tan": 26,
        "gt": 40,
        "lt": 42,
        "eq": 44,
        "max": 46,
        "min": 47,
        "dup": 50,
        "swap": 51,
        "drop": 52,
        "over": 53,
        "clear": 55,
        "dot": 60,
        "cross": 61,
        "mag": 62,
        "norm": 63,
        "rotate": 70,
        "scale": 71,
        "translate": 72,
        "ifelse": 80,
    }

    CONSTANTS: Dict[str, float] = {
        "pi": math.pi,
        "π": math.pi,
        "tau": math.tau,
        "phi": (1.0 + math.sqrt(5.0)) / 2.0,
        "φ": (1.0 + math.sqrt(5.0)) / 2.0,
        "e": math.e,
    }

    def __init__(self, max_instances: int = _INSTANCE_COUNT) -> None:
        if max_instances > self._INSTANCE_COUNT:
            raise ValueError(f"Maximum supported instances is {self._INSTANCE_COUNT}")
        self.max_instances = max_instances
        self._compile_kernel()
        self._allocate_state()

    # -------------------------- CUDA setup --------------------------
    def _compile_kernel(self) -> None:
        prog = nvrtc.nvrtcCreateProgram(
            CUDA_SOURCE.encode("utf-8"),
            b"modular_rpn.cu",
            0,
            [],
            []
        )
        opts = [b"--gpu-architecture=compute_70", b"--fmad=false"]
        res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if res != nvrtc.NVRTC_SUCCESS:
            log = nvrtc.nvrtcGetProgramLog(prog)[1].decode("utf-8")
            raise RuntimeError(f"NVRTC compilation failed:\n{log}")
        ptx = nvrtc.nvrtcGetPTX(prog)[1]
        nvrtc.nvrtcDestroyProgram(prog)

        err, = cuda.cuInit(0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuInit failed: {err}")
        err, dev = cuda.cuDeviceGet(0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDeviceGet failed: {err}")
        err, ctx = cuda.cuCtxCreate(0, dev)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuCtxCreate failed: {err}")
        self._ctx = ctx
        err, module = cuda.cuModuleLoadData(ptx)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleLoadData failed: {err}")
        self._module = module
        err, func = cuda.cuModuleGetFunction(module, b"modular_rpn_geometric_kernel")
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleGetFunction failed: {err}")
        self._kernel = func

    def _allocate_state(self) -> None:
        class RPNInstance(ctypes.Structure):
            _fields_ = [
                ("stack", ctypes.c_float * (self._STACK_MAX * 4)),
                ("head", ctypes.c_int),
                ("size", ctypes.c_int),
                ("error", ctypes.c_int),
                ("reserved", ctypes.c_int),
            ]
        self._instance_struct = RPNInstance
        self._instance_size = ctypes.sizeof(RPNInstance)
        total_bytes = self._instance_size * self.max_instances
        err, d_states = cuda.cuMemAlloc(total_bytes)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemAlloc for RPN instance state failed: {err}")
        cuda.cuMemsetD8(d_states, 0, total_bytes)
        self._d_states = d_states

    # -------------------------- Token handling --------------------------
    def _to_float(self, token: str, variables: Optional[Dict[str, float]]) -> float:
        if token in self.CONSTANTS:
            return self.CONSTANTS[token]
        if variables and token in variables:
            return float(variables[token])
        try:
            return float(token)
        except ValueError as exc:
            raise ValueError(f"Unknown literal '{token}'") from exc

    def _parse_vector(self, token: str) -> Optional[List[float]]:
        if not token.startswith("[") or not token.endswith("]"):
            return None
        body = token[1:-1]
        parts = [p.strip() for p in body.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(f"Vector literal must have 3 components: {token}")
        return [float(parts[0]), float(parts[1]), float(parts[2])]

    def tokenize_rpn(self, expression: str) -> List[str]:
        tokens: List[str] = []
        for raw in expression.strip().split():
            if raw == "->":  # allow arrows in legacy expressions, skip
                continue
            tokens.append(raw)
        return tokens

    def compile_tokens(
        self,
        tokens: Sequence[str],
        variables: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        op_codes: List[int] = []
        scalars: List[float] = []
        vectors: List[List[float]] = []
        for tok in tokens:
            vector_literal = self._parse_vector(tok)
            if vector_literal is not None:
                op_codes.append(self.OP_LITERAL_VEC)
                scalars.append(0.0)
                vectors.append(vector_literal)
                continue
            opcode = self.OPCODES.get(tok)
            if opcode is not None:
                op_codes.append(opcode)
                scalars.append(0.0)
                vectors.append([0.0, 0.0, 0.0])
                continue
            value = self._to_float(tok, variables)
            op_codes.append(self.OP_LITERAL)
            scalars.append(value)
            vectors.append([0.0, 0.0, 0.0])
        op_arr = np.asarray(op_codes, dtype=np.uint16)
        scalars_arr = np.asarray(scalars, dtype=np.float32)
        vectors_arr = np.asarray(vectors, dtype=np.float32)
        return op_arr, scalars_arr, vectors_arr

    # -------------------------- Evaluation --------------------------
    def evaluate(
        self,
        expression: str,
        instance_id: int = 0,
        variables: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        tokens = self.tokenize_rpn(expression)
        if not tokens:
            raise ValueError("Empty expression")
        op_codes, scalars, vectors = self.compile_tokens(tokens, variables)
        token_count = op_codes.shape[0]

        # Allocate device buffers
        err, d_ops = cuda.cuMemAlloc(op_codes.nbytes)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemAlloc op codes failed: {err}")
        err, d_scalars = cuda.cuMemAlloc(scalars.nbytes)
        if err != cuda.CUresult.CUDA_SUCCESS:
            cuda.cuMemFree(d_ops)
            raise RuntimeError(f"cuMemAlloc scalars failed: {err}")
        err, d_vectors = cuda.cuMemAlloc(vectors.nbytes)
        if err != cuda.CUresult.CUDA_SUCCESS:
            cuda.cuMemFree(d_ops)
            cuda.cuMemFree(d_scalars)
            raise RuntimeError(f"cuMemAlloc vectors failed: {err}")

        try:
            cuda.cuMemcpyHtoD(d_ops, op_codes.ctypes.data, op_codes.nbytes)
            cuda.cuMemcpyHtoD(d_scalars, scalars.ctypes.data, scalars.nbytes)
            cuda.cuMemcpyHtoD(d_vectors, vectors.ctypes.data, vectors.nbytes)

            arg_instance = ctypes.c_int(int(instance_id))
            arg_ops = ctypes.c_void_p(int(d_ops))
            arg_scalars = ctypes.c_void_p(int(d_scalars))
            arg_vectors = ctypes.c_void_p(int(d_vectors))
            arg_states = ctypes.c_void_p(int(self._d_states))
            arg_token_count = ctypes.c_int(int(token_count))

            args = (ctypes.c_void_p * 6)(
                ctypes.cast(ctypes.pointer(arg_instance), ctypes.c_void_p),
                ctypes.cast(ctypes.pointer(arg_ops), ctypes.c_void_p),
                ctypes.cast(ctypes.pointer(arg_scalars), ctypes.c_void_p),
                ctypes.cast(ctypes.pointer(arg_vectors), ctypes.c_void_p),
                ctypes.cast(ctypes.pointer(arg_states), ctypes.c_void_p),
                ctypes.cast(ctypes.pointer(arg_token_count), ctypes.c_void_p),
            )

            err, = cuda.cuLaunchKernel(
                self._kernel,
                1, 1, 1,
                1, 1, 1,
                0,
                0,
                args,
                0,
            )
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuLaunchKernel failed: {err}")
            cuda.cuCtxSynchronize()

            # Copy the instance state back to host
            instance_struct = self._instance_struct()
            offset = instance_id * self._instance_size
            cuda.cuMemcpyDtoH(ctypes.addressof(instance_struct), self._d_states + offset, self._instance_size)

            if instance_struct.error != 0:
                raise RuntimeError(f"RPN engine error code {instance_struct.error}")
            if instance_struct.size <= 0:
                raise RuntimeError("RPN stack empty after evaluation")
            top_index = (instance_struct.head + instance_struct.size - 1) & 63
            base = top_index * 4
            result = np.array(
                [
                    instance_struct.stack[base + 0],
                    instance_struct.stack[base + 1],
                    instance_struct.stack[base + 2],
                    instance_struct.stack[base + 3],
                ],
                dtype=np.float32,
            )
            return result
        finally:
            cuda.cuMemFree(d_ops)
            cuda.cuMemFree(d_scalars)
            cuda.cuMemFree(d_vectors)

    def reset(self) -> None:
        total_bytes = self._instance_size * self.max_instances
        cuda.cuMemsetD8(self._d_states, 0, total_bytes)

    def close(self) -> None:
        if hasattr(self, "_d_states"):
            cuda.cuMemFree(self._d_states)
            del self._d_states
        if hasattr(self, "_module"):
            cuda.cuModuleUnload(self._module)
            del self._module
        if hasattr(self, "_ctx"):
            cuda.cuCtxDestroy(self._ctx)
            del self._ctx

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


__all__ = ["ModularRPNEngine"]
