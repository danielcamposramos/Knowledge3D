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
    from cuda.bindings import driver as cuda  # type: ignore
    from cuda.bindings import nvrtc  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "cuda-python bindings are required for ModularRPNEngine; install `cuda-python` and ensure a CUDA device is available"
    ) from exc

CUDA_SOURCE = r"""
extern "C" __device__ float sinf(float);
extern "C" __device__ float cosf(float);
extern "C" __device__ float tanf(float);
extern "C" __device__ float asinf(float);
extern "C" __device__ float acosf(float);
extern "C" __device__ float atanf(float);
extern "C" __device__ float sinhf(float);
extern "C" __device__ float coshf(float);
extern "C" __device__ float tanhf(float);
extern "C" __device__ float expf(float);
extern "C" __device__ float logf(float);
extern "C" __device__ float log10f(float);
extern "C" __device__ float powf(float, float);
extern "C" __device__ float fabsf(float);

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
            case 13: { // div (scalar focus)
                if (!pop(inst, b) || !pop(inst, a)) { inst.error = 1002; return; }
                if (b.x == 0.f) { inst.error = 1003; return; }
                float4 out = make_float4(a.x / b.x, 0.f, 0.f, 0.f);
                push(inst, out);
                break;
            }
            case 14: { // pow (scalar only)
                if (!pop(inst, b) || !pop(inst, a)) { inst.error = 1002; return; }
                float base = clamp_min(fabsf(a.x), 1e-6f);
                float val = powf(base, b.x);
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
                float val = expf(a.x);
                push(inst, make_float4(val, 0.f, 0.f, 0.f));
                break;
            }
            case 22: { // log
                if (!pop(inst, a)) { inst.error = 1002; return; }
                float val = logf(clamp_min(a.x, 1e-6f));
                push(inst, make_float4(val, 0.f, 0.f, 0.f));
                break;
            }
            case 23: { // log10
                if (!pop(inst, a)) { inst.error = 1002; return; }
                float val = log10f(clamp_min(a.x, 1e-6f));
                push(inst, make_float4(val, 0.f, 0.f, 0.f));
                break;
            }
            case 24: { // sin
                if (!pop(inst, a)) { inst.error = 1002; return; }
                push(inst, make_float4(sinf(a.x), 0.f, 0.f, 0.f));
                break;
            }
            case 25: { // cos
                if (!pop(inst, a)) { inst.error = 1002; return; }
                push(inst, make_float4(cosf(a.x), 0.f, 0.f, 0.f));
                break;
            }
            case 26: { // tan
                if (!pop(inst, a)) { inst.error = 1002; return; }
                push(inst, make_float4(tanf(a.x), 0.f, 0.f, 0.f));
                break;
            }
            case 27: { // asin
                if (!pop(inst, a)) { inst.error = 1002; return; }
                push(inst, make_float4(asinf(a.x), 0.f, 0.f, 0.f));
                break;
            }
            case 28: { // acos
                if (!pop(inst, a)) { inst.error = 1002; return; }
                push(inst, make_float4(acosf(a.x), 0.f, 0.f, 0.f));
                break;
            }
            case 29: { // atan
                if (!pop(inst, a)) { inst.error = 1002; return; }
                push(inst, make_float4(atanf(a.x), 0.f, 0.f, 0.f));
                break;
            }
            case 30: { // sinh
                if (!pop(inst, a)) { inst.error = 1002; return; }
                push(inst, make_float4(sinhf(a.x), 0.f, 0.f, 0.f));
                break;
            }
            case 31: { // cosh
                if (!pop(inst, a)) { inst.error = 1002; return; }
                push(inst, make_float4(coshf(a.x), 0.f, 0.f, 0.f));
                break;
            }
            case 32: { // tanh
                if (!pop(inst, a)) { inst.error = 1002; return; }
                push(inst, make_float4(tanhf(a.x), 0.f, 0.f, 0.f));
                break;
            }
            case 33: { // sigmoid
                if (!pop(inst, a)) { inst.error = 1002; return; }
                float val = 1.f / (1.f + expf(-a.x));
                push(inst, make_float4(val, 0.f, 0.f, 0.f));
                break;
            }
            case 34: { // abs
                if (!pop(inst, a)) { inst.error = 1002; return; }
                float4 out;
                out.x = fabsf(a.x);
                out.y = fabsf(a.y);
                out.z = fabsf(a.z);
                out.w = fabsf(a.w);
                push(inst, out);
                break;
            }
            case 35: { // relu
                if (!pop(inst, a)) { inst.error = 1002; return; }
                float4 out;
                out.x = fmaxf(0.f, a.x);
                out.y = fmaxf(0.f, a.y);
                out.z = fmaxf(0.f, a.z);
                out.w = fmaxf(0.f, a.w);
                push(inst, out);
                break;
            }
            case 36: { // floor
                if (!pop(inst, a)) { inst.error = 1002; return; }
                float val = floorf(a.x);
                push(inst, make_float4(val, 0.f, 0.f, 0.f));
                break;
            }
            case 37: { // ceil
                if (!pop(inst, a)) { inst.error = 1002; return; }
                float val = ceilf(a.x);
                push(inst, make_float4(val, 0.f, 0.f, 0.f));
                break;
            }
            case 38: { // mod (a % b)
                if (!pop(inst, b) || !pop(inst, a)) { inst.error = 1002; return; }
                if (b.x == 0.f) { inst.error = 1003; return; }
                float val = fmodf(a.x, b.x);
                push(inst, make_float4(val, 0.f, 0.f, 0.f));
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
        "log10": 23,
        "sin": 24,
        "cos": 25,
        "tan": 26,
        "asin": 27,
        "acos": 28,
        "atan": 29,
        "sinh": 30,
        "cosh": 31,
        "tanh": 32,
        "sigmoid": 33,
        "abs": 34,
        "relu": 35,
        "floor": 36,
        "ceil": 37,
        "mod": 38,
        "round": 39,
        "round_he": 41,
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
        "clamp": 73,
        "fact": 74,
        "ifelse": 80,
        "gcd": 64,
        "lcm": 65,
        "nCr": 66,
        "nPr": 67,
        "store": 90,
        "load": 91,
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
        import os as _os
        if max_instances > self._INSTANCE_COUNT:
            raise ValueError(f"Maximum supported instances is {self._INSTANCE_COUNT}")
        self.max_instances = max_instances
        # Default to double precision; allow override via env
        self._use_double: bool = str(_os.environ.get("K3D_RPN_USE_DOUBLE", "1")).lower() not in {"0", "false", "no"}
        self._compile_kernel()
        self._allocate_state()

    # -------------------------- CUDA setup --------------------------
    def _cuda_source(self) -> bytes:
        # Macro-based single-precision/double-precision kernel with convenience ops and registers
        src = r"""
extern "C" {
#if !defined(USE_DOUBLE)
#define USE_DOUBLE 1
#endif
#if USE_DOUBLE
typedef double scalar_t; typedef double3 vec3_t; typedef double4 vec4_t;
#define VEC4(x,y,z,w) make_double4(x,y,z,w)
#define VEC3(x,y,z) make_double3(x,y,z)
#define SIN sin
#define COS cos
#define TAN tan
#define ASIN asin
#define ACOS acos
#define ATAN atan
#define SINH sinh
#define COSH cosh
#define TANH tanh
#define EXP exp
#define LOG log
#define LOG10 log10
#define POW pow
#define FABS fabs
#define SQRT sqrt
#define FMOD fmod
#define FLOOR floor
#define CEIL ceil
#else
typedef float scalar_t; typedef float3 vec3_t; typedef float4 vec4_t;
#define VEC4(x,y,z,w) make_float4(x,y,z,w)
#define VEC3(x,y,z) make_float3(x,y,z)
#define SIN sinf
#define COS cosf
#define TAN tanf
#define ASIN asinf
#define ACOS acosf
#define ATAN atanf
#define SINH sinhf
#define COSH coshf
#define TANH tanhf
#define EXP expf
#define LOG logf
#define LOG10 log10f
#define POW powf
#define FABS fabsf
#define SQRT sqrtf
#define FMOD fmodf
#define FLOOR floorf
#define CEIL ceilf
#endif

struct RPNInstance { vec4_t stack[64]; scalar_t regs[16]; int head; int size; int error; int reserved; };
__device__ inline void push(RPNInstance& i, const vec4_t& v){ const int m=63; int idx=(i.head+i.size)&m; i.stack[idx]=v; if(i.size<64) i.size+=1; else i.head=(i.head+1)&m; }
__device__ inline bool pop(RPNInstance& i, vec4_t& out){ if(i.size<=0) return false; const int m=63; i.size-=1; int idx=(i.head+i.size)&m; out=i.stack[idx]; if(i.size==0) i.head=0; return true; }
__device__ inline scalar_t clamp_min(scalar_t v, scalar_t e){ return (v<e)?e:v; }

__global__ void modular_rpn_geometric_kernel(int instance_id, const unsigned short* op, const scalar_t* sc, const vec3_t* ve, RPNInstance* insts, int n){
  const int MAXI=15; if(instance_id<0) instance_id=0; if(instance_id>=MAXI) instance_id=MAXI-1; RPNInstance& s=insts[instance_id];
  for(int i=0;i<n;++i){ unsigned short c=op[i]; scalar_t sv=sc[i]; vec3_t vv=ve[i]; vec4_t a,b;
    switch(c){
      case 0: push(s,VEC4(sv,0.0,0.0,0.0)); break;
      case 1: push(s,VEC4(vv.x,vv.y,vv.z,0.0)); break;
      case 10: if(!pop(s,b)||!pop(s,a)){s.error=1002;return;} push(s,VEC4(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w)); break;
      case 11: if(!pop(s,b)||!pop(s,a)){s.error=1002;return;} push(s,VEC4(a.x-b.x,a.y-b.y,a.z-b.z,a.w-b.w)); break;
      case 12: if(!pop(s,b)||!pop(s,a)){s.error=1002;return;} push(s,VEC4(a.x*b.x,a.y*b.y,a.z*b.z,a.w*b.w)); break;
      case 13: if(!pop(s,b)||!pop(s,a)){s.error=1002;return;} if(b.x==0.0){s.error=1003;return;} push(s,VEC4(a.x/b.x,0.0,0.0,0.0)); break;
      case 14: if(!pop(s,b)||!pop(s,a)){s.error=1002;return;} push(s,VEC4(POW(clamp_min(FABS(a.x),(scalar_t)1e-12), b.x),0.0,0.0,0.0)); break;
      case 15: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(-a.x,-a.y,-a.z,-a.w)); break;
      case 20: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(SQRT(a.x<0.0?0.0:a.x),0.0,0.0,0.0)); break;
      case 21: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(EXP(a.x),0.0,0.0,0.0)); break;
      case 22: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(LOG(clamp_min(a.x,(scalar_t)1e-12)),0.0,0.0,0.0)); break;
      case 23: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(LOG10(clamp_min(a.x,(scalar_t)1e-12)),0.0,0.0,0.0)); break;
      case 24: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(SIN(a.x),0.0,0.0,0.0)); break;
      case 25: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(COS(a.x),0.0,0.0,0.0)); break;
      case 26: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(TAN(a.x),0.0,0.0,0.0)); break;
      case 27: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(ASIN(a.x),0.0,0.0,0.0)); break;
      case 28: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(ACOS(a.x),0.0,0.0,0.0)); break;
      case 29: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(ATAN(a.x),0.0,0.0,0.0)); break;
      case 30: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(SINH(a.x),0.0,0.0,0.0)); break;
      case 31: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(COSH(a.x),0.0,0.0,0.0)); break;
      case 32: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(TANH(a.x),0.0,0.0,0.0)); break;
      case 33: if(!pop(s,a)){s.error=1002;return;} {scalar_t v=(scalar_t)1.0/((scalar_t)1.0+EXP(-a.x)); push(s,VEC4(v,0.0,0.0,0.0)); } break;
      case 34: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(FABS(a.x),FABS(a.y),FABS(a.z),FABS(a.w))); break;
      case 35: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(a.x>0?a.x:0.0, a.y>0?a.y:0.0, a.z>0?a.z:0.0, a.w>0?a.w:0.0)); break;
      case 36: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(FLOOR(a.x),0.0,0.0,0.0)); break;
      case 37: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(CEIL(a.x),0.0,0.0,0.0)); break;
      case 38: if(!pop(s,b)||!pop(s,a)){s.error=1002;return;} if(b.x==0.0){s.error=1003;return;} push(s,VEC4(FMOD(a.x,b.x),0.0,0.0,0.0)); break;
      case 39: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(FLOOR(a.x+(scalar_t)0.5),0.0,0.0,0.0)); break;
      case 41: { if(!pop(s,a)){s.error=1002;return;} scalar_t f=FLOOR(a.x); scalar_t frac=a.x-f; scalar_t out; if(frac>0.5) out=f+1.0; else if(frac<0.5) out=f; else { out = (FLOOR(fmod(f,2.0))==0.0 ? f : f+1.0); } push(s,VEC4(out,0.0,0.0,0.0)); } break;
      case 40: if(!pop(s,b)||!pop(s,a)){s.error=1002;return;} push(s,VEC4(a.x>b.x?1.0:0.0,0.0,0.0,0.0)); break;
      case 42: if(!pop(s,b)||!pop(s,a)){s.error=1002;return;} push(s,VEC4(a.x<b.x?1.0:0.0,0.0,0.0,0.0)); break;
      case 44: if(!pop(s,b)||!pop(s,a)){s.error=1002;return;} push(s,VEC4(a.x==b.x?1.0:0.0,0.0,0.0,0.0)); break;
      case 46: if(!pop(s,b)||!pop(s,a)){s.error=1002;return;} push(s,VEC4(a.x>b.x?a.x:b.x,0.0,0.0,0.0)); break;
      case 47: if(!pop(s,b)||!pop(s,a)){s.error=1002;return;} push(s,VEC4(a.x<b.x?a.x:b.x,0.0,0.0,0.0)); break;
      case 50: if(!pop(s,a)){s.error=1002;return;} push(s,a); push(s,a); break;
      case 51: if(!pop(s,a)||!pop(s,b)){s.error=1002;return;} push(s,a); push(s,b); break;
      case 52: if(!pop(s,a)){s.error=1002;return;} break;
      case 53: if(!pop(s,a)||!pop(s,b)){s.error=1002;return;} push(s,b); push(s,a); push(s,b); break;
      case 55: s.head=0; s.size=0; break;
      case 60: if(!pop(s,a)||!pop(s,b)){s.error=1002;return;} {scalar_t v=b.x*a.x+b.y*a.y+b.z*a.z; push(s,VEC4(v,0.0,0.0,0.0));} break;
      case 61: if(!pop(s,a)||!pop(s,b)){s.error=1002;return;} push(s,VEC4(b.y*a.z-b.z*a.y,b.z*a.x-b.x*a.z,b.x*a.y-b.y*a.x,0.0)); break;
      case 62: if(!pop(s,a)){s.error=1002;return;} {scalar_t v=SQRT(a.x*a.x+a.y*a.y+a.z*a.z); push(s,VEC4(v,0.0,0.0,0.0));} break;
      case 63: if(!pop(s,a)){s.error=1002;return;} {scalar_t m=clamp_min(SQRT(a.x*a.x+a.y*a.y+a.z*a.z),(scalar_t)1e-12); push(s,VEC4(a.x/m,a.y/m,a.z/m,0.0));} break;
      case 64: if(!pop(s,b)||!pop(s,a)){s.error=1002;return;} { long long x=(long long) llround(a.x), y=(long long) llround(b.x); x=x<0?-x:x; y=y<0?-y:y; while(y!=0){ long long t=x%y; x=y; y=t;} push(s,VEC4((scalar_t)x,0.0,0.0,0.0)); } break;
      case 65: if(!pop(s,b)||!pop(s,a)){s.error=1002;return;} { long long x=(long long) llround(a.x), y=(long long) llround(b.x); long long ax=x<0?-x:x, ay=y<0?-y:y; long long u=ax,v=ay; while(v!=0){ long long t=u%v; u=v; v=t;} long long g=u; long long l=(ax==0||ay==0)?0:(ax/g)*ay; push(s,VEC4((scalar_t)l,0.0,0.0,0.0)); } break;
      case 66: if(!pop(s,b)||!pop(s,a)){s.error=1002;return;} { scalar_t n=a.x, r=b.x; if(r<0.0||r>n){ push(s,VEC4(0.0,0.0,0.0,0.0)); break;} scalar_t val=EXP(LOG(tgamma(n+1.0))-LOG(tgamma(r+1.0))-LOG(tgamma(n-r+1.0))); push(s,VEC4(val,0.0,0.0,0.0)); } break;
      case 67: if(!pop(s,b)||!pop(s,a)){s.error=1002;return;} { scalar_t n=a.x, r=b.x; if(r<0.0||r>n){ push(s,VEC4(0.0,0.0,0.0,0.0)); break;} scalar_t val=EXP(LOG(tgamma(n+1.0))-LOG(tgamma(n-r+1.0))); push(s,VEC4(val,0.0,0.0,0.0)); } break;
      case 70: { vec4_t axis,angle,v; if(!pop(s,axis)||!pop(s,angle)||!pop(s,v)){s.error=1002;return;} scalar_t ax=axis.x,ay=axis.y,az=axis.z; scalar_t norm=clamp_min(SQRT(ax*ax+ay*ay+az*az),(scalar_t)1e-12); ax/=norm; ay/=norm; az/=norm; scalar_t sn=SIN(angle.x), cs=COS(angle.x); scalar_t dot=ax*v.x+ay*v.y+az*v.z; vec4_t cr=VEC4(ay*v.z-az*v.y, az*v.x-ax*v.z, ax*v.y-ay*v.x,0.0); vec4_t out=VEC4(v.x*cs+cr.x*sn+ax*dot*(1.0-cs), v.y*cs+cr.y*sn+ay*dot*(1.0-cs), v.z*cs+cr.z*sn+az*dot*(1.0-cs),0.0); push(s,out);} break;
      case 71: if(!pop(s,a)){s.error=1002;return;} { scalar_t f=a.x; if(!pop(s,b)){s.error=1002;return;} push(s,VEC4(b.x*f,b.y*f,b.z*f,0.0)); } break;
      case 72: if(!pop(s,a)||!pop(s,b)){s.error=1002;return;} push(s,VEC4(b.x+a.x,b.y+a.y,b.z+a.z,0.0)); break;
      case 73: { vec4_t hi,lo,x; if(!pop(s,hi)||!pop(s,lo)||!pop(s,x)){s.error=1002;return;} scalar_t v=x.x; if(v<lo.x) v=lo.x; if(v>hi.x) v=hi.x; push(s,VEC4(v,0.0,0.0,0.0)); } break;
      case 74: if(!pop(s,a)){s.error=1002;return;} push(s,VEC4(tgamma(a.x+1.0),0.0,0.0,0.0)); break;
      case 80: { vec4_t f,tv,cv; if(!pop(s,f)||!pop(s,tv)||!pop(s,cv)){s.error=1002;return;} push(s, cv.x!=0.0?tv:f);} break;
      case 90: if(!pop(s,b)||!pop(s,a)){s.error=1002;return;} { int idx=((int) llround(b.x)) & 15; s.regs[idx]=a.x; push(s,VEC4(a.x,0.0,0.0,0.0)); } break;
      case 91: if(!pop(s,a)){s.error=1002;return;} { int idx=((int) llround(a.x)) & 15; push(s,VEC4(s.regs[idx],0.0,0.0,0.0)); } break;
      default: s.error=9001; return;
    }
  }
}

} // extern "C"
"""
        return src.encode("utf-8")

    def _compile_kernel(self) -> None:
        res, prog = nvrtc.nvrtcCreateProgram(self._cuda_source(), b"modular_rpn.cu", 0, [], [])
        if res != 0:
            raise RuntimeError(f"nvrtcCreateProgram failed: {res}")

        opts = [b"--gpu-architecture=compute_70", b"--fmad=false", (b"-DUSE_DOUBLE=1" if self._use_double else b"-DUSE_DOUBLE=0")]
        res, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if res != 0:
            log_size_res, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
            if log_size_res == 0 and log_size > 0:
                log_buffer = bytearray(log_size)
                nvrtc.nvrtcGetProgramLog(prog, log_buffer)
                log_text = log_buffer.decode("utf-8", errors="replace")
            else:
                log_text = "<unavailable>"
            nvrtc.nvrtcDestroyProgram(prog)
            raise RuntimeError(f"NVRTC compilation failed (code {res}):\n{log_text}")

        res, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
        if res != 0:
            nvrtc.nvrtcDestroyProgram(prog)
            raise RuntimeError(f"nvrtcGetPTXSize failed: {res}")
        ptx_buffer = bytearray(ptx_size)
        res, = nvrtc.nvrtcGetPTX(prog, ptx_buffer)
        if res != 0:
            nvrtc.nvrtcDestroyProgram(prog)
            raise RuntimeError(f"nvrtcGetPTX failed: {res}")
        ptx = bytes(ptx_buffer)
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
        scalar_ctype = ctypes.c_double if self._use_double else ctypes.c_float
        class RPNInstance(ctypes.Structure):
            _fields_ = [
                ("stack", scalar_ctype * (self._STACK_MAX * 4)),
                ("regs", scalar_ctype * 16),
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
        if self._use_double:
            scalars_arr = np.asarray(scalars, dtype=np.float64)
            vectors_arr = np.asarray(vectors, dtype=np.float64)
        else:
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

        # Ensure the original CUDA context associated with this engine is current.
        err, = cuda.cuCtxSetCurrent(self._ctx)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuCtxSetCurrent failed: {err}")

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
            state_ptr = cuda.CUdeviceptr(int(self._d_states) + offset)
            cuda.cuMemcpyDtoH(ctypes.addressof(instance_struct), state_ptr, self._instance_size)

            if instance_struct.error != 0:
                raise RuntimeError(f"RPN engine error code {instance_struct.error}")
            if instance_struct.size <= 0:
                raise RuntimeError("RPN stack empty after evaluation")
            top_index = (instance_struct.head + instance_struct.size - 1) & 63
            base = top_index * 4
            if self._use_double:
                result = np.array([
                    instance_struct.stack[base + 0],
                    instance_struct.stack[base + 1],
                    instance_struct.stack[base + 2],
                    instance_struct.stack[base + 3],
                ], dtype=np.float64)
            else:
                result = np.array([
                    instance_struct.stack[base + 0],
                    instance_struct.stack[base + 1],
                    instance_struct.stack[base + 2],
                    instance_struct.stack[base + 3],
                ], dtype=np.float32)
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
