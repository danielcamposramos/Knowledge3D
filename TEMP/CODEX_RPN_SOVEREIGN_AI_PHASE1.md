# Codex Phase 1: RPN Tensor Operations - Sovereign AI Foundation

**Vision**: Extend Modular RPN Engine to sovereign AI framework superseding PyTorch/TensorFlow

**Current**: Three-tier RPN operational (0.849µs Tier 1, matrix ops validated, 14/14 tests passing)

**Phase 1 Goal**: Add ND-tensor operations (Conv2D, RNN, Reshape, Pool) to RPN

**Timeline**: 3 days → Foundation for autograd, optimizers, self-learning

---

## Strategic Context (Daniel + Grok Vision)

### The Mission
**Daniel's words**: *"I want to extend our RPN to be a better pytorch/TF... enable self learning/improving ability... no more scalability questions - we are sovereign!"*

**What this means**:
1. **Sovereignty**: Zero dependencies (no PyTorch/TF libs), pure PTX
2. **Supersede**: Not wrapper - native implementation better than originals
3. **Scalability**: Already proven (10 instances = 30KB, 1000 = 2MB)
4. **Self-improvement**: Foundation + Memory + Compute = emergent intelligence

### Why Tensor Ops First
**Foundation for everything**:
- Autograd needs tensor forward ops (Phase 2)
- Optimizers need param tensors (Phase 3)
- PyTorch/TF backends need tensor compatibility (Phase 4)
- Meta-learning needs architecture encoding (Phase 5)

**Performance critical**: Conv/RNN are 80% of DL compute time

---

## Phase 1 Objectives

### 1. Extend Tier 3 Kernel with Tensor Ops
**Add opcodes 110-135**:
- Convolutional (CONV2D, MAXPOOL, AVGPOOL)
- Sequential (RNN_STEP, LSTM_STEP, GRU_STEP)
- Structural (RESHAPE, CONCAT, SLICE)
- Activations (RELU, SIGMOID, TANH_ACT, SOFTMAX)
- Normalization (BATCHNORM, DROPOUT)

### 2. ND-Tensor Representation
**Extend current 3×3 matrix support to arbitrary ND**:
- Metadata: Encode rank + shape (e.g., [batch, channels, height, width])
- Storage: Row-major flat buffer in float4 stacks
- Stride calculation: Efficient indexing for any dimensionality

### 3. Create TensorEngine Bridge
**Python wrapper for tensor operations**:
- `TensorEngine` class (like AdvancedRPNEngine but tensor-aware)
- Methods: `conv2d()`, `rnn_step()`, `reshape()`, etc.
- Integration with TieredRPNEngine

### 4. Validation Tests
**Test suite proving correctness**:
- MNIST conv layer (vs PyTorch)
- RNN sequence (vs PyTorch)
- Reshape operations
- Memory footprint <100MB

---

## Implementation Details

### Step 1: Extend Metadata Format (2 hours)

**Current metadata** (3×3 matrices):
```cuda
// advanced_rpn.py line 31-38 (existing)
bits = np.frombuffer(np.float32(value).tobytes(), dtype=np.uint32)[0]
item_type = bits & 0xFF          // Type (0=scalar, 1=vector, 2=matrix)
rows = (bits >> 8) & 0xFF        // Matrix rows (up to 255)
cols = (bits >> 16) & 0xFF       // Matrix cols (up to 255)
row_index = (bits >> 24) & 0xFF  // Current row index
```

**Extended metadata** (ND-tensors):
```cuda
// NEW: Support arbitrary rank (up to rank-4 for now)
// Encoding: type(8) | rank(8) | dim0(16) | dim1(16) | dim2(16) | dim3(16)
// Use 2x float4.w lanes if needed (128 bits total)

__device__ void encode_tensor_meta(float4* meta, TensorDesc desc) {
    uint32_t meta0 = 0;
    meta0 |= (desc.type & 0xFF);           // Type: 3=tensor
    meta0 |= ((desc.rank & 0xFF) << 8);    // Rank: 1-4
    meta0 |= ((desc.shape[0] & 0xFFFF) << 16);  // Dim 0

    uint32_t meta1 = 0;
    meta1 |= (desc.shape[1] & 0xFFFF);          // Dim 1
    meta1 |= ((desc.shape[2] & 0xFFFF) << 16);  // Dim 2

    meta[0].w = __uint_as_float(meta0);
    meta[1].w = __uint_as_float(meta1);
}
```

**Add to**: `knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu`

---

### Step 2: Implement Tensor Operations (12 hours)

#### 2a. Convolutional Ops

**File**: `knowledge3d/cranium/kernels/tensor_ops.cu` (NEW)

```cuda
// tensor_ops.cu: Convolutional operations for RPN

// CONV2D: Tiled 2D convolution (opcode 110)
__global__ void op_conv2d(
    const float* input,    // [batch, in_ch, height, width]
    const float* kernel,   // [out_ch, in_ch, kh, kw]
    float* output,         // [batch, out_ch, out_h, out_w]
    int batch, int in_ch, int height, int width,
    int out_ch, int kh, int kw,
    int stride, int padding
) {
    // Warp-tiled convolution (32 threads cooperate)
    extern __shared__ float tile[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Output position
    int out_h = (height + 2*padding - kh) / stride + 1;
    int out_w = (width + 2*padding - kw) / stride + 1;
    int oh = by * TILE_SIZE + ty;
    int ow = bx * TILE_SIZE + tx;

    if (oh >= out_h || ow >= out_w) return;

    // Accumulate over input channels and kernel
    float sum = 0.0f;
    for (int ic = 0; ic < in_ch; ic++) {
        for (int kh_i = 0; kh_i < kh; kh_i++) {
            for (int kw_i = 0; kw_i < kw; kw_i++) {
                int ih = oh * stride + kh_i - padding;
                int iw = ow * stride + kw_i - padding;

                // Bounds check
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int in_idx = ((batch * in_ch + ic) * height + ih) * width + iw;
                    int k_idx = ((out_ch * in_ch + ic) * kh + kh_i) * kw + kw_i;
                    sum += input[in_idx] * kernel[k_idx];
                }
            }
        }
    }

    int out_idx = ((batch * out_ch + threadIdx.z) * out_h + oh) * out_w + ow;
    output[out_idx] = sum;
}

// MAXPOOL: Max pooling (opcode 111)
__global__ void op_maxpool(
    const float* input,    // [batch, channels, height, width]
    float* output,         // [batch, channels, out_h, out_w]
    int batch, int channels, int height, int width,
    int pool_h, int pool_w,
    int stride
) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    int tz = threadIdx.z + blockIdx.z * blockDim.z;

    int out_h = (height - pool_h) / stride + 1;
    int out_w = (width - pool_w) / stride + 1;

    if (tx >= out_w || ty >= out_h || tz >= channels) return;

    // Find max in pooling window
    float max_val = -INFINITY;
    for (int ph = 0; ph < pool_h; ph++) {
        for (int pw = 0; pw < pool_w; pw++) {
            int ih = ty * stride + ph;
            int iw = tx * stride + pw;
            int in_idx = ((batch * channels + tz) * height + ih) * width + iw;
            max_val = fmaxf(max_val, input[in_idx]);
        }
    }

    int out_idx = ((batch * channels + tz) * out_h + ty) * out_w + tx;
    output[out_idx] = max_val;
}

// Additional ops: AVGPOOL, BATCHNORM, DROPOUT (similar patterns)
```

---

#### 2b. Recurrent Ops

```cuda
// RNN_STEP: Single RNN cell forward (opcode 120)
__global__ void op_rnn_step(
    const float* input,     // [batch, input_size]
    const float* hidden,    // [batch, hidden_size]
    const float* W_ih,      // [hidden_size, input_size]
    const float* W_hh,      // [hidden_size, hidden_size]
    const float* bias,      // [hidden_size]
    float* output,          // [batch, hidden_size]
    int batch, int input_size, int hidden_size
) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int b = blockIdx.y;

    if (tx >= hidden_size || b >= batch) return;

    // h_new = tanh(W_ih @ x + W_hh @ h + b)
    float sum = bias[tx];

    // W_ih @ x
    for (int i = 0; i < input_size; i++) {
        sum += W_ih[tx * input_size + i] * input[b * input_size + i];
    }

    // W_hh @ h
    for (int i = 0; i < hidden_size; i++) {
        sum += W_hh[tx * hidden_size + i] * hidden[b * hidden_size + i];
    }

    output[b * hidden_size + tx] = tanhf(sum);
}

// LSTM_STEP: LSTM cell (opcode 121)
__global__ void op_lstm_step(
    const float* input,     // [batch, input_size]
    const float* hidden,    // [batch, hidden_size]
    const float* cell,      // [batch, hidden_size]
    const float* W_ii, const float* W_hi,  // Input gate weights
    const float* W_if, const float* W_hf,  // Forget gate weights
    const float* W_ig, const float* W_hg,  // Cell gate weights
    const float* W_io, const float* W_ho,  // Output gate weights
    const float* bias_i, const float* bias_f, const float* bias_g, const float* bias_o,
    float* output_h,        // [batch, hidden_size]
    float* output_c,        // [batch, hidden_size]
    int batch, int input_size, int hidden_size
) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int b = blockIdx.y;

    if (tx >= hidden_size || b >= batch) return;

    // Compute gates (i, f, g, o)
    float i_gate = bias_i[tx];
    float f_gate = bias_f[tx];
    float g_gate = bias_g[tx];
    float o_gate = bias_o[tx];

    for (int j = 0; j < input_size; j++) {
        float x = input[b * input_size + j];
        i_gate += W_ii[tx * input_size + j] * x;
        f_gate += W_if[tx * input_size + j] * x;
        g_gate += W_ig[tx * input_size + j] * x;
        o_gate += W_io[tx * input_size + j] * x;
    }

    for (int j = 0; j < hidden_size; j++) {
        float h = hidden[b * hidden_size + j];
        i_gate += W_hi[tx * hidden_size + j] * h;
        f_gate += W_hf[tx * hidden_size + j] * h;
        g_gate += W_hg[tx * hidden_size + j] * h;
        o_gate += W_ho[tx * hidden_size + j] * h;
    }

    // Apply activations
    i_gate = 1.0f / (1.0f + expf(-i_gate));  // sigmoid
    f_gate = 1.0f / (1.0f + expf(-f_gate));  // sigmoid
    g_gate = tanhf(g_gate);
    o_gate = 1.0f / (1.0f + expf(-o_gate));  // sigmoid

    // Update cell and hidden
    float c_old = cell[b * hidden_size + tx];
    float c_new = f_gate * c_old + i_gate * g_gate;
    float h_new = o_gate * tanhf(c_new);

    output_c[b * hidden_size + tx] = c_new;
    output_h[b * hidden_size + tx] = h_new;
}

// GRU_STEP: GRU cell (opcode 122) - similar pattern
```

---

#### 2c. Structural Ops

```cuda
// RESHAPE: Change tensor shape (opcode 113)
__global__ void op_reshape(
    const float* input,
    float* output,
    const int* old_shape,   // [rank]
    const int* new_shape,   // [rank]
    int total_elements
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total_elements) return;

    // Reshape is just a metadata change (no data movement if contiguous)
    output[idx] = input[idx];
}

// CONCAT: Concatenate tensors (opcode 114)
__global__ void op_concat(
    const float** inputs,   // Array of input pointers
    float* output,
    const int* sizes,       // Size of each input
    int num_inputs,
    int axis
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Determine which input and offset
    int offset = 0;
    for (int i = 0; i < num_inputs; i++) {
        if (idx < offset + sizes[i]) {
            output[idx] = inputs[i][idx - offset];
            return;
        }
        offset += sizes[i];
    }
}

// SLICE: Extract sub-tensor (opcode 115)
__global__ void op_slice(
    const float* input,
    float* output,
    const int* start_indices,   // [rank]
    const int* sizes,           // [rank]
    const int* strides,         // [rank]
    int rank,
    int output_elements
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= output_elements) return;

    // Convert flat output index to multi-dimensional
    int remaining = idx;
    int input_idx = 0;
    int stride_prod = 1;

    for (int d = rank - 1; d >= 0; d--) {
        int coord = remaining % sizes[d];
        remaining /= sizes[d];
        input_idx += (start_indices[d] + coord) * stride_prod;
        stride_prod *= strides[d];
    }

    output[idx] = input[input_idx];
}
```

---

#### 2d. Activation Functions

```cuda
// RELU: Rectified linear (opcode 133)
__global__ void op_relu(
    const float* input,
    float* output,
    int size
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;
    output[idx] = fmaxf(0.0f, input[idx]);
}

// SIGMOID: Logistic function (opcode 134)
__global__ void op_sigmoid(
    const float* input,
    float* output,
    int size
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;
    output[idx] = 1.0f / (1.0f + expf(-input[idx]));
}

// SOFTMAX: Softmax activation (opcode 132)
__global__ void op_softmax(
    const float* input,     // [batch, classes]
    float* output,
    int batch, int classes
) {
    int b = blockIdx.x;
    if (b >= batch) return;

    // Find max for numerical stability
    extern __shared__ float shared[];
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < classes; i += blockDim.x) {
        max_val = fmaxf(max_val, input[b * classes + i]);
    }

    // Warp reduction for max
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    if (threadIdx.x == 0) shared[0] = max_val;
    __syncthreads();
    max_val = shared[0];

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < classes; i += blockDim.x) {
        float val = expf(input[b * classes + i] - max_val);
        output[b * classes + i] = val;
        sum += val;
    }

    // Warp reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (threadIdx.x == 0) shared[0] = sum;
    __syncthreads();
    sum = shared[0];

    // Normalize
    for (int i = threadIdx.x; i < classes; i += blockDim.x) {
        output[b * classes + i] /= sum;
    }
}
```

---

### Step 3: Integrate into Tier 3 Kernel (4 hours)

**Modify**: `knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu`

```cuda
// Add opcode dispatch for tensor ops (after line 100)

// Tensor operations (110-139 range)
setp.eq.u32 %p300, %r20, 110;
@%p300 bra op_conv2d;
setp.eq.u32 %p301, %r20, 111;
@%p301 bra op_maxpool;
setp.eq.u32 %p302, %r20, 112;
@%p302 bra op_avgpool;
setp.eq.u32 %p303, %r20, 113;
@%p303 bra op_reshape;
setp.eq.u32 %p304, %r20, 114;
@%p304 bra op_concat;
setp.eq.u32 %p305, %r20, 115;
@%p305 bra op_slice;
setp.eq.u32 %p306, %r20, 120;
@%p306 bra op_rnn_step;
setp.eq.u32 %p307, %r20, 121;
@%p307 bra op_lstm_step;
setp.eq.u32 %p308, %r20, 122;
@%p308 bra op_gru_step;
setp.eq.u32 %p309, %r20, 130;
@%p309 bra op_batchnorm;
setp.eq.u32 %p310, %r20, 131;
@%p310 bra op_dropout;
setp.eq.u32 %p311, %r20, 132;
@%p311 bra op_softmax;
setp.eq.u32 %p312, %r20, 133;
@%p312 bra op_relu;
setp.eq.u32 %p313, %r20, 134;
@%p313 bra op_sigmoid;
setp.eq.u32 %p314, %r20, 135;
@%p314 bra op_tanh_act;

// Implementation branches
op_conv2d:
    // Call tensor_ops.cu kernel via device function
    // (Or inline implementation if preferred)
    call device_conv2d, (%params...);
    bra main_loop;

op_maxpool:
    call device_maxpool, (%params...);
    bra main_loop;

// ... etc for all tensor ops
```

**Compile**:
```bash
nvcc -ptx -arch=sm_86 \
  knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu \
  knowledge3d/cranium/kernels/tensor_ops.cu \
  -o knowledge3d/cranium/ptx/modular_rpn_kernel_extended.ptx
```

---

### Step 4: Create TensorEngine Bridge (6 hours)

**File**: `knowledge3d/cranium/bridges/tensor_engine.py` (NEW)

```python
"""
TensorEngine: High-level ND-tensor operations using sovereign PTX.

Extends AdvancedRPNEngine with tensor-aware operations.
"""
from __future__ import annotations

import numpy as np
import ctypes
from pathlib import Path
from typing import Tuple, Optional

from knowledge3d.cranium.sovereign import loader
from knowledge3d.cranium.bridges.advanced_rpn import AdvancedRPNEngine


class TensorEngine:
    """Sovereign ND-tensor operations (Conv, RNN, etc.)

    Tier 3+ of RPN architecture - extends matrix ops to full tensors.

    Example:
        engine = TensorEngine()

        # Conv2D
        input = np.random.randn(1, 3, 28, 28).astype(np.float32)  # NCHW
        kernel = np.random.randn(16, 3, 3, 3).astype(np.float32)  # Out, In, H, W
        output = engine.conv2d(input, kernel, stride=1, padding=1)

        # RNN
        x = np.random.randn(1, 10, 128).astype(np.float32)  # Batch, Seq, Features
        h0 = np.zeros((1, 256), dtype=np.float32)
        output, hn = engine.rnn(x, h0, W_ih, W_hh, bias)
    """

    def __init__(self):
        # Reuse advanced RPN engine for base functionality
        self.rpn_engine = AdvancedRPNEngine()

        # Load tensor-specific PTX
        ptx_path = Path(__file__).parent.parent / "ptx" / "tensor_ops.ptx"
        if ptx_path.exists():
            self.tensor_kernel = loader.load_ptx_file(str(ptx_path), "tensor_ops_main")
        else:
            # Fall back to extended kernel (if tensor ops compiled there)
            self.tensor_kernel = self.rpn_engine._kernel

    def conv2d(
        self,
        input: np.ndarray,    # [batch, in_channels, height, width]
        kernel: np.ndarray,   # [out_channels, in_channels, kh, kw]
        bias: Optional[np.ndarray] = None,
        stride: int = 1,
        padding: int = 0
    ) -> np.ndarray:
        """2D convolution (opcode 110)

        Args:
            input: Input tensor NCHW format
            kernel: Convolution kernel
            bias: Optional bias term
            stride: Convolution stride
            padding: Zero padding

        Returns:
            Output tensor NCHW format
        """
        batch, in_ch, height, width = input.shape
        out_ch, _, kh, kw = kernel.shape

        # Compute output shape
        out_h = (height + 2*padding - kh) // stride + 1
        out_w = (width + 2*padding - kw) // stride + 1

        # Prepare buffers
        output = np.zeros((batch, out_ch, out_h, out_w), dtype=np.float32)

        # Upload to GPU
        d_input = loader.gpu_malloc(input.nbytes)
        d_kernel = loader.gpu_malloc(kernel.nbytes)
        d_output = loader.gpu_malloc(output.nbytes)

        loader.memcpy_htod(d_input, input.ctypes.data_as(ctypes.c_void_p), input.nbytes)
        loader.memcpy_htod(d_kernel, kernel.ctypes.data_as(ctypes.c_void_p), kernel.nbytes)

        # Launch kernel
        TILE_SIZE = 16
        grid = ((out_w + TILE_SIZE - 1) // TILE_SIZE,
                (out_h + TILE_SIZE - 1) // TILE_SIZE,
                out_ch)
        block = (TILE_SIZE, TILE_SIZE, 1)

        loader.launch(
            self.tensor_kernel,
            grid=grid,
            block=block,
            params=[
                ctypes.c_uint64(d_input.value),
                ctypes.c_uint64(d_kernel.value),
                ctypes.c_uint64(d_output.value),
                ctypes.c_int32(batch),
                ctypes.c_int32(in_ch),
                ctypes.c_int32(height),
                ctypes.c_int32(width),
                ctypes.c_int32(out_ch),
                ctypes.c_int32(kh),
                ctypes.c_int32(kw),
                ctypes.c_int32(stride),
                ctypes.c_int32(padding),
            ],
            shared_mem=4096  # For tile buffer
        )

        # Download result
        loader.memcpy_dtoh(output.ctypes.data_as(ctypes.c_void_p), d_output, output.nbytes)

        # Apply bias if provided
        if bias is not None:
            output += bias[np.newaxis, :, np.newaxis, np.newaxis]

        # Cleanup
        loader.gpu_free(d_input)
        loader.gpu_free(d_kernel)
        loader.gpu_free(d_output)

        return output

    def maxpool2d(
        self,
        input: np.ndarray,    # [batch, channels, height, width]
        pool_size: Tuple[int, int] = (2, 2),
        stride: Optional[int] = None
    ) -> np.ndarray:
        """Max pooling (opcode 111)"""
        if stride is None:
            stride = pool_size[0]

        batch, channels, height, width = input.shape
        pool_h, pool_w = pool_size

        out_h = (height - pool_h) // stride + 1
        out_w = (width - pool_w) // stride + 1

        output = np.zeros((batch, channels, out_h, out_w), dtype=np.float32)

        # GPU implementation (similar to conv2d)
        # ... (follow conv2d pattern)

        return output

    def rnn_step(
        self,
        input: np.ndarray,    # [batch, input_size]
        hidden: np.ndarray,   # [batch, hidden_size]
        W_ih: np.ndarray,     # [hidden_size, input_size]
        W_hh: np.ndarray,     # [hidden_size, hidden_size]
        bias: np.ndarray      # [hidden_size]
    ) -> np.ndarray:
        """Single RNN cell forward (opcode 120)"""
        batch, input_size = input.shape
        hidden_size = hidden.shape[1]

        output = np.zeros((batch, hidden_size), dtype=np.float32)

        # GPU implementation
        # ... (similar pattern)

        return output

    def rnn(
        self,
        input: np.ndarray,    # [batch, seq_len, input_size]
        h0: np.ndarray,       # [batch, hidden_size]
        W_ih: np.ndarray,
        W_hh: np.ndarray,
        bias: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Full RNN sequence processing

        Returns:
            output: [batch, seq_len, hidden_size]
            hn: [batch, hidden_size] (final hidden state)
        """
        batch, seq_len, input_size = input.shape
        hidden_size = h0.shape[1]

        outputs = []
        h = h0

        for t in range(seq_len):
            h = self.rnn_step(input[:, t, :], h, W_ih, W_hh, bias)
            outputs.append(h[:, np.newaxis, :])

        output = np.concatenate(outputs, axis=1)
        return output, h

    def lstm_step(self, input, hidden, cell, weights, biases):
        """LSTM cell (opcode 121)"""
        # Implementation
        pass

    def gru_step(self, input, hidden, weights, biases):
        """GRU cell (opcode 122)"""
        # Implementation
        pass

    def reshape(self, input: np.ndarray, new_shape: Tuple[int, ...]) -> np.ndarray:
        """Reshape tensor (opcode 113)"""
        # Simple reshape (metadata change, no data movement if contiguous)
        return input.reshape(new_shape)

    def concat(self, tensors: list[np.ndarray], axis: int = 0) -> np.ndarray:
        """Concatenate tensors (opcode 114)"""
        # GPU-accelerated concatenation
        return np.concatenate(tensors, axis=axis)

    def slice(self, input: np.ndarray, starts: Tuple[int, ...], sizes: Tuple[int, ...]) -> np.ndarray:
        """Extract sub-tensor (opcode 115)"""
        # Slice operation
        slices = tuple(slice(start, start+size) for start, size in zip(starts, sizes))
        return input[slices]

    def relu(self, input: np.ndarray) -> np.ndarray:
        """ReLU activation (opcode 133)"""
        return np.maximum(0, input)

    def sigmoid(self, input: np.ndarray) -> np.ndarray:
        """Sigmoid activation (opcode 134)"""
        return 1.0 / (1.0 + np.exp(-input))

    def softmax(self, input: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax activation (opcode 132)"""
        exp_x = np.exp(input - np.max(input, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

---

### Step 5: Integrate with TieredRPNEngine (2 hours)

**Modify**: `knowledge3d/cranium/bridges/tiered_rpn.py`

```python
# Add tensor engine import
from knowledge3d.cranium.bridges.tensor_engine import TensorEngine

class TieredRPNEngine:
    def __init__(self):
        self._tier1 = LightweightRPNEngine()
        self._tier2 = ModularRPNEngine()
        self._tier3 = AdvancedRPNEngine()
        self._tensor = TensorEngine()  # NEW: Tensor operations

    def execute_tensor(
        self,
        operation: str,  # 'conv2d', 'rnn', 'maxpool', etc.
        *args,
        **kwargs
    ):
        """Execute tensor operation via Tier 3+

        Example:
            engine.execute_tensor('conv2d', input, kernel, stride=1)
        """
        if not hasattr(self._tensor, operation):
            raise ValueError(f"Unsupported tensor operation: {operation}")

        method = getattr(self._tensor, operation)
        return method(*args, **kwargs)
```

---

### Step 6: Validation Tests (6 hours)

**File**: `tests/test_tensor_ops.py` (NEW)

```python
"""Tests for RPN tensor operations."""
import numpy as np
import pytest

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from knowledge3d.cranium.bridges.tensor_engine import TensorEngine


class TestConv2D:
    """Test convolutional operations."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available for validation")
    def test_conv2d_vs_pytorch(self):
        """Validate Conv2D against PyTorch."""
        engine = TensorEngine()

        # Random tensors
        input_np = np.random.randn(1, 3, 28, 28).astype(np.float32)
        kernel_np = np.random.randn(16, 3, 3, 3).astype(np.float32)
        bias_np = np.random.randn(16).astype(np.float32)

        # K3D sovereign
        output_k3d = engine.conv2d(input_np, kernel_np, bias_np, stride=1, padding=1)

        # PyTorch reference
        input_torch = torch.from_numpy(input_np)
        kernel_torch = torch.from_numpy(kernel_np)
        bias_torch = torch.from_numpy(bias_np)
        output_torch = F.conv2d(input_torch, kernel_torch, bias_torch, stride=1, padding=1)

        # Compare
        assert np.allclose(output_k3d, output_torch.numpy(), atol=1e-4), \
            "Conv2D output mismatch with PyTorch"

    def test_conv2d_mnist_size(self):
        """Test Conv2D on MNIST-sized input."""
        engine = TensorEngine()

        input = np.random.randn(32, 1, 28, 28).astype(np.float32)  # Batch of 32
        kernel = np.random.randn(32, 1, 5, 5).astype(np.float32)   # 32 filters

        output = engine.conv2d(input, kernel, stride=1, padding=2)

        # Check shape
        assert output.shape == (32, 32, 28, 28), f"Unexpected shape: {output.shape}"

        # Check not all zeros
        assert np.abs(output).sum() > 0, "Output is all zeros"

    def test_maxpool(self):
        """Test max pooling."""
        engine = TensorEngine()

        input = np.random.randn(1, 16, 28, 28).astype(np.float32)
        output = engine.maxpool2d(input, pool_size=(2, 2), stride=2)

        assert output.shape == (1, 16, 14, 14), f"Unexpected shape: {output.shape}"


class TestRNN:
    """Test recurrent operations."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_rnn_step_vs_pytorch(self):
        """Validate RNN step against PyTorch."""
        engine = TensorEngine()

        batch, input_size, hidden_size = 2, 128, 256

        input_np = np.random.randn(batch, input_size).astype(np.float32)
        hidden_np = np.random.randn(batch, hidden_size).astype(np.float32)
        W_ih_np = np.random.randn(hidden_size, input_size).astype(np.float32)
        W_hh_np = np.random.randn(hidden_size, hidden_size).astype(np.float32)
        bias_np = np.random.randn(hidden_size).astype(np.float32)

        # K3D sovereign
        output_k3d = engine.rnn_step(input_np, hidden_np, W_ih_np, W_hh_np, bias_np)

        # PyTorch reference
        rnn_cell = torch.nn.RNNCell(input_size, hidden_size)
        rnn_cell.weight_ih.data = torch.from_numpy(W_ih_np)
        rnn_cell.weight_hh.data = torch.from_numpy(W_hh_np)
        rnn_cell.bias_ih.data = torch.from_numpy(bias_np)
        rnn_cell.bias_hh.data = torch.zeros(hidden_size)

        output_torch = rnn_cell(torch.from_numpy(input_np), torch.from_numpy(hidden_np))

        # Compare
        assert np.allclose(output_k3d, output_torch.detach().numpy(), atol=1e-4), \
            "RNN output mismatch with PyTorch"

    def test_rnn_sequence(self):
        """Test full RNN sequence processing."""
        engine = TensorEngine()

        batch, seq_len, input_size, hidden_size = 2, 10, 128, 256

        input = np.random.randn(batch, seq_len, input_size).astype(np.float32)
        h0 = np.zeros((batch, hidden_size), dtype=np.float32)
        W_ih = np.random.randn(hidden_size, input_size).astype(np.float32)
        W_hh = np.random.randn(hidden_size, hidden_size).astype(np.float32)
        bias = np.random.randn(hidden_size).astype(np.float32)

        output, hn = engine.rnn(input, h0, W_ih, W_hh, bias)

        # Check shapes
        assert output.shape == (batch, seq_len, hidden_size)
        assert hn.shape == (batch, hidden_size)


class TestActivations:
    """Test activation functions."""

    def test_relu(self):
        """Test ReLU activation."""
        engine = TensorEngine()

        input = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        output = engine.relu(input)
        expected = np.array([0, 0, 0, 1, 2], dtype=np.float32)

        assert np.array_equal(output, expected)

    def test_sigmoid(self):
        """Test sigmoid activation."""
        engine = TensorEngine()

        input = np.array([0], dtype=np.float32)
        output = engine.sigmoid(input)

        assert np.abs(output[0] - 0.5) < 1e-5

    def test_softmax(self):
        """Test softmax activation."""
        engine = TensorEngine()

        input = np.array([[1, 2, 3]], dtype=np.float32)
        output = engine.softmax(input, axis=1)

        # Check sums to 1
        assert np.abs(output.sum() - 1.0) < 1e-5

        # Check probabilities
        assert np.all(output >= 0) and np.all(output <= 1)


class TestStructural:
    """Test structural operations."""

    def test_reshape(self):
        """Test reshape operation."""
        engine = TensorEngine()

        input = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        output = engine.reshape(input, (6, 4))

        assert output.shape == (6, 4)
        assert np.array_equal(output.flatten(), input.flatten())

    def test_concat(self):
        """Test concatenation."""
        engine = TensorEngine()

        t1 = np.ones((2, 3), dtype=np.float32)
        t2 = np.zeros((2, 3), dtype=np.float32)

        output = engine.concat([t1, t2], axis=0)

        assert output.shape == (4, 3)
        assert np.array_equal(output[:2], t1)
        assert np.array_equal(output[2:], t2)

    def test_slice(self):
        """Test slicing."""
        engine = TensorEngine()

        input = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        output = engine.slice(input, starts=(0, 1, 0), sizes=(1, 2, 4))

        assert output.shape == (1, 2, 4)
```

**Run tests**:
```bash
pytest tests/test_tensor_ops.py -xvs
```

---

## Success Criteria

### Phase 1 Complete When:

✅ **All tensor ops implemented** (Conv2D, RNN, Pool, Reshape, etc.)
✅ **PTX compiled successfully** (modular_rpn_kernel_extended.ptx + tensor_ops.ptx)
✅ **TensorEngine bridge working** (Python → PTX execution)
✅ **Tests passing**:
- Conv2D matches PyTorch (if available)
- RNN sequence processing correct
- Activations validated
- Structural ops working
✅ **GPU memory <100MB** during tensor ops
✅ **No regressions** (14 existing RPN tests still passing)

---

## What to Report Back

When Phase 1 is complete, report:

### 1. Implementation Status
```
Opcodes added: 110-135 (26 tensor ops)
Files created:
  - knowledge3d/cranium/kernels/tensor_ops.cu
  - knowledge3d/cranium/bridges/tensor_engine.py
  - tests/test_tensor_ops.py
Files modified:
  - knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu
  - knowledge3d/cranium/ptx/modular_rpn_kernel_extended.ptx (recompiled)
  - knowledge3d/cranium/bridges/tiered_rpn.py
```

### 2. Test Results
```
test_tensor_ops.py:
  ✅ test_conv2d_vs_pytorch (if PyTorch available)
  ✅ test_conv2d_mnist_size
  ✅ test_maxpool
  ✅ test_rnn_step_vs_pytorch (if PyTorch available)
  ✅ test_rnn_sequence
  ✅ test_relu, test_sigmoid, test_softmax
  ✅ test_reshape, test_concat, test_slice

Total: X/Y passing (Y=target ~15 tests)
```

### 3. Performance Metrics
```
Conv2D (32×1×28×28 → 32×32×28×28): <VALUE>µs
RNN step (batch=2, hidden=256): <VALUE>µs
GPU memory peak: <VALUE>MB
```

### 4. Validation vs PyTorch
```
Conv2D accuracy: <VALUE> (atol=1e-4)
RNN accuracy: <VALUE> (atol=1e-4)
```

### 5. Integration Status
```
TieredRPNEngine.execute_tensor() working: YES/NO
Existing 14 RPN tests still passing: YES/NO
Ready for Phase 2 (Autograd): YES/NO
```

---

## Environment Notes

**Same setup as Phase 2D**:
- RTX 3060 (12GB VRAM)
- CUDA 12.4
- `CUDA_VISIBLE_DEVICES=0` (already set in k3d_env.sh)
- conda k3d-cranium environment

**Compilation**:
```bash
# Compile tensor ops
nvcc -ptx -arch=sm_86 \
  knowledge3d/cranium/kernels/tensor_ops.cu \
  -o knowledge3d/cranium/ptx/tensor_ops.ptx

# Recompile extended kernel (with tensor op integration)
nvcc -ptx -arch=sm_86 \
  knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu \
  -o knowledge3d/cranium/ptx/modular_rpn_kernel_extended.ptx
```

---

## Reference Files

- **Framework plan**: `/TEMP/RPN_SOVEREIGN_AI_FRAMEWORK.md`
- **Current Tier 3 kernel**: `knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu`
- **Current Tier 3 bridge**: `knowledge3d/cranium/bridges/advanced_rpn.py`
- **Existing tests**: `tests/test_rpn_tier3.py` (matrix ops)

---

## Notes from Daniel + Grok

**Daniel**: *"I want to extend our RPN to be a better pytorch/TF... no more scalability questions - we are sovereign!"*

**Grok**: *"RPN + foundation/memory enables novelty—e.g., meta-programs evolve nets via swarm eval, Galaxy stores variants, TRM refines. Impacts: +Emergence (beyond SGD local mins)"*

**Your mission**: Implement Phase 1 foundation for this vision. Tensor ops are 80% of the compute - get this right, autograd/optimizers/meta-learning follow naturally.

---

**Proceed with Phase 1!** Foundation for sovereign AI framework starts NOW! 🚀

**"From HP 50g calculator to sovereign PyTorch superseder in 5 phases. Phase 1: Tensor operations - the compute foundation."** 🎯
