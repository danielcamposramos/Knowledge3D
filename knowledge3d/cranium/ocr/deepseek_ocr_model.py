"""DeepSeek OCR Model: Sovereign CNN for PDF Text Extraction

Phase F.1 complete OCR architecture using sovereign GPU kernels.

Architecture (inspired by DeepSeek's efficient design):
    Input: PDF page image [H, W, 3] RGB

    Stage 1: Feature extraction
        conv1: 3→32, 3×3, ReLU → [H, W, 32]
        pool1: maxpool 2×2 → [H/2, W/2, 32]
        norm1: batchnorm

    Stage 2: Feature enhancement
        conv2: 32→64, 3×3, ReLU → [H/2, W/2, 64]
        pool2: maxpool 2×2 → [H/4, W/4, 64]
        norm2: batchnorm

    Stage 3: High-level features
        conv3: 64→128, 3×3, ReLU → [H/4, W/4, 128]
        norm3: batchnorm

    Stage 4: Character detection
        Sliding window over 8×8 patches
        glyph_match: 128-dim features → character probabilities

    Output: Character bounding boxes + confidences

Performance target:
    - Latency: <50ms per page (256×256 input)
    - Accuracy: >95% character recognition
    - Memory: <500 MB VRAM
"""

from typing import List, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import ctypes

from knowledge3d.cranium.sovereign import loader


class DeepSeekOCRModel:
    """Sovereign OCR model using custom GPU kernels."""

    def __init__(
        self,
        num_glyphs: int = 256,
        input_channels: int = 3,
        use_micro_trm: bool = False
    ):
        """Initialize DeepSeek OCR model.

        Args:
            num_glyphs: Number of character classes (default: 256 for ASCII)
            input_channels: Input image channels (default: 3 for RGB)
            use_micro_trm: Enable micro-TRM refinement in convolutions
        """
        self.num_glyphs = num_glyphs
        self.input_channels = input_channels
        self.use_micro_trm = use_micro_trm

        # Compile and load kernels
        self._load_kernels()

        # Initialize model parameters (random initialization)
        self._init_parameters()

        # GPU buffer cache
        self._buffers: Dict[str, loader.CUdeviceptr] = {}
        self._buffer_sizes: Dict[str, int] = {}

    def _load_kernels(self):
        """Compile and load all required kernels."""
        import subprocess
        import tempfile

        kernel_dir = Path(__file__).parent.parent / "ptx"

        # Kernels to compile
        kernels_to_load = [
            ("conv2d_3x3_v2.cu", ["conv2d_3x3_v2_fused", "conv2d_3x3_v2_no_relu"]),
            ("maxpool_2x2.cu", ["maxpool_2x2"]),
            ("batchnorm.cu", ["batchnorm_fused", "batchnorm_forward_training"]),
            ("glyph_match.cu", ["glyph_match_ncc", "glyph_match_top_k"]),
        ]

        self.modules = {}
        self.kernels = {}

        for cu_file, kernel_names in kernels_to_load:
            cu_path = kernel_dir / cu_file

            # Compile to PTX
            with tempfile.NamedTemporaryFile(suffix=".ptx", delete=False) as ptx_file:
                ptx_path = ptx_file.name

            try:
                cmd = [
                    "nvcc", "-ptx", str(cu_path), "-o", ptx_path,
                    "-arch=sm_75", "-O3", "--use_fast_math"
                ]
                subprocess.run(cmd, capture_output=True, text=True, check=True)

                # Load module
                module = loader.load_module_from_file(ptx_path)
                self.modules[cu_file] = module

                # Get kernel functions
                for kernel_name in kernel_names:
                    self.kernels[kernel_name] = loader.get_function(module, kernel_name)

                print(f"✓ Loaded {cu_file}: {', '.join(kernel_names)}")

            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to compile {cu_file}: {e.stderr}")

            finally:
                Path(ptx_path).unlink(missing_ok=True)

    def _init_parameters(self):
        """Initialize model parameters (weights and biases)."""
        np.random.seed(42)

        # Conv1: 3→32
        self.conv1_weight = np.random.randn(32, 3, 3, 3).astype(np.float32) * 0.1
        self.conv1_bias = np.zeros(32, dtype=np.float32)
        self.bn1_gamma = np.ones(32, dtype=np.float32)
        self.bn1_beta = np.zeros(32, dtype=np.float32)
        # CRITICAL FIX: Running statistics for BatchNorm1
        self.bn1_running_mean = np.zeros(32, dtype=np.float32)
        self.bn1_running_var = np.ones(32, dtype=np.float32)

        # Conv2: 32→64
        self.conv2_weight = np.random.randn(64, 3, 3, 32).astype(np.float32) * 0.1
        self.conv2_bias = np.zeros(64, dtype=np.float32)
        self.bn2_gamma = np.ones(64, dtype=np.float32)
        self.bn2_beta = np.zeros(64, dtype=np.float32)
        # CRITICAL FIX: Running statistics for BatchNorm2
        self.bn2_running_mean = np.zeros(64, dtype=np.float32)
        self.bn2_running_var = np.ones(64, dtype=np.float32)

        # Conv3: 64→128
        self.conv3_weight = np.random.randn(128, 3, 3, 64).astype(np.float32) * 0.1
        self.conv3_bias = np.zeros(128, dtype=np.float32)
        self.bn3_gamma = np.ones(128, dtype=np.float32)
        self.bn3_beta = np.zeros(128, dtype=np.float32)
        # CRITICAL FIX: Running statistics for BatchNorm3
        self.bn3_running_mean = np.zeros(128, dtype=np.float32)
        self.bn3_running_var = np.ones(128, dtype=np.float32)

        # Glyph templates: [num_glyphs, 8, 8, 128]
        self.glyph_templates = np.random.randn(
            self.num_glyphs, 8, 8, 128
        ).astype(np.float32) * 0.1

        # Micro-TRM weights (if enabled)
        if self.use_micro_trm:
            self.micro_w1_conv1 = np.random.randn(32, 64).astype(np.float32) * 0.1
            self.micro_w2_conv1 = np.random.randn(64, 32).astype(np.float32) * 0.1
            self.micro_w1_conv2 = np.random.randn(64, 64).astype(np.float32) * 0.1
            self.micro_w2_conv2 = np.random.randn(64, 64).astype(np.float32) * 0.1
            self.micro_w1_conv3 = np.random.randn(128, 64).astype(np.float32) * 0.1
            self.micro_w2_conv3 = np.random.randn(64, 128).astype(np.float32) * 0.1
        else:
            self.micro_w1_conv1 = None
            self.micro_w2_conv1 = None
            self.micro_w1_conv2 = None
            self.micro_w2_conv2 = None
            self.micro_w1_conv3 = None
            self.micro_w2_conv3 = None

    def _allocate_buffer(self, name: str, size_bytes: int) -> loader.CUdeviceptr:
        """Allocate or reuse GPU buffer."""
        if name in self._buffers:
            current_size = self._buffer_sizes.get(name, 0)
            if current_size >= size_bytes:
                return self._buffers[name]
            loader.gpu_free(self._buffers[name])

        ptr = loader.gpu_malloc(size_bytes)
        self._buffers[name] = ptr
        self._buffer_sizes[name] = size_bytes
        return ptr

    def _conv2d_forward(
        self,
        d_input: loader.CUdeviceptr,
        d_weight: loader.CUdeviceptr,
        d_bias: loader.CUdeviceptr,
        d_micro_w1: Optional[loader.CUdeviceptr],
        d_micro_w2: Optional[loader.CUdeviceptr],
        d_output: loader.CUdeviceptr,
        H: int,
        W: int,
        Cin: int,
        Cout: int,
        relu: bool = True
    ):
        """Execute convolution forward pass."""
        grid_x = (W + 15) // 16
        grid_y = (H + 15) // 16
        grid_z = Cout
        grid = (grid_x, grid_y, grid_z)
        block = (16, 16, 1)

        kernel = self.kernels["conv2d_3x3_v2_fused" if relu else "conv2d_3x3_v2_no_relu"]

        # Prepare micro-TRM pointers (use nullptr if disabled)
        micro_w1_ptr = d_micro_w1.value if d_micro_w1 is not None else 0
        micro_w2_ptr = d_micro_w2.value if d_micro_w2 is not None else 0

        params = [
            ctypes.c_uint64(d_input.value),
            ctypes.c_uint64(d_weight.value),
            ctypes.c_uint64(d_bias.value),
            ctypes.c_uint64(micro_w1_ptr),
            ctypes.c_uint64(micro_w2_ptr),
            ctypes.c_uint64(d_output.value),
            ctypes.c_int(H),
            ctypes.c_int(W),
            ctypes.c_int(Cin),
            ctypes.c_int(Cout),
            ctypes.c_int(1),  # stride
            ctypes.c_int(1),  # padding
            ctypes.c_bool(self.use_micro_trm),
        ]

        loader.launch(kernel, grid, block, params)

    def _maxpool_forward(
        self,
        d_input: loader.CUdeviceptr,
        d_output: loader.CUdeviceptr,
        H: int,
        W: int,
        C: int
    ):
        """Execute max pooling forward pass."""
        H_out = H // 2
        W_out = W // 2

        grid_x = (W_out + 15) // 16
        grid_y = (H_out + 15) // 16
        grid_z = C
        grid = (grid_x, grid_y, grid_z)
        block = (16, 16, 1)

        params = [
            ctypes.c_uint64(d_input.value),
            ctypes.c_uint64(d_output.value),
            ctypes.c_int(H),
            ctypes.c_int(W),
            ctypes.c_int(C),
        ]

        loader.launch(self.kernels["maxpool_2x2"], grid, block, params)

    def _batchnorm_forward(
        self,
        d_input: loader.CUdeviceptr,
        d_output: loader.CUdeviceptr,
        d_gamma: loader.CUdeviceptr,
        d_beta: loader.CUdeviceptr,
        H: int,
        W: int,
        C: int,
        *,
        return_stats: bool = False
    ):
        """
        Execute batch normalization forward pass.

        Args:
            d_input: Input tensor pointer
            d_output: Output tensor pointer
            d_gamma: Scale parameter pointer
            d_beta: Shift parameter pointer
            H, W, C: Spatial dimensions (NHWC layout)
            return_stats: If True, also compute and return batch statistics
                          along with normalized activations for backward pass.

        Returns:
            Dict with keys {"mean", "var", "x_hat"} when return_stats=True.
            Otherwise, returns None.
        """
        grid = (C, 1, 1)
        block = (256, 1, 1)

        if not return_stats:
            params = [
                ctypes.c_uint64(d_input.value),
                ctypes.c_uint64(d_output.value),
                ctypes.c_uint64(d_gamma.value),
                ctypes.c_uint64(d_beta.value),
                ctypes.c_int(H),
                ctypes.c_int(W),
                ctypes.c_int(C),
                ctypes.c_float(1e-5),
            ]
            loader.launch(self.kernels["batchnorm_fused"], grid, block, params)
            return None

        spatial_size = H * W * C
        d_x_hat = loader.gpu_malloc(spatial_size * 4)
        d_batch_mean = loader.gpu_malloc(C * 4)
        d_batch_var = loader.gpu_malloc(C * 4)

        params = [
            ctypes.c_uint64(d_input.value),
            ctypes.c_uint64(d_output.value),
            ctypes.c_uint64(d_x_hat.value),
            ctypes.c_uint64(d_batch_mean.value),
            ctypes.c_uint64(d_batch_var.value),
            ctypes.c_uint64(d_gamma.value),
            ctypes.c_uint64(d_beta.value),
            ctypes.c_int(1),  # Batch dimension N (single image)
            ctypes.c_int(H),
            ctypes.c_int(W),
            ctypes.c_int(C),
            ctypes.c_float(1e-5),
        ]

        loader.launch(self.kernels["batchnorm_forward_training"], grid, block, params)
        loader.synchronize()

        batch_mean = np.empty(C, dtype=np.float32)
        batch_var = np.empty(C, dtype=np.float32)
        x_hat = np.empty((H, W, C), dtype=np.float32)

        loader.memcpy_dtoh(batch_mean.ctypes.data_as(ctypes.c_void_p), d_batch_mean, batch_mean.nbytes)
        loader.memcpy_dtoh(batch_var.ctypes.data_as(ctypes.c_void_p), d_batch_var, batch_var.nbytes)
        loader.memcpy_dtoh(x_hat.ctypes.data_as(ctypes.c_void_p), d_x_hat, x_hat.nbytes)

        loader.gpu_free(d_x_hat)
        loader.gpu_free(d_batch_mean)
        loader.gpu_free(d_batch_var)

        return {
            "mean": batch_mean,
            "var": batch_var,
            "x_hat": x_hat,
        }

    def forward(self, image: np.ndarray, cache_for_backward: bool = False) -> Dict[str, np.ndarray]:
        """Run complete OCR pipeline.

        Args:
            image: Input PDF page image [H, W, 3] RGB, float32, range [0, 1]
            cache_for_backward: If True, cache all intermediate activations for backprop

        Returns:
            Dictionary with:
                - feature_map: Final feature map [H/4, W/4, 128]
                - patches: Extracted 8×8 patches for character matching
                - cache: (if cache_for_backward=True) All intermediate activations
                - (Future: character detections with bounding boxes)
        """
        assert image.dtype == np.float32
        assert len(image.shape) == 3
        H, W, C_in = image.shape
        assert C_in == 3

        # print(f"\n🔍 DeepSeek OCR Forward Pass: {H}×{W}×{C_in}")

        # Allocate buffers
        # (Simplified: allocate max sizes, reuse across layers)
        max_size = H * W * 128 * 4  # Max feature map size
        d_stage1_in = self._allocate_buffer("stage1_in", max_size)
        d_stage1_out = self._allocate_buffer("stage1_out", max_size)
        d_stage2_in = self._allocate_buffer("stage2_in", max_size)
        d_stage2_out = self._allocate_buffer("stage2_out", max_size)

        # Upload input image
        loader.memcpy_htod(d_stage1_in, image.ctypes.data_as(ctypes.c_void_p), image.nbytes)

        # Initialize cache if requested
        cache = {} if cache_for_backward else None
        if cache_for_backward:
            cache['input'] = image.copy()
            cache['shapes'] = {}  # Track shapes at each stage

        # Upload weights (simplified: upload all at once)
        # TODO: Cache weights on GPU
        d_conv1_w = loader.gpu_malloc(self.conv1_weight.nbytes)
        d_conv1_b = loader.gpu_malloc(self.conv1_bias.nbytes)
        loader.memcpy_htod(d_conv1_w, self.conv1_weight.ctypes.data_as(ctypes.c_void_p), self.conv1_weight.nbytes)
        loader.memcpy_htod(d_conv1_b, self.conv1_bias.ctypes.data_as(ctypes.c_void_p), self.conv1_bias.nbytes)

        # Stage 1: Conv1 + Pool1 + BN1
        # print("  Stage 1: Conv1 (3→32) + MaxPool + BatchNorm")
        self._conv2d_forward(
            d_stage1_in, d_conv1_w, d_conv1_b, None, None, d_stage1_out,
            H, W, 3, 32, relu=True
        )
        loader.synchronize()

        # Cache Conv1 output (after ReLU, before MaxPool)
        if cache_for_backward:
            conv1_out = np.empty((H, W, 32), dtype=np.float32)
            loader.memcpy_dtoh(conv1_out.ctypes.data_as(ctypes.c_void_p), d_stage1_out, conv1_out.nbytes)
            cache['conv1_out'] = conv1_out
            cache['shapes']['after_conv1'] = (H, W, 32)

        # MaxPool: [H, W, 32] → [H/2, W/2, 32]
        self._maxpool_forward(d_stage1_out, d_stage2_in, H, W, 32)
        H, W = H // 2, W // 2
        loader.synchronize()

        # Cache MaxPool1 output (before BN1)
        if cache_for_backward:
            pool1_out = np.empty((H, W, 32), dtype=np.float32)
            loader.memcpy_dtoh(pool1_out.ctypes.data_as(ctypes.c_void_p), d_stage2_in, pool1_out.nbytes)
            cache['pool1_out'] = pool1_out
            cache['shapes']['after_pool1'] = (H, W, 32)

        # BatchNorm
        d_bn1_gamma = loader.gpu_malloc(self.bn1_gamma.nbytes)
        d_bn1_beta = loader.gpu_malloc(self.bn1_beta.nbytes)
        loader.memcpy_htod(d_bn1_gamma, self.bn1_gamma.ctypes.data_as(ctypes.c_void_p), self.bn1_gamma.nbytes)
        loader.memcpy_htod(d_bn1_beta, self.bn1_beta.ctypes.data_as(ctypes.c_void_p), self.bn1_beta.nbytes)
        bn1_stats = self._batchnorm_forward(
            d_stage2_in,
            d_stage2_out,
            d_bn1_gamma,
            d_bn1_beta,
            H,
            W,
            32,
            return_stats=cache_for_backward,
        )
        loader.synchronize()

        if cache_for_backward and bn1_stats is not None:
            bn1_mean = bn1_stats["mean"]
            bn1_var = np.maximum(bn1_stats["var"], 1e-3)
            cache['bn1_mean'] = bn1_mean
            cache['bn1_var'] = bn1_var
            cache['bn1_x_hat'] = bn1_stats["x_hat"]

            momentum = 0.1
            self.bn1_running_mean = (1 - momentum) * self.bn1_running_mean + momentum * bn1_mean
            self.bn1_running_var = (1 - momentum) * self.bn1_running_var + momentum * bn1_var

        # Cache BN1 output
        if cache_for_backward:
            bn1_out = np.empty((H, W, 32), dtype=np.float32)
            loader.memcpy_dtoh(bn1_out.ctypes.data_as(ctypes.c_void_p), d_stage2_out, bn1_out.nbytes)
            cache['bn1_out'] = bn1_out
            cache['shapes']['after_bn1'] = (H, W, 32)

        # Stage 2: Conv2 + Pool2 + BN2
        # print(f"  Stage 2: Conv2 (32→64) + MaxPool + BatchNorm (now {H}×{W})")
        d_conv2_w = loader.gpu_malloc(self.conv2_weight.nbytes)
        d_conv2_b = loader.gpu_malloc(self.conv2_bias.nbytes)
        loader.memcpy_htod(d_conv2_w, self.conv2_weight.ctypes.data_as(ctypes.c_void_p), self.conv2_weight.nbytes)
        loader.memcpy_htod(d_conv2_b, self.conv2_bias.ctypes.data_as(ctypes.c_void_p), self.conv2_bias.nbytes)

        self._conv2d_forward(
            d_stage2_out, d_conv2_w, d_conv2_b, None, None, d_stage1_out,
            H, W, 32, 64, relu=True
        )
        loader.synchronize()

        # Cache Conv2 output (after ReLU, before MaxPool)
        if cache_for_backward:
            conv2_out = np.empty((H, W, 64), dtype=np.float32)
            loader.memcpy_dtoh(conv2_out.ctypes.data_as(ctypes.c_void_p), d_stage1_out, conv2_out.nbytes)
            cache['conv2_out'] = conv2_out
            cache['shapes']['after_conv2'] = (H, W, 64)

        self._maxpool_forward(d_stage1_out, d_stage2_in, H, W, 64)
        H, W = H // 2, W // 2
        loader.synchronize()

        # Cache MaxPool2 output (before BN2)
        if cache_for_backward:
            pool2_out = np.empty((H, W, 64), dtype=np.float32)
            loader.memcpy_dtoh(pool2_out.ctypes.data_as(ctypes.c_void_p), d_stage2_in, pool2_out.nbytes)
            cache['pool2_out'] = pool2_out
            cache['shapes']['after_pool2'] = (H, W, 64)

        d_bn2_gamma = loader.gpu_malloc(self.bn2_gamma.nbytes)
        d_bn2_beta = loader.gpu_malloc(self.bn2_beta.nbytes)
        loader.memcpy_htod(d_bn2_gamma, self.bn2_gamma.ctypes.data_as(ctypes.c_void_p), self.bn2_gamma.nbytes)
        loader.memcpy_htod(d_bn2_beta, self.bn2_beta.ctypes.data_as(ctypes.c_void_p), self.bn2_beta.nbytes)
        bn2_stats = self._batchnorm_forward(
            d_stage2_in,
            d_stage2_out,
            d_bn2_gamma,
            d_bn2_beta,
            H,
            W,
            64,
            return_stats=cache_for_backward,
        )
        loader.synchronize()

        # Cache BN2 output
        if cache_for_backward:
            bn2_out = np.empty((H, W, 64), dtype=np.float32)
            loader.memcpy_dtoh(bn2_out.ctypes.data_as(ctypes.c_void_p), d_stage2_out, bn2_out.nbytes)
            cache['bn2_out'] = bn2_out
            cache['shapes']['after_bn2'] = (H, W, 64)

            if bn2_stats is not None:
                bn2_mean = bn2_stats["mean"]
                bn2_var = np.maximum(bn2_stats["var"], 1e-3)
                cache['bn2_mean'] = bn2_mean
                cache['bn2_var'] = bn2_var
                cache['bn2_x_hat'] = bn2_stats["x_hat"]

                momentum = 0.1
                self.bn2_running_mean = (1 - momentum) * self.bn2_running_mean + momentum * bn2_mean
                self.bn2_running_var = (1 - momentum) * self.bn2_running_var + momentum * bn2_var

        # Stage 3: Conv3 + BN3
        # print(f"  Stage 3: Conv3 (64→128) + BatchNorm (now {H}×{W})")
        d_conv3_w = loader.gpu_malloc(self.conv3_weight.nbytes)
        d_conv3_b = loader.gpu_malloc(self.conv3_bias.nbytes)
        loader.memcpy_htod(d_conv3_w, self.conv3_weight.ctypes.data_as(ctypes.c_void_p), self.conv3_weight.nbytes)
        loader.memcpy_htod(d_conv3_b, self.conv3_bias.ctypes.data_as(ctypes.c_void_p), self.conv3_bias.nbytes)

        self._conv2d_forward(
            d_stage2_out, d_conv3_w, d_conv3_b, None, None, d_stage1_out,
            H, W, 64, 128, relu=True
        )
        loader.synchronize()

        # Cache Conv3 output (after ReLU, before BN3)
        if cache_for_backward:
            conv3_out = np.empty((H, W, 128), dtype=np.float32)
            loader.memcpy_dtoh(conv3_out.ctypes.data_as(ctypes.c_void_p), d_stage1_out, conv3_out.nbytes)
            cache['conv3_out'] = conv3_out
            cache['shapes']['after_conv3'] = (H, W, 128)
            cache['conv3_out_device'] = d_stage1_out

        d_bn3_gamma = loader.gpu_malloc(self.bn3_gamma.nbytes)
        d_bn3_beta = loader.gpu_malloc(self.bn3_beta.nbytes)
        loader.memcpy_htod(d_bn3_gamma, self.bn3_gamma.ctypes.data_as(ctypes.c_void_p), self.bn3_gamma.nbytes)
        loader.memcpy_htod(d_bn3_beta, self.bn3_beta.ctypes.data_as(ctypes.c_void_p), self.bn3_beta.nbytes)
        bn3_stats = self._batchnorm_forward(
            d_stage1_out,
            d_stage2_out,
            d_bn3_gamma,
            d_bn3_beta,
            H,
            W,
            128,
            return_stats=cache_for_backward,
        )
        loader.synchronize()

        # Download final feature map
        feature_map = np.empty((H, W, 128), dtype=np.float32)
        loader.memcpy_dtoh(
            feature_map.ctypes.data_as(ctypes.c_void_p),
            d_stage2_out,
            feature_map.nbytes
        )

        # print(f"  ✓ Feature extraction complete: {H}×{W}×128")

        # Clean up temporary weights
        loader.gpu_free(d_conv1_w)
        loader.gpu_free(d_conv1_b)
        loader.gpu_free(d_bn1_gamma)
        loader.gpu_free(d_bn1_beta)
        loader.gpu_free(d_conv2_w)
        loader.gpu_free(d_conv2_b)
        loader.gpu_free(d_bn2_gamma)
        loader.gpu_free(d_bn2_beta)
        loader.gpu_free(d_conv3_w)
        loader.gpu_free(d_conv3_b)
        loader.gpu_free(d_bn3_gamma)
        loader.gpu_free(d_bn3_beta)

        result = {
            "feature_map": feature_map,
            "output_shape": (H, W, 128),
        }

        # Include cache if requested
        if cache_for_backward:
            cache['bn3_out'] = feature_map  # BN3 output = final feature map
            cache['shapes']['after_bn3'] = (H, W, 128)
            if bn3_stats is not None:
                bn3_mean = bn3_stats["mean"]
                bn3_var = np.maximum(bn3_stats["var"], 1e-3)
                cache['bn3_mean'] = bn3_mean
                cache['bn3_var'] = bn3_var
                cache['bn3_x_hat'] = bn3_stats["x_hat"]

                momentum = 0.1
                self.bn3_running_mean = (1 - momentum) * self.bn3_running_mean + momentum * bn3_mean
                self.bn3_running_var = (1 - momentum) * self.bn3_running_var + momentum * bn3_var
            result['cache'] = cache

        return result

    def __del__(self):
        """Clean up GPU resources."""
        if not hasattr(self, "_buffers"):
            return
        for ptr in self._buffers.values():
            loader.gpu_free(ptr)
        self._buffers.clear()
        if hasattr(self, "_buffer_sizes"):
            self._buffer_sizes.clear()

    # ------------------------------------------------------------------ #
    # Weight loading helpers
    # ------------------------------------------------------------------ #
    def load_state_dict(self, state: Dict[str, np.ndarray], *, strict: bool = True) -> bool:
        """
        Load model parameters from dictionary.

        Args:
            state: Mapping of parameter name → numpy array.
            strict: If True, require all parameters.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not isinstance(state, dict):
            raise TypeError("State dict must be a dictionary")

        expected = {
            "conv1_weight": self.conv1_weight.shape,
            "conv1_bias": self.conv1_bias.shape,
            "bn1_gamma": self.bn1_gamma.shape,
            "bn1_beta": self.bn1_beta.shape,
            "conv2_weight": self.conv2_weight.shape,
            "conv2_bias": self.conv2_bias.shape,
            "bn2_gamma": self.bn2_gamma.shape,
            "bn2_beta": self.bn2_beta.shape,
            "conv3_weight": self.conv3_weight.shape,
            "conv3_bias": self.conv3_bias.shape,
            "bn3_gamma": self.bn3_gamma.shape,
            "bn3_beta": self.bn3_beta.shape,
        }

        loaded_any = False

        for name, shape in expected.items():
            if name not in state:
                if strict:
                    raise KeyError(f"Missing parameter '{name}' in state dict")
                continue

            value = np.asarray(state[name], dtype=np.float32)
            if value.shape != shape:
                raise ValueError(f"Parameter '{name}' has shape {value.shape}, expected {shape}")

            setattr(self, name, value.copy())
            loaded_any = True

        if not loaded_any:
            if strict:
                raise ValueError("No matching parameters found in provided state dict.")
            return False

        return True

    def get_state_dict(self) -> Dict[str, np.ndarray]:
        """
        Get model parameters as dictionary.

        Returns:
            Dictionary mapping parameter name → numpy array.
        """
        return {
            "conv1_weight": self.conv1_weight.copy(),
            "conv1_bias": self.conv1_bias.copy(),
            "bn1_gamma": self.bn1_gamma.copy(),
            "bn1_beta": self.bn1_beta.copy(),
            "conv2_weight": self.conv2_weight.copy(),
            "conv2_bias": self.conv2_bias.copy(),
            "bn2_gamma": self.bn2_gamma.copy(),
            "bn2_beta": self.bn2_beta.copy(),
            "conv3_weight": self.conv3_weight.copy(),
            "conv3_bias": self.conv3_bias.copy(),
            "bn3_gamma": self.bn3_gamma.copy(),
            "bn3_beta": self.bn3_beta.copy(),
        }

    def load_weights_from_file(self, path: Path, *, strict: bool = True) -> bool:
        """
        Load parameters from serialized file (.npz/.npy/.pkl).

        Args:
            path: Path to weights file.
            strict: Enforce presence of every expected parameter.

        Returns:
            True if weights loaded, False otherwise.
        """
        if not path.exists():
            raise FileNotFoundError(str(path))

        suffix = path.suffix.lower()
        state: Dict[str, np.ndarray]

        if suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            if "state_dict" in data.files:
                state_obj = data["state_dict"].item()
            else:
                state_obj = {k: data[k] for k in data.files}
            if not isinstance(state_obj, dict):
                raise ValueError(f"Unexpected data structure in {path}")
            state = state_obj
        elif suffix in {".npy", ".npz"}:
            arr = np.load(path, allow_pickle=True)
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                state = arr.item()
            elif isinstance(arr, dict):
                state = arr
            else:
                raise ValueError(f"Unsupported .npy structure for weights ({type(arr)})")
        elif suffix in {".pkl", ".pickle"}:
            import pickle

            with path.open("rb") as handle:
                state = pickle.load(handle)
            if not isinstance(state, dict):
                raise ValueError(f"Unsupported pickle contents in {path}")
        else:
            raise ValueError(f"Unsupported weight file extension: {suffix}")

        return self.load_state_dict(state, strict=strict)


__all__ = ["DeepSeekOCRModel"]
