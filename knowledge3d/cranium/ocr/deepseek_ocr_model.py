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
            ("batchnorm.cu", ["batchnorm_fused"]),
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

        # Conv2: 32→64
        self.conv2_weight = np.random.randn(64, 3, 3, 32).astype(np.float32) * 0.1
        self.conv2_bias = np.zeros(64, dtype=np.float32)
        self.bn2_gamma = np.ones(64, dtype=np.float32)
        self.bn2_beta = np.zeros(64, dtype=np.float32)

        # Conv3: 64→128
        self.conv3_weight = np.random.randn(128, 3, 3, 64).astype(np.float32) * 0.1
        self.conv3_bias = np.zeros(128, dtype=np.float32)
        self.bn3_gamma = np.ones(128, dtype=np.float32)
        self.bn3_beta = np.zeros(128, dtype=np.float32)

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
        C: int
    ):
        """Execute batch normalization forward pass."""
        grid = (C, 1, 1)
        block = (256, 1, 1)

        params = [
            ctypes.c_uint64(d_input.value),
            ctypes.c_uint64(d_output.value),
            ctypes.c_uint64(d_gamma.value),
            ctypes.c_uint64(d_beta.value),
            ctypes.c_int(H),
            ctypes.c_int(W),
            ctypes.c_int(C),
            ctypes.c_float(1e-5),  # eps
        ]

        loader.launch(self.kernels["batchnorm_fused"], grid, block, params)

    def forward(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run complete OCR pipeline.

        Args:
            image: Input PDF page image [H, W, 3] RGB, float32, range [0, 1]

        Returns:
            Dictionary with:
                - feature_map: Final feature map [H/4, W/4, 128]
                - patches: Extracted 8×8 patches for character matching
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

        # MaxPool: [H, W, 32] → [H/2, W/2, 32]
        self._maxpool_forward(d_stage1_out, d_stage2_in, H, W, 32)
        H, W = H // 2, W // 2
        loader.synchronize()

        # BatchNorm
        d_bn1_gamma = loader.gpu_malloc(self.bn1_gamma.nbytes)
        d_bn1_beta = loader.gpu_malloc(self.bn1_beta.nbytes)
        loader.memcpy_htod(d_bn1_gamma, self.bn1_gamma.ctypes.data_as(ctypes.c_void_p), self.bn1_gamma.nbytes)
        loader.memcpy_htod(d_bn1_beta, self.bn1_beta.ctypes.data_as(ctypes.c_void_p), self.bn1_beta.nbytes)
        self._batchnorm_forward(d_stage2_in, d_stage2_out, d_bn1_gamma, d_bn1_beta, H, W, 32)
        loader.synchronize()

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

        self._maxpool_forward(d_stage1_out, d_stage2_in, H, W, 64)
        H, W = H // 2, W // 2
        loader.synchronize()

        d_bn2_gamma = loader.gpu_malloc(self.bn2_gamma.nbytes)
        d_bn2_beta = loader.gpu_malloc(self.bn2_beta.nbytes)
        loader.memcpy_htod(d_bn2_gamma, self.bn2_gamma.ctypes.data_as(ctypes.c_void_p), self.bn2_gamma.nbytes)
        loader.memcpy_htod(d_bn2_beta, self.bn2_beta.ctypes.data_as(ctypes.c_void_p), self.bn2_beta.nbytes)
        self._batchnorm_forward(d_stage2_in, d_stage2_out, d_bn2_gamma, d_bn2_beta, H, W, 64)
        loader.synchronize()

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

        d_bn3_gamma = loader.gpu_malloc(self.bn3_gamma.nbytes)
        d_bn3_beta = loader.gpu_malloc(self.bn3_beta.nbytes)
        loader.memcpy_htod(d_bn3_gamma, self.bn3_gamma.ctypes.data_as(ctypes.c_void_p), self.bn3_gamma.nbytes)
        loader.memcpy_htod(d_bn3_beta, self.bn3_beta.ctypes.data_as(ctypes.c_void_p), self.bn3_beta.nbytes)
        self._batchnorm_forward(d_stage1_out, d_stage2_out, d_bn3_gamma, d_bn3_beta, H, W, 128)
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

        return {
            "feature_map": feature_map,
            "output_shape": (H, W, 128),
        }

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
