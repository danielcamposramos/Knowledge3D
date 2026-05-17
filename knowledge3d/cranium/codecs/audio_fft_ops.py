"""Audio FFT + Spectrogram Operations Launcher (Pure ctypes over audio_fft.ptx)

Sovereignty-compliant pure-ctypes launcher for opcodes 0x250-0x25F.
No numpy, no scipy, no torch in hot path. Callers must pass pre-allocated
ctypes buffers and manually manage GPU memory.

Kernels sourced from: knowledge3d/cranium/codecs/kernels/audio_fft.cu
Compiled to: knowledge3d/cranium/ptx/audio_fft.ptx (via nvcc -ptx)

Supported operations:
  0x250-0x253: FFT_FORWARD_256/512/1024/2048
  0x254: FFT_INVERSE
  0x255: FFT_WINDOW_HANN / 0x256: FFT_WINDOW_HAMM
  0x257: STFT_FORWARD
  0x258: STFT_INVERSE (placeholder)
  0x259: MEL_FILTER_BANK
  0x25A-0x25C: SPECTROGRAM_LINEAR / MEL / LOG
  0x25D: AUDIO_TO_DOTMAP
  0x25E: DOTMAP_TO_AUDIO (placeholder)
  0x25F: HRTF_CONVOLVE (placeholder)

Design:
  - One FFT per block (coalesced shared memory access)
  - RTX 3070 occupancy: 2-3 blocks/SM for 1024-point FFT
  - Pre-computed twiddle LUT in __constant__ memory (initialized at module load)
  - All inputs/outputs on GPU (no host staging except launches)

References:
  - Stockham FFT avoids bit-reversal permutation (1969)
  - NVIDIA cuFFT design patterns (coalesced I/O, bank conflict mitigation)
  - K3D TernaryCodecOps (pattern for kernel wiring + execution plan)
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import Any, Dict, Sequence, Union, Tuple

from knowledge3d.cranium.sovereign import loader


class AudioFFTOps:
    """Pure-ctypes launcher for audio FFT + spectrogram kernels (0x250-0x25F)."""

    # Device GPU memory pointers for twiddle LUT and windows
    _d_twiddle_1024 = None
    _d_window_hann_1024 = None
    _d_window_hamm_1024 = None

    # Host copies (for reference, not used in hot path)
    _h_twiddle_1024 = None
    _h_window_hann_1024 = None
    _h_window_hamm_1024 = None

    def __init__(self) -> None:
        """Load audio_fft.ptx and initialize constant memory with twiddles/windows."""
        ptx_path = Path(__file__).parent.parent / "ptx" / "audio_fft.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"PTX module not found at {ptx_path.resolve()}")

        module = loader.load_module_from_file(str(ptx_path))

        # Load kernel functions
        self._kernels = {
            # FFT forward (size-specialized)
            "fft_forward_1024": loader.get_function(
                module, "_Z25fft_forward_stockham_1024ILi1024EEvPK7float2PS0_i"
            ),
            "fft_forward_512": loader.get_function(
                module, "_Z25fft_forward_stockham_512ILi512EEvPK7float2PS0_i"
            ),
            "fft_forward_256": loader.get_function(
                module, "_Z25fft_forward_stockham_256ILi256EEvPK7float2PS0_i"
            ),
            "fft_forward_2048": loader.get_function(
                module, "_Z25fft_forward_stockham_2048ILi2048EEvPK7float2PS0_i"
            ),
            # FFT inverse
            "fft_inverse_1024": loader.get_function(
                module, "_Z24fft_inverse_stockham_1024ILi1024EEvPK7float2PS0_i"
            ),
            # Windowing
            "fft_window_load_hann_1024": loader.get_function(
                module, "_Z25fft_window_load_hann_1024ILi1024EvPKfP7float2PKf"
            ),
            "fft_window_load_hamming_1024": loader.get_function(
                module, "_Z28fft_window_load_hamming_1024ILi1024EvPKfP7float2PKf"
            ),
            # STFT
            "stft_forward": loader.get_function(
                module, "_Z12stft_forwardPKfP7float2iiiPKfS3_i"
            ),
            # Mel filter bank
            "mel_filter_bank": loader.get_function(
                module, "_Z15mel_filter_bankPK7float2PfPKfPKiS5_iii"
            ),
            # Spectrogram
            "spectrogram_linear": loader.get_function(
                module, "_Z17spectrogram_linearPK7float2Pfii"
            ),
            "spectrogram_log": loader.get_function(
                module, "_Z13spectrogram_logPK7float2Pfii"
            ),
            # DotMap
            "audio_to_dotmap": loader.get_function(
                module, "_Z14audio_to_dotmapPKfPciiif"
            ),
            "dotmap_to_audio": loader.get_function(
                module, "_Z13dotmap_to_audioPK7float2PfiiiPKfi"
            ),
        }

        # Pre-compute and upload constant memory
        self._init_constant_memory()

    def _init_constant_memory(self) -> None:
        """Pre-compute twiddle factors and windows, upload to GPU constant memory."""
        # 1024-point twiddle factors: e^(-2πi*k/1024) for k=0..511
        self._h_twiddle_1024 = []
        for k in range(512):
            angle = -2.0 * math.pi * k / 1024.0
            self._h_twiddle_1024.append((math.cos(angle), math.sin(angle)))

        # Hann window: 0.5 * (1 - cos(2π*n/(N-1)))
        self._h_window_hann_1024 = []
        for n in range(1024):
            angle = 2.0 * math.pi * n / 1023.0
            self._h_window_hann_1024.append(0.5 * (1.0 - math.cos(angle)))

        # Hamming window: 0.54 - 0.46 * cos(2π*n/(N-1))
        self._h_window_hamm_1024 = []
        for n in range(1024):
            angle = 2.0 * math.pi * n / 1023.0
            self._h_window_hamm_1024.append(0.54 - 0.46 * math.cos(angle))

        # Allocate GPU memory for these constants
        # Twiddle: 512 complex (float2) = 512 * 8 bytes = 4096 bytes
        self._d_twiddle_1024 = loader.gpu_malloc(512 * 8)

        # Windows: 1024 floats each
        self._d_window_hann_1024 = loader.gpu_malloc(1024 * 4)
        self._d_window_hamm_1024 = loader.gpu_malloc(1024 * 4)

        # Upload from host to device
        twiddle_array = (ctypes.c_float * 1024)()
        for i, (re, im) in enumerate(self._h_twiddle_1024):
            twiddle_array[2 * i] = ctypes.c_float(re)
            twiddle_array[2 * i + 1] = ctypes.c_float(im)

        loader.memcpy_htod(
            self._d_twiddle_1024,
            ctypes.cast(twiddle_array, ctypes.c_void_p),
            512 * 8,
        )

        hann_array = (ctypes.c_float * 1024)(*self._h_window_hann_1024)
        loader.memcpy_htod(
            self._d_window_hann_1024,
            ctypes.cast(hann_array, ctypes.c_void_p),
            1024 * 4,
        )

        hamm_array = (ctypes.c_float * 1024)(*self._h_window_hamm_1024)
        loader.memcpy_htod(
            self._d_window_hamm_1024,
            ctypes.cast(hamm_array, ctypes.c_void_p),
            1024 * 4,
        )

    def fft_forward(
        self,
        d_input: int,
        d_output: int,
        num_ffts: int,
        fft_size: int = 1024,
    ) -> None:
        """Launch FFT forward kernel.

        Args:
            d_input: GPU pointer to input (num_ffts * fft_size * 8 bytes for float2)
            d_output: GPU pointer to output (same size)
            num_ffts: Number of FFTs to compute in parallel
            fft_size: Size of each FFT (256, 512, 1024, or 2048)

        Execution plan (RTX 3070, 46 SMs):
            1024-point: 512-thread block, 96 KB shared, 3 blocks/SM -> ~46 occupancy
            512-point:  256-thread block, 24 KB shared, 6 blocks/SM -> ~46 occupancy
            256-point:  128-thread block, 6 KB shared, 12 blocks/SM -> ~46 occupancy
        """
        kernel_name = f"fft_forward_{fft_size}"
        if kernel_name not in self._kernels:
            raise ValueError(f"Unsupported FFT size: {fft_size}")

        kernel = self._kernels[kernel_name]

        # Thread block size tuned per FFT size
        block_size_map = {256: 128, 512: 256, 1024: 512, 2048: 512}
        block_size = block_size_map.get(fft_size, 512)

        # Shared memory: fft_size complex = fft_size * 8 bytes
        smem_bytes = fft_size * 8

        grid = (num_ffts, 1, 1)
        block = (block_size, 1, 1)

        loader.launch(
            kernel,
            grid=grid,
            block=block,
            shared_memory=smem_bytes,
            params=[
                ctypes.c_uint64(d_input),
                ctypes.c_uint64(d_output),
                ctypes.c_int(num_ffts),
            ],
        )
        loader.synchronize()

    def fft_inverse(
        self,
        d_input: int,
        d_output: int,
        num_ffts: int,
        fft_size: int = 1024,
    ) -> None:
        """Launch FFT inverse kernel.

        Performs: IFFT(x) = conj(FFT(conj(x))) / N + scaling.
        """
        kernel_name = f"fft_inverse_{fft_size}"
        if kernel_name not in self._kernels:
            raise ValueError(f"Inverse not implemented for FFT size: {fft_size}")

        kernel = self._kernels[kernel_name]

        block_size_map = {256: 128, 512: 256, 1024: 512, 2048: 512}
        block_size = block_size_map.get(fft_size, 512)

        smem_bytes = fft_size * 8
        grid = (num_ffts, 1, 1)
        block = (block_size, 1, 1)

        loader.launch(
            kernel,
            grid=grid,
            block=block,
            shared_memory=smem_bytes,
            params=[
                ctypes.c_uint64(d_input),
                ctypes.c_uint64(d_output),
                ctypes.c_int(num_ffts),
            ],
        )
        loader.synchronize()

    def window_hann_1024(
        self,
        d_input_float: int,
        d_output_complex: int,
        num_windows: int,
    ) -> None:
        """Apply Hann window during FFT input load.

        Args:
            d_input_float: GPU pointer to float input (num_windows * 1024 * 4 bytes)
            d_output_complex: GPU pointer to output (num_windows * 1024 * 8 bytes)
            num_windows: Number of windowed frames to process
        """
        kernel = self._kernels["fft_window_load_hann_1024"]

        grid = (num_windows, 1, 1)
        block = (256, 1, 1)

        loader.launch(
            kernel,
            grid=grid,
            block=block,
            params=[
                ctypes.c_uint64(d_input_float),
                ctypes.c_uint64(d_output_complex),
                ctypes.c_uint64(self._d_window_hann_1024.value),
            ],
        )
        loader.synchronize()

    def window_hamming_1024(
        self,
        d_input_float: int,
        d_output_complex: int,
        num_windows: int,
    ) -> None:
        """Apply Hamming window during FFT input load."""
        kernel = self._kernels["fft_window_load_hamming_1024"]

        grid = (num_windows, 1, 1)
        block = (256, 1, 1)

        loader.launch(
            kernel,
            grid=grid,
            block=block,
            params=[
                ctypes.c_uint64(d_input_float),
                ctypes.c_uint64(d_output_complex),
                ctypes.c_uint64(self._d_window_hamm_1024.value),
            ],
        )
        loader.synchronize()

    def stft_forward(
        self,
        d_input_float: int,
        d_output_complex: int,
        input_len: int,
        frame_size: int,
        hop_size: int,
        num_frames: int,
    ) -> None:
        """Launch STFT (overlapping windowed FFT).

        Args:
            d_input_float: GPU pointer to mono audio (input_len * 4 bytes)
            d_output_complex: GPU pointer to spectrogram output
                (num_frames * (frame_size/2 + 1) * 8 bytes)
            input_len: Total samples in input audio
            frame_size: Size of each frame (typically 512 or 1024)
            hop_size: Samples between frame starts (typically frame_size/4 or /2)
            num_frames: Number of frames to compute

        Memory layout:
            Input:  [s_0, s_1, ..., s_{input_len-1}]
            Output: [spectrogram[0][0..frame_size/2], spectrogram[1][0..frame_size/2], ...]
        """
        kernel = self._kernels["stft_forward"]

        grid = (num_frames, 1, 1)
        block = (256, 1, 1)
        smem_bytes = frame_size * 8  # Shared FFT buffer

        loader.launch(
            kernel,
            grid=grid,
            block=block,
            shared_memory=smem_bytes,
            params=[
                ctypes.c_uint64(d_input_float),
                ctypes.c_uint64(d_output_complex),
                ctypes.c_int(input_len),
                ctypes.c_int(frame_size),
                ctypes.c_int(hop_size),
                ctypes.c_uint64(self._d_window_hann_1024.value),
                ctypes.c_uint64(self._d_twiddle_1024.value),
                ctypes.c_int(int(math.log2(frame_size))),
            ],
        )
        loader.synchronize()

    def mel_filter_bank(
        self,
        d_spectrogram: int,
        d_mel_output: int,
        d_filter_data: int,
        d_filter_cols: int,
        d_filter_rows: int,
        n_frames: int,
        n_freqs: int,
        n_mels: int,
    ) -> None:
        """Apply Mel filter bank (triangle filters in frequency domain).

        Args:
            d_spectrogram: (n_frames, n_freqs) complex spectrogram
            d_mel_output: (n_frames, n_mels) mel spectrogram (float magnitude)
            d_filter_data: CSR format non-zero values
            d_filter_cols: CSR column indices
            d_filter_rows: CSR row pointers (n_mels + 1 entries)
            n_frames, n_freqs, n_mels: Dimensions
        """
        kernel = self._kernels["mel_filter_bank"]

        grid = (n_frames, 1, 1)
        block = (min(n_mels, 256), 1, 1)

        loader.launch(
            kernel,
            grid=grid,
            block=block,
            params=[
                ctypes.c_uint64(d_spectrogram),
                ctypes.c_uint64(d_mel_output),
                ctypes.c_uint64(d_filter_data),
                ctypes.c_uint64(d_filter_cols),
                ctypes.c_uint64(d_filter_rows),
                ctypes.c_int(n_frames),
                ctypes.c_int(n_freqs),
                ctypes.c_int(n_mels),
            ],
        )
        loader.synchronize()

    def spectrogram_magnitude(
        self,
        d_complex_spec: int,
        d_magnitude_spec: int,
        n_frames: int,
        n_freqs: int,
    ) -> None:
        """Extract magnitude from complex spectrogram.

        Output: sqrt(real^2 + imag^2) for each bin
        """
        kernel = self._kernels["spectrogram_linear"]

        total = n_frames * n_freqs
        block = (256, 1, 1)
        grid = ((total + 255) // 256, 1, 1)

        loader.launch(
            kernel,
            grid=grid,
            block=block,
            params=[
                ctypes.c_uint64(d_complex_spec),
                ctypes.c_uint64(d_magnitude_spec),
                ctypes.c_int(n_frames),
                ctypes.c_int(n_freqs),
            ],
        )
        loader.synchronize()

    def spectrogram_log(
        self,
        d_complex_spec: int,
        d_log_spec: int,
        n_frames: int,
        n_freqs: int,
        eps: float = 1e-10,
    ) -> None:
        """Extract log-magnitude from complex spectrogram.

        Output: log(sqrt(real^2 + imag^2) + eps)
        """
        kernel = self._kernels["spectrogram_log"]

        total = n_frames * n_freqs
        block = (256, 1, 1)
        grid = ((total + 255) // 256, 1, 1)

        loader.launch(
            kernel,
            grid=grid,
            block=block,
            params=[
                ctypes.c_uint64(d_complex_spec),
                ctypes.c_uint64(d_log_spec),
                ctypes.c_int(n_frames),
                ctypes.c_int(n_freqs),
                ctypes.c_float(eps),
            ],
        )
        loader.synchronize()

    def audio_to_dotmap(
        self,
        d_magnitude_spec: int,
        d_dotmap_indices: int,
        n_frames: int,
        n_freqs: int,
        max_val: float,
    ) -> None:
        """Quantize spectrogram to 8-level DotMap procedural image.

        Args:
            d_magnitude_spec: (n_frames, n_freqs) float magnitude
            d_dotmap_indices: (n_frames, n_freqs) int8 quantized levels [0-7]
            max_val: Global maximum magnitude for normalization

        Output mapping:
            magnitude / max_val * 7.5 -> quantize to [0, 7]
            Each level is an RPN procedural color reference in Galaxy
        """
        kernel = self._kernels["audio_to_dotmap"]

        total = n_frames * n_freqs
        block = (256, 1, 1)
        grid = ((total + 255) // 256, 1, 1)

        loader.launch(
            kernel,
            grid=grid,
            block=block,
            params=[
                ctypes.c_uint64(d_magnitude_spec),
                ctypes.c_uint64(d_dotmap_indices),
                ctypes.c_int(n_frames),
                ctypes.c_int(n_freqs),
                ctypes.c_float(max_val),
            ],
        )
        loader.synchronize()

    def execution_plan(
        self, *, work_items: int, preferred_tier: int = 2, fft_size: int = 1024
    ) -> Dict[str, Any]:
        """Return execution plan for audio FFT work distribution.

        Accounts for FFT size and occupancy constraints on RTX 3070.

        Returns:
            Dict with keys:
                preferred_tier: Processing tier (1=GPU-local, 2=multi-block, 3=multi-SM)
                work_items: Total FFTs to process
                fanout: Parallel FFT launches
                batch_size: FFTs per fanout
                cascade: Pipeline stages (prefetch, compute, writeback)
                occupancy_per_sm: Estimated blocks per SM for given fft_size
        """
        block_map = {256: 128, 512: 256, 1024: 512, 2048: 512}
        block_size = block_map.get(fft_size, 512)
        smem_per_block = fft_size * 8

        # RTX 3070: 96 KB shared memory per block
        blocks_per_sm = max(1, 96 * 1024 // smem_per_block)

        if work_items <= 0:
            return {
                "preferred_tier": int(preferred_tier),
                "work_items": 0,
                "fanout": 1,
                "batch_size": 1,
                "cascade": ["prefetch_twiddles", "compute_fft", "writeback"],
                "occupancy_per_sm": blocks_per_sm,
                "fft_size": fft_size,
            }

        fanout = min(work_items, 4 if preferred_tier <= 2 else 2)
        batch_size = max(1, work_items // max(1, fanout))

        return {
            "preferred_tier": int(preferred_tier),
            "work_items": int(work_items),
            "fanout": int(fanout),
            "batch_size": int(batch_size),
            "cascade": ["prefetch_twiddles", "compute_fft", "writeback"],
            "occupancy_per_sm": blocks_per_sm,
            "fft_size": fft_size,
        }


__all__ = ["AudioFFTOps"]
