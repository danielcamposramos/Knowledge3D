from __future__ import annotations

import ctypes
import hashlib
import math
import threading
from pathlib import Path
import os
import fcntl
from typing import Dict, Tuple

import numpy as np


class PTXModalityOps:
    """Compile and run PTX kernels that extract modality features."""

    TEXT_DIM = 16
    AUDIO_DIM = 24
    IMAGE_DIM = 24
    VIDEO_DIM = 24

    _MAX_AUDIO_SAMPLES = 32768
    _MAX_VIDEO_FRAMES = 8
    _MAX_IMAGE_DIM = 192
    _MAX_VIDEO_DIM = 160

    def __init__(self) -> None:
        try:
            from cuda import cuda, nvrtc  # type: ignore
        except Exception:
            try:
                from cuda.bindings import driver as cuda  # type: ignore
                from cuda.bindings import nvrtc  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "cuda-python bindings are required for PTXModalityOps; install `cuda-python` and ensure a CUDA device is available"
                ) from exc

        self._cuda = cuda
        self._nvrtc = nvrtc
        self._ctx = None
        self._module = None
        self._func_text = None
        self._func_audio = None
        self._func_image = None
        self._func_video = None
        self._lock = threading.RLock()

        self._initialise()

    # ------------------------------------------------------------------
    def _check(self, err: int, label: str) -> None:
        if err != self._cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{label} failed with error code {err}")

    def _initialise(self) -> None:
        cuda = self._cuda
        nvrtc = self._nvrtc

        err, = cuda.cuInit(0)
        self._check(err, "cuInit")

        err, dev = cuda.cuDeviceGet(0)
        self._check(err, "cuDeviceGet")

        err, ctx = cuda.cuDevicePrimaryCtxRetain(dev)
        self._check(err, "cuDevicePrimaryCtxRetain")
        self._ctx = ctx

        err, = cuda.cuCtxSetCurrent(ctx)
        self._check(err, "cuCtxSetCurrent")

        # Inter-process lock to avoid NVRTC races across concurrent runs
        lock_path = Path(os.getenv("K3D_NVRTC_LOCK", "/tmp/k3d_nvrtc_modality.lock"))
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as lfh:
            fcntl.flock(lfh.fileno(), fcntl.LOCK_EX)
            source_path = Path(__file__).with_name("modality_kernels.cu")
            if not source_path.exists():
                raise FileNotFoundError(f"Missing PTX modality kernel source: {source_path}")
            ptx_path = source_path.with_suffix('.ptx')
            module = None
            if ptx_path.exists():
                err, module = cuda.cuModuleLoad(ptx_path.as_posix().encode("utf-8"))
                if err != cuda.CUresult.CUDA_SUCCESS:
                    module = None
            if module is None:
                source = source_path.read_text(encoding="utf-8")

                if "NVRTC_BUILTINS_PATH" not in os.environ:
                    for candidate in (
                        "/usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so",
                        "/usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so.12.4",
                    ):
                        if Path(candidate).exists():
                            os.environ["NVRTC_BUILTINS_PATH"] = candidate
                            break

                res, prog = nvrtc.nvrtcCreateProgram(source.encode("utf-8"), b"modality_kernels.cu", 0, [], [])
                if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                    raise RuntimeError(f"nvrtcCreateProgram failed: {res}")

                major_attr = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
                minor_attr = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
                err, major = cuda.cuDeviceGetAttribute(major_attr, dev)
                self._check(err, "cuDeviceGetAttribute major")
                err, minor = cuda.cuDeviceGetAttribute(minor_attr, dev)
                self._check(err, "cuDeviceGetAttribute minor")

                arch = f"--gpu-architecture=compute_{major}{minor}".encode("utf-8")
                opts = [arch, b"--fmad=false", b"-I/usr/include", b"--device-as-default-execution-space"]
                res, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
                if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                    log_size_res, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
                    log = ""
                    if log_size_res == nvrtc.nvrtcResult.NVRTC_SUCCESS and log_size > 1:
                        buffer = bytearray(log_size)
                        nvrtc.nvrtcGetProgramLog(prog, buffer)
                        log = buffer.decode("utf-8", errors="replace")
                    nvrtc.nvrtcDestroyProgram(prog)
                    raise RuntimeError(f"nvrtcCompileProgram failed ({res}):\n{log}")

                res, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
                if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                    nvrtc.nvrtcDestroyProgram(prog)
                    raise RuntimeError(f"nvrtcGetPTXSize failed: {res}")

                ptx = bytearray(ptx_size)
                res, = nvrtc.nvrtcGetPTX(prog, ptx)
                nvrtc.nvrtcDestroyProgram(prog)
                if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                    raise RuntimeError(f"nvrtcGetPTX failed: {res}")

                err, module = cuda.cuModuleLoadData(bytes(ptx))
                self._check(err, "cuModuleLoadData")
        self._module = module

        err, func = cuda.cuModuleGetFunction(module, b"encode_text")
        self._check(err, "cuModuleGetFunction encode_text")
        self._func_text = func

        err, func = cuda.cuModuleGetFunction(module, b"encode_audio")
        self._check(err, "cuModuleGetFunction encode_audio")
        self._func_audio = func

        err, func = cuda.cuModuleGetFunction(module, b"encode_image")
        self._check(err, "cuModuleGetFunction encode_image")
        self._func_image = func

        err, func = cuda.cuModuleGetFunction(module, b"encode_video")
        self._check(err, "cuModuleGetFunction encode_video")
        self._func_video = func

    # ------------------------------------------------------------------
    def text_features(self, text: str) -> Tuple[np.ndarray, Dict[str, float]]:
        data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        if data.size == 0:
            data = np.zeros(1, dtype=np.uint8)
        features = self._launch_text(data)
        metrics = self._text_metrics(features)
        return features, metrics

    def audio_features(self, path: str) -> Tuple[np.ndarray, Dict[str, float]]:
        samples, sample_rate = self._load_audio(Path(path))
        features = self._launch_audio(samples, sample_rate)
        metrics = self._audio_metrics(features)
        metrics["sample_rate"] = float(sample_rate)
        metrics["sample_count"] = float(samples.size)
        return features, metrics

    def image_features(self, path: str) -> Tuple[np.ndarray, Dict[str, float]]:
        tensor, width, height, channels = self._load_image(Path(path))
        features = self._launch_image(tensor, width, height, channels)
        metrics = self._image_metrics(features)
        metrics.update({
            "width": float(width),
            "height": float(height),
            "channels": float(channels),
        })
        return features, metrics

    def video_features(self, path: str) -> Tuple[np.ndarray, Dict[str, float]]:
        tensor, frames, width, height, channels = self._load_video(Path(path))
        features = self._launch_video(tensor, frames, width, height, channels)
        metrics = self._video_metrics(features)
        metrics.update({
            "frames": float(frames),
            "width": float(width),
            "height": float(height),
            "channels": float(channels),
        })
        return features, metrics

    # ------------------------------------------------------------------
    def _launch_text(self, data: np.ndarray) -> np.ndarray:
        cuda = self._cuda
        data = np.ascontiguousarray(data)
        with self._lock:
            err, = cuda.cuCtxSetCurrent(self._ctx)
            self._check(err, "cuCtxSetCurrent")
            err, d_in = cuda.cuMemAlloc(data.nbytes)
            self._check(err, "cuMemAlloc text input")
            err, d_out = cuda.cuMemAlloc(self.TEXT_DIM * 4)
            self._check(err, "cuMemAlloc text output")
            try:
                self._check(cuda.cuMemsetD8(d_out, 0, self.TEXT_DIM * 4)[0], "cuMemsetD8")
                self._check(cuda.cuMemcpyHtoD(d_in, data.ctypes.data, data.nbytes)[0], "cuMemcpyHtoD")

                length = ctypes.c_int(int(data.size))
                out_ptr = ctypes.c_void_p(int(d_out))
                in_ptr = ctypes.c_void_p(int(d_in))
                args = (ctypes.c_void_p * 3)(
                    ctypes.cast(ctypes.pointer(in_ptr), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(length), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(out_ptr), ctypes.c_void_p),
                )
                self._check(
                    cuda.cuLaunchKernel(
                        self._func_text,
                        1,
                        1,
                        1,
                        128,
                        1,
                        1,
                        0,
                        0,
                        args,
                        0,
                    )[0],
                    "cuLaunchKernel encode_text",
                )
                self._check(cuda.cuCtxSynchronize()[0], "cuCtxSynchronize")
                result = np.zeros(self.TEXT_DIM, dtype=np.float32)
                self._check(cuda.cuMemcpyDtoH(result.ctypes.data, d_out, result.nbytes)[0], "cuMemcpyDtoH")
            finally:
                cuda.cuMemFree(d_in)
                cuda.cuMemFree(d_out)
        return result

    def _launch_audio(self, samples: np.ndarray, sample_rate: float) -> np.ndarray:
        cuda = self._cuda
        samples = np.ascontiguousarray(samples.astype(np.float32))
        with self._lock:
            err, = cuda.cuCtxSetCurrent(self._ctx)
            self._check(err, "cuCtxSetCurrent")
            err, d_in = cuda.cuMemAlloc(samples.nbytes)
            self._check(err, "cuMemAlloc audio input")
            err, d_out = cuda.cuMemAlloc(self.AUDIO_DIM * 4)
            self._check(err, "cuMemAlloc audio output")
            try:
                self._check(cuda.cuMemsetD8(d_out, 0, self.AUDIO_DIM * 4)[0], "cuMemsetD8")
                self._check(cuda.cuMemcpyHtoD(d_in, samples.ctypes.data, samples.nbytes)[0], "cuMemcpyHtoD")

                sample_count = ctypes.c_int(int(samples.size))
                rate = ctypes.c_float(float(sample_rate))
                in_ptr = ctypes.c_void_p(int(d_in))
                out_ptr = ctypes.c_void_p(int(d_out))
                args = (ctypes.c_void_p * 4)(
                    ctypes.cast(ctypes.pointer(in_ptr), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(sample_count), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(rate), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(out_ptr), ctypes.c_void_p),
                )
                self._check(
                    cuda.cuLaunchKernel(
                        self._func_audio,
                        1,
                        1,
                        1,
                        128,
                        1,
                        1,
                        0,
                        0,
                        args,
                        0,
                    )[0],
                    "cuLaunchKernel encode_audio",
                )
                self._check(cuda.cuCtxSynchronize()[0], "cuCtxSynchronize")
                result = np.zeros(self.AUDIO_DIM, dtype=np.float32)
                self._check(cuda.cuMemcpyDtoH(result.ctypes.data, d_out, result.nbytes)[0], "cuMemcpyDtoH")
            finally:
                cuda.cuMemFree(d_in)
                cuda.cuMemFree(d_out)
        return result

    def _launch_image(self, tensor: np.ndarray, width: int, height: int, channels: int) -> np.ndarray:
        cuda = self._cuda
        tensor = np.ascontiguousarray(tensor.astype(np.float32))
        with self._lock:
            err, = cuda.cuCtxSetCurrent(self._ctx)
            self._check(err, "cuCtxSetCurrent")
            err, d_in = cuda.cuMemAlloc(tensor.nbytes)
            self._check(err, "cuMemAlloc image input")
            err, d_out = cuda.cuMemAlloc(self.IMAGE_DIM * 4)
            self._check(err, "cuMemAlloc image output")
            try:
                self._check(cuda.cuMemsetD8(d_out, 0, self.IMAGE_DIM * 4)[0], "cuMemsetD8")
                self._check(cuda.cuMemcpyHtoD(d_in, tensor.ctypes.data, tensor.nbytes)[0], "cuMemcpyHtoD")

                width_c = ctypes.c_int(int(width))
                height_c = ctypes.c_int(int(height))
                channels_c = ctypes.c_int(int(channels))
                in_ptr = ctypes.c_void_p(int(d_in))
                out_ptr = ctypes.c_void_p(int(d_out))
                args = (ctypes.c_void_p * 5)(
                    ctypes.cast(ctypes.pointer(in_ptr), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(width_c), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(height_c), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(channels_c), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(out_ptr), ctypes.c_void_p),
                )
                self._check(
                    cuda.cuLaunchKernel(
                        self._func_image,
                        1,
                        1,
                        1,
                        128,
                        1,
                        1,
                        0,
                        0,
                        args,
                        0,
                    )[0],
                    "cuLaunchKernel encode_image",
                )
                self._check(cuda.cuCtxSynchronize()[0], "cuCtxSynchronize")
                result = np.zeros(self.IMAGE_DIM, dtype=np.float32)
                self._check(cuda.cuMemcpyDtoH(result.ctypes.data, d_out, result.nbytes)[0], "cuMemcpyDtoH")
            finally:
                cuda.cuMemFree(d_in)
                cuda.cuMemFree(d_out)
        return result

    def _launch_video(self, tensor: np.ndarray, frames: int, width: int, height: int, channels: int) -> np.ndarray:
        cuda = self._cuda
        tensor = np.ascontiguousarray(tensor.astype(np.float32))
        with self._lock:
            err, = cuda.cuCtxSetCurrent(self._ctx)
            self._check(err, "cuCtxSetCurrent")
            err, d_in = cuda.cuMemAlloc(tensor.nbytes)
            self._check(err, "cuMemAlloc video input")
            err, d_out = cuda.cuMemAlloc(self.VIDEO_DIM * 4)
            self._check(err, "cuMemAlloc video output")
            try:
                self._check(cuda.cuMemsetD8(d_out, 0, self.VIDEO_DIM * 4)[0], "cuMemsetD8")
                self._check(cuda.cuMemcpyHtoD(d_in, tensor.ctypes.data, tensor.nbytes)[0], "cuMemcpyHtoD")

                frames_c = ctypes.c_int(int(frames))
                width_c = ctypes.c_int(int(width))
                height_c = ctypes.c_int(int(height))
                channels_c = ctypes.c_int(int(channels))
                in_ptr = ctypes.c_void_p(int(d_in))
                out_ptr = ctypes.c_void_p(int(d_out))
                args = (ctypes.c_void_p * 6)(
                    ctypes.cast(ctypes.pointer(in_ptr), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(frames_c), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(width_c), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(height_c), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(channels_c), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(out_ptr), ctypes.c_void_p),
                )
                self._check(
                    cuda.cuLaunchKernel(
                        self._func_video,
                        1,
                        1,
                        1,
                        128,
                        1,
                        1,
                        0,
                        0,
                        args,
                        0,
                    )[0],
                    "cuLaunchKernel encode_video",
                )
                self._check(cuda.cuCtxSynchronize()[0], "cuCtxSynchronize")
                result = np.zeros(self.VIDEO_DIM, dtype=np.float32)
                self._check(cuda.cuMemcpyDtoH(result.ctypes.data, d_out, result.nbytes)[0], "cuMemcpyDtoH")
            finally:
                cuda.cuMemFree(d_in)
                cuda.cuMemFree(d_out)
        return result

    # ------------------------------------------------------------------
    def _load_audio(self, path: Path) -> Tuple[np.ndarray, float]:
        try:
            import soundfile as sf  # type: ignore

            samples, sample_rate = sf.read(path.as_posix(), dtype="float32")
        except Exception:
            samples = None
            sample_rate = 16000
        if samples is None:
            try:
                from scipy.io import wavfile  # type: ignore

                sample_rate, samples = wavfile.read(path.as_posix())
            except Exception:
                samples = None
        if samples is None:
            digest = hashlib.sha256(path.read_bytes()).digest()
            arr = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            arr = arr / 127.5 - 1.0
            return arr, float(sample_rate)
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim > 1:
            array = array.mean(axis=1)
        if array.size > self._MAX_AUDIO_SAMPLES:
            array = array[: self._MAX_AUDIO_SAMPLES]
        if array.size == 0:
            array = np.zeros(1, dtype=np.float32)
        return array, float(sample_rate)

    def _load_image(self, path: Path) -> Tuple[np.ndarray, int, int, int]:
        try:
            from PIL import Image  # type: ignore

            img = Image.open(path.as_posix()).convert("RGB")
            w, h = img.size
            max_dim = max(w, h)
            if max_dim > self._MAX_IMAGE_DIM:
                scale = self._MAX_IMAGE_DIM / float(max_dim)
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                img = img.resize(new_size)
            tensor = np.asarray(img, dtype=np.float32) / 255.0
            height, width, channels = tensor.shape
            return tensor.reshape(-1), width, height, channels
        except Exception:
            digest = hashlib.sha256(path.read_bytes()).digest()
            arr = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            arr = arr / 255.0
            width = height = int(math.sqrt(arr.size // 3) or 1)
            channels = 3
            tiled = np.tile(arr, int(math.ceil((width * height * channels) / arr.size)))
            tensor = tiled[: width * height * channels]
            return tensor.astype(np.float32), width, height, channels

    def _load_video(self, path: Path) -> Tuple[np.ndarray, int, int, int, int]:
        frames: list[np.ndarray] = []
        width = height = 0
        try:
            import imageio.v3 as iio  # type: ignore
            from PIL import Image  # type: ignore

            meta = iio.immeta(path.as_posix(), exclude_applied=True)
            frame_count = int(meta.get("nframes") or meta.get("n_frames") or 0)
            if frame_count <= 0:
                frame_count = self._MAX_VIDEO_FRAMES
            indices = np.linspace(0, max(frame_count - 1, 0), self._MAX_VIDEO_FRAMES).astype(int)
            for idx in indices:
                try:
                    frame = iio.imread(path.as_posix(), index=int(idx))
                except Exception:
                    continue
                img = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
                w, h = img.size
                max_dim = max(w, h)
                if max_dim > self._MAX_VIDEO_DIM:
                    scale = self._MAX_VIDEO_DIM / float(max_dim)
                    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                    img = img.resize(new_size)
                width, height = img.size
                tensor = np.asarray(img, dtype=np.float32) / 255.0
                frames.append(tensor)
                if len(frames) >= self._MAX_VIDEO_FRAMES:
                    break
        except Exception:
            frames = []

        if not frames:
            digest = hashlib.sha256(path.read_bytes()).digest()
            arr = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
            frames = [np.tile(arr, self._MAX_VIDEO_DIM * self._MAX_VIDEO_DIM)[: self._MAX_VIDEO_DIM * self._MAX_VIDEO_DIM]
                      .reshape(self._MAX_VIDEO_DIM, self._MAX_VIDEO_DIM, 1)]
            width = height = self._MAX_VIDEO_DIM

        stack = np.stack(frames, axis=0)
        frames_count, height, width, channels = stack.shape
        return stack.reshape(-1), frames_count, width, height, channels

    # ------------------------------------------------------------------
    def _text_metrics(self, features: np.ndarray) -> Dict[str, float]:
        hist = features[8:16]
        hist = np.clip(hist, 1e-6, 1.0)
        entropy = float(-(hist * np.log(hist)).sum() / math.log(8.0))
        return {
            "length_norm": float(features[0]),
            "mean_norm": float(features[1]),
            "std_norm": float(features[2]),
            "uppercase_ratio": float(features[3]),
            "digit_ratio": float(features[4]),
            "vowel_ratio": float(features[5]),
            "hist_entropy": entropy,
        }

    def _audio_metrics(self, features: np.ndarray) -> Dict[str, float]:
        hist = features[8:24]
        hist = np.clip(hist, 1e-6, 1.0)
        entropy = float(-(hist * np.log(hist)).sum() / math.log(16.0))
        return {
            "duration": float(features[0]),
            "mean": float(features[1]),
            "rms": float(features[2]),
            "abs_mean": float(features[3]),
            "zero_cross": float(features[4]),
            "energy": float(features[5]),
            "band_uniformity": entropy,
        }

    def _image_metrics(self, features: np.ndarray) -> Dict[str, float]:
        hist = features[12:20]
        hist = np.clip(hist, 1e-6, 1.0)
        entropy = float(-(hist * np.log(hist)).sum() / math.log(8.0))
        return {
            "width_norm": float(features[0]),
            "height_norm": float(features[1]),
            "brightness_std": float(features[9]),
            "saturation_std": float(features[11]),
            "colorfulness": float(features[22]),
            "dynamic_range": float(features[23]),
            "hist_entropy": entropy,
        }

    def _video_metrics(self, features: np.ndarray) -> Dict[str, float]:
        hist = features[13:21]
        hist = np.clip(hist, 1e-6, 1.0)
        entropy = float(-(hist * np.log(hist)).sum() / math.log(8.0))
        return {
            "frame_norm": float(features[0]),
            "motion_mean": float(features[9]),
            "motion_std": float(features[10]),
            "brightness_std": float(features[8]),
            "saturation_std": float(features[12]),
            "hist_entropy": entropy,
            "colorfulness": float(features[23]),
        }


__all__ = ["PTXModalityOps"]
