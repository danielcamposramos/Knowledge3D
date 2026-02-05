"""
Sovereign TRM implementation (PTX-only inference).
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Dict, Optional, Tuple

from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
from knowledge3d.cranium.sovereign import loader


PAD_ID = 0
BOS_ID = 1
RULE_OFFSET = 2


class SovereignTRM:
    """Sovereign TRM using PTX kernels (no PyTorch in hot path)."""

    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.rpn_engine = ModularRPNEngine()
        self.weights: Dict[str, loader.CUdeviceptr] = {}
        self.lstm_h: Optional[loader.CUdeviceptr] = None
        self.lstm_c: Optional[loader.CUdeviceptr] = None
        self.rule_vocab_size: Optional[int] = None
        self.confidence_hidden_dim: Optional[int] = None

    def load_weights(self, checkpoint_dir: str) -> None:
        """Load TRM weights from a converted checkpoint directory."""
        weight_files = [
            "embedding",
            "lstm_weight_ih",
            "lstm_weight_hh",
            "lstm_bias_ih",
            "lstm_bias_hh",
            "rule_head_weight",
            "rule_head_bias",
            "confidence_head_0_weight",
            "confidence_head_0_bias",
            "confidence_head_2_weight",
            "confidence_head_2_bias",
        ]
        base = Path(checkpoint_dir)
        if not base.exists():
            raise FileNotFoundError(f"Checkpoint dir not found: {base}")
        for name in weight_files:
            path = base / f"{name}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Weight file not found: {path}")
            array = self._load_numpy(path)
            self.weights[name] = self._upload_to_gpu(array)
            if name == "rule_head_weight":
                self.rule_vocab_size = int(array.shape[0])
            elif name == "confidence_head_0_weight":
                self.confidence_hidden_dim = int(array.shape[0])

    def reset_lstm_state(self) -> None:
        """Reset LSTM hidden and cell state to zeros (GPU buffers)."""
        state_bytes = self.hidden_dim * 4
        self.lstm_h = loader.gpu_malloc(state_bytes)
        self.lstm_c = loader.gpu_malloc(state_bytes)
        zeros = (ctypes.c_float * self.hidden_dim)()
        loader.memcpy_htod(self.lstm_h, ctypes.cast(zeros, ctypes.c_void_p), state_bytes)
        loader.memcpy_htod(self.lstm_c, ctypes.cast(zeros, ctypes.c_void_p), state_bytes)

    def infer(self, problem_tokens: list[int], max_rules: int = 20) -> Tuple[list[int], list[float]]:
        """Run sovereign inference with autoregressive decoding."""
        if not problem_tokens:
            return [], []
        self.reset_lstm_state()
        for token in problem_tokens:
            self._lstm_step(int(token))

        rules: list[int] = []
        confidences: list[float] = []
        current_token = BOS_ID

        for _ in range(int(max_rules)):
            hidden = self._lstm_step(current_token)
            logits = self._rule_head(hidden)
            try:
                next_id = self._argmax(logits, self._get_rule_vocab_size())
            finally:
                self._free_ptr(logits)

            conf_value = self._confidence_head(hidden)

            if next_id == PAD_ID:
                break
            if next_id < RULE_OFFSET:
                break
            if next_id >= self.vocab_size:
                break
            rule_idx = next_id - RULE_OFFSET
            rules.append(rule_idx)
            confidences.append(conf_value)
            current_token = next_id

        return rules, confidences

    def _embedding_lookup(self, token_id: int) -> loader.CUdeviceptr:
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(f"Token id out of range: {token_id}")
        if "embedding" not in self.weights:
            raise RuntimeError("Embedding weights not loaded.")
        result = loader.gpu_malloc(self.embedding_dim * 4)
        offset = token_id * self.embedding_dim * 4
        loader.gpu_to_gpu_copy(result, self.weights["embedding"], offset, self.embedding_dim * 4)
        return result

    def _matvec_add_bias(
        self,
        weight: loader.CUdeviceptr,
        vec: loader.CUdeviceptr,
        bias: loader.CUdeviceptr,
        *,
        rows: int,
        cols: int,
    ) -> loader.CUdeviceptr:
        result = loader.gpu_malloc(rows * 4)
        if rows == 512 and cols == 1024:
            self._matvec_fast_512x1024(weight, vec, result)
        elif rows == 1024 and cols == 512:
            self._matvec_fast_1024x512(weight, vec, result)
        else:
            self._matvec_elementwise(weight, vec, result, rows, cols)
        self._vector_add_inplace(result, bias, rows)
        return result

    def _matvec_fast_512x1024(
        self,
        weight: loader.CUdeviceptr,
        vec: loader.CUdeviceptr,
        result: loader.CUdeviceptr,
    ) -> None:
        raise NotImplementedError("OP_TRM_MATVEC_512x1024 integration pending")

    def _matvec_fast_1024x512(
        self,
        weight: loader.CUdeviceptr,
        vec: loader.CUdeviceptr,
        result: loader.CUdeviceptr,
    ) -> None:
        raise NotImplementedError("OP_TRM_MATVEC_1024x512 integration pending")

    def _matvec_elementwise(
        self,
        weight: loader.CUdeviceptr,
        vec: loader.CUdeviceptr,
        result: loader.CUdeviceptr,
        rows: int,
        cols: int,
    ) -> None:
        weight_cpu = loader.gpu_to_cpu_array(weight, rows * cols)
        vec_cpu = loader.gpu_to_cpu_array(vec, cols)
        weight_vals = weight_cpu.tolist()
        vec_vals = vec_cpu.tolist()
        programs = []
        for row in range(rows):
            tokens = []
            base = row * cols
            for col in range(cols):
                tokens.append(self._format_float(weight_vals[base + col]))
                tokens.append(self._format_float(vec_vals[col]))
                tokens.append("*")
                if col > 0:
                    tokens.append("+")
            programs.append(" ".join(tokens))
        result_ptr, count = self.rpn_engine.evaluate_batch_device(programs)
        if count != rows:
            loader.gpu_free(result_ptr)
            raise RuntimeError(f"Matvec expected {rows} results, got {count}")
        loader.gpu_to_gpu_copy(result, result_ptr, 0, rows * 4)
        loader.gpu_free(result_ptr)

    def _vector_add_inplace(self, vec: loader.CUdeviceptr, bias: loader.CUdeviceptr, size: int) -> None:
        result_ptr = self._vector_add(vec, bias, size)
        loader.gpu_to_gpu_copy(vec, result_ptr, 0, size * 4)
        loader.gpu_free(result_ptr)

    def _vector_add(self, a: loader.CUdeviceptr, b: loader.CUdeviceptr, size: int) -> loader.CUdeviceptr:
        a_vals = loader.gpu_to_cpu_array(a, size).tolist()
        b_vals = loader.gpu_to_cpu_array(b, size).tolist()
        programs = [
            f"{self._format_float(av)} {self._format_float(bv)} +"
            for av, bv in zip(a_vals, b_vals)
        ]
        result_ptr, count = self.rpn_engine.evaluate_batch_device(programs)
        if count != size:
            loader.gpu_free(result_ptr)
            raise RuntimeError(f"Vector add expected {size} results, got {count}")
        return result_ptr

    def _elementwise_mul(self, a: loader.CUdeviceptr, b: loader.CUdeviceptr, size: int) -> loader.CUdeviceptr:
        a_vals = loader.gpu_to_cpu_array(a, size).tolist()
        b_vals = loader.gpu_to_cpu_array(b, size).tolist()
        programs = [
            f"{self._format_float(av)} {self._format_float(bv)} *"
            for av, bv in zip(a_vals, b_vals)
        ]
        result_ptr, count = self.rpn_engine.evaluate_batch_device(programs)
        if count != size:
            loader.gpu_free(result_ptr)
            raise RuntimeError(f"Vector mul expected {size} results, got {count}")
        return result_ptr

    def _sigmoid_vector(self, vec: loader.CUdeviceptr, size: int) -> loader.CUdeviceptr:
        vec_vals = loader.gpu_to_cpu_array(vec, size).tolist()
        programs = [f"{self._format_float(v)} sigmoid" for v in vec_vals]
        result_ptr, count = self.rpn_engine.evaluate_batch_device(programs)
        if count != size:
            loader.gpu_free(result_ptr)
            raise RuntimeError(f"Sigmoid expected {size} results, got {count}")
        return result_ptr

    def _tanh_vector(self, vec: loader.CUdeviceptr, size: int) -> loader.CUdeviceptr:
        vec_vals = loader.gpu_to_cpu_array(vec, size).tolist()
        two = self._format_float(2.0)
        one = self._format_float(1.0)
        programs = [
            f"{self._format_float(v)} {two} * sigmoid {two} * {one} -"
            for v in vec_vals
        ]
        result_ptr, count = self.rpn_engine.evaluate_batch_device(programs)
        if count != size:
            loader.gpu_free(result_ptr)
            raise RuntimeError(f"Tanh expected {size} results, got {count}")
        return result_ptr

    def _relu_vector(self, vec: loader.CUdeviceptr, size: int) -> loader.CUdeviceptr:
        vec_vals = loader.gpu_to_cpu_array(vec, size).tolist()
        programs = [f"{self._format_float(v)} 0 max" for v in vec_vals]
        result_ptr, count = self.rpn_engine.evaluate_batch_device(programs)
        if count != size:
            loader.gpu_free(result_ptr)
            raise RuntimeError(f"ReLU expected {size} results, got {count}")
        return result_ptr

    def _slice_vector(self, vec: loader.CUdeviceptr, start: int, length: int) -> loader.CUdeviceptr:
        result = loader.gpu_malloc(length * 4)
        offset = start * 4
        loader.gpu_to_gpu_copy(result, vec, offset, length * 4)
        return result

    @staticmethod
    def _format_float(value: float) -> str:
        return format(float(value), ".9g")

    def _lstm_step(self, token_id: int) -> loader.CUdeviceptr:
        if self.lstm_h is None or self.lstm_c is None:
            self.reset_lstm_state()

        embedding_vec = self._embedding_lookup(token_id)

        ih_proj = self._matvec_add_bias(
            self.weights["lstm_weight_ih"],
            embedding_vec,
            self.weights["lstm_bias_ih"],
            rows=4 * self.hidden_dim,
            cols=self.embedding_dim,
        )
        hh_proj = self._matvec_add_bias(
            self.weights["lstm_weight_hh"],
            self.lstm_h,
            self.weights["lstm_bias_hh"],
            rows=4 * self.hidden_dim,
            cols=self.hidden_dim,
        )

        gates = self._vector_add(ih_proj, hh_proj, 4 * self.hidden_dim)

        i_raw = self._slice_vector(gates, 0, self.hidden_dim)
        f_raw = self._slice_vector(gates, self.hidden_dim, self.hidden_dim)
        g_raw = self._slice_vector(gates, 2 * self.hidden_dim, self.hidden_dim)
        o_raw = self._slice_vector(gates, 3 * self.hidden_dim, self.hidden_dim)

        i_gate = self._sigmoid_vector(i_raw, self.hidden_dim)
        f_gate = self._sigmoid_vector(f_raw, self.hidden_dim)
        g_gate = self._tanh_vector(g_raw, self.hidden_dim)
        o_gate = self._sigmoid_vector(o_raw, self.hidden_dim)

        self._free_ptr(i_raw)
        self._free_ptr(f_raw)
        self._free_ptr(g_raw)
        self._free_ptr(o_raw)

        fc = self._elementwise_mul(f_gate, self.lstm_c, self.hidden_dim)
        ig = self._elementwise_mul(i_gate, g_gate, self.hidden_dim)
        c_new = self._vector_add(fc, ig, self.hidden_dim)

        c_tanh = self._tanh_vector(c_new, self.hidden_dim)
        h_new = self._elementwise_mul(o_gate, c_tanh, self.hidden_dim)

        self._free_ptr(self.lstm_h)
        self._free_ptr(self.lstm_c)
        self.lstm_h = h_new
        self.lstm_c = c_new

        self._free_ptr(embedding_vec)
        self._free_ptr(ih_proj)
        self._free_ptr(hh_proj)
        self._free_ptr(gates)
        self._free_ptr(i_gate)
        self._free_ptr(f_gate)
        self._free_ptr(g_gate)
        self._free_ptr(o_gate)
        self._free_ptr(fc)
        self._free_ptr(ig)
        self._free_ptr(c_tanh)

        return self.lstm_h

    def _rule_head(self, hidden: loader.CUdeviceptr) -> loader.CUdeviceptr:
        vocab_size = self._get_rule_vocab_size()
        return self._matvec_add_bias(
            self.weights["rule_head_weight"],
            hidden,
            self.weights["rule_head_bias"],
            rows=vocab_size,
            cols=self.hidden_dim,
        )

    def _confidence_head(self, hidden: loader.CUdeviceptr) -> float:
        hidden_dim = self._get_confidence_hidden_dim()
        h1 = self._matvec_add_bias(
            self.weights["confidence_head_0_weight"],
            hidden,
            self.weights["confidence_head_0_bias"],
            rows=hidden_dim,
            cols=self.hidden_dim,
        )
        h1_relu = self._relu_vector(h1, hidden_dim)
        h2 = self._matvec_add_bias(
            self.weights["confidence_head_2_weight"],
            h1_relu,
            self.weights["confidence_head_2_bias"],
            rows=1,
            cols=hidden_dim,
        )
        conf_ptr = self._sigmoid_vector(h2, 1)
        try:
            conf_value = float(loader.gpu_to_cpu_array(conf_ptr, 1)[0])
        finally:
            self._free_ptr(h1)
            self._free_ptr(h1_relu)
            self._free_ptr(h2)
            self._free_ptr(conf_ptr)
        return conf_value

    def _argmax(self, logits: loader.CUdeviceptr, size: int) -> int:
        values = loader.gpu_to_cpu_array(logits, size)
        return int(values.argmax())

    def _get_rule_vocab_size(self) -> int:
        if self.rule_vocab_size is None:
            raise RuntimeError("Rule head weights not loaded.")
        return self.rule_vocab_size

    def _get_confidence_hidden_dim(self) -> int:
        if self.confidence_hidden_dim is None:
            raise RuntimeError("Confidence head weights not loaded.")
        return self.confidence_hidden_dim

    @staticmethod
    def _free_ptr(ptr: Optional[loader.CUdeviceptr]) -> None:
        if ptr is None:
            return
        try:
            loader.gpu_free(ptr)
        except Exception:
            pass

    def close(self) -> None:
        """Release GPU buffers owned by this instance."""
        for ptr in self.weights.values():
            try:
                loader.gpu_free(ptr)
            except Exception:
                pass
        self.weights.clear()
        for ptr in (self.lstm_h, self.lstm_c):
            if ptr is None:
                continue
            try:
                loader.gpu_free(ptr)
            except Exception:
                pass
        self.lstm_h = None
        self.lstm_c = None

    def cleanup(self) -> None:
        self.close()

    def _upload_to_gpu(self, array) -> loader.CUdeviceptr:
        """Upload a NumPy array to GPU memory (ingestion path)."""
        device_ptr = loader.gpu_malloc(array.nbytes)
        loader.cpu_to_gpu(device_ptr, array)
        return device_ptr

    @staticmethod
    def _load_numpy(path: Path):
        try:
            import numpy as np  # type: ignore
        except Exception as exc:  # pragma: no cover - numpy required for ingestion
            raise RuntimeError("NumPy is required to load Sovereign TRM weights") from exc
        return np.load(str(path))


__all__ = ["SovereignTRM"]
