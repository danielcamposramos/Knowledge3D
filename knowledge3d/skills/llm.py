from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class LLMConfig:
    backend: str = os.getenv("K3D_LLM_BACKEND", "transformers").strip()  # transformers|llama_cpp
    model: str = os.getenv("K3D_LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0").strip()
    url: str = ""  # no external endpoints
    max_tokens: int = int(os.getenv("K3D_LLM_MAX_TOKENS", "384"))
    # llama.cpp (GGUF) params
    n_gpu_layers: int = int(os.getenv("K3D_LLM_N_GPU_LAYERS", "0"))
    n_ctx: int = int(os.getenv("K3D_LLM_CTX", "2048"))
    # Optional PEFT base model for adapter directories
    peft_base: str = os.getenv("K3D_LLM_PEFT_BASE", "").strip()


class LLMSkill:
    def __init__(self, cfg: Optional[LLMConfig] = None) -> None:
        self.cfg = cfg or LLMConfig()
        self._llama = None  # llama-cpp handle (lazy)
        self._hf_model = None  # transformers model cache
        self._hf_tokenizer = None
        self._hf_model_key = None  # str identifying model or adapter path

    def set_backend(self, backend: str, model: Optional[str] = None, url: Optional[str] = None) -> None:
        self.cfg.backend = backend
        if model:
            self.cfg.model = model
        if url:
            self.cfg.url = url

    def configure_llama_cpp(self, n_gpu_layers: Optional[int] = None, n_ctx: Optional[int] = None) -> None:
        if n_gpu_layers is not None:
            self.cfg.n_gpu_layers = int(n_gpu_layers)
        if n_ctx is not None:
            self.cfg.n_ctx = int(n_ctx)

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        # Transformers-only integrated backend
        if self.cfg.backend == "llama_cpp":
            return self._gen_llama_cpp(prompt, system, max_tokens)
        return self._gen_transformers(prompt, system, max_tokens)

    def answer_with_rag(self, query: str, contexts: List[Tuple[str, str]], max_tokens: Optional[int] = None) -> str:
        sys = (
            "You are K3D's integrated LLM skill. You must answer using ONLY the provided "
            "contexts from the House memory. Do not invent facts. If information is missing, "
            "explicitly say 'I don't know' and suggest where to look next (label/room). "
            "When appropriate, cite labels you used in parentheses, e.g., (sources: labelA, labelB)."
        )
        ctx_lines = []
        for lab, txt in contexts:
            lab_s = str(lab)
            txt_s = str(txt or "")
            if len(txt_s) > 300:
                txt_s = txt_s[:297] + "..."
            ctx_lines.append(f"- {lab_s}: {txt_s}")
        prompt = (
            "Context (ground truth; use only this):\n" + "\n".join(ctx_lines) +
            f"\n\nQuestion: {query}\n\nAnswer (grounded, concise, with citations):"
        )
        return self.generate(prompt, system=sys, max_tokens=max_tokens)

    # --- Backend ---
    def _gen_transformers(self, prompt: str, system: Optional[str], max_tokens: Optional[int]) -> str:
        # Minimal local HF path (CPU/GPU). Assumes correct model path in cfg.model.
        try:
            import torch  # type: ignore
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
            from transformers.utils import is_bitsandbytes_available  # type: ignore
        except Exception as e:
            return f"[llm:transformers import error] {e}"
        try:
            import os as _os
            from pathlib import Path as _Path
            model_id_or_path = self.cfg.model
            adapter_path = _Path(model_id_or_path)
            use_peft = adapter_path.exists() and adapter_path.is_dir() and (adapter_path/"adapter_config.json").exists()
            base_model = None
            if use_peft:
                base_txt = (adapter_path/"base_model_id.txt")
                base_model = (self.cfg.peft_base or (base_txt.read_text(encoding="utf-8").strip() if base_txt.exists() else "")).strip()
                if not base_model:
                    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            quant = os.getenv("K3D_LLM_QUANT", "").lower().strip()
            quant_cfg = None
            kwargs = {}
            if quant in ("4bit", "4gb", "4") and is_bitsandbytes_available():
                try:
                    from transformers import BitsAndBytesConfig  # type: ignore
                    quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
                    kwargs["quantization_config"] = quant_cfg
                    kwargs["device_map"] = "auto"
                except Exception:
                    quant_cfg = None
            elif quant in ("8bit", "8") and is_bitsandbytes_available():
                try:
                    from transformers import BitsAndBytesConfig  # type: ignore
                    quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
                    kwargs["quantization_config"] = quant_cfg
                    kwargs["device_map"] = "auto"
                except Exception:
                    quant_cfg = None
            if quant_cfg is None:
                kwargs["torch_dtype"] = (torch.float16 if device == "cuda" else None)
            # Cache key
            model_key = (str(adapter_path.resolve()) if use_peft else model_id_or_path) + f"|q={quant if quant else 'none'}"
            if self._hf_model is None or self._hf_model_key != model_key:
                tok = AutoTokenizer.from_pretrained(base_model or model_id_or_path)
                if use_peft:
                    mdl = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
                    try:
                        from peft import PeftModel  # type: ignore
                        mdl = PeftModel.from_pretrained(mdl, str(adapter_path))
                    except Exception as e:
                        return f"[llm:transformers peft load error] {e}"
                    if quant_cfg is None:
                        mdl = mdl.to(device)
                else:
                    mdl = AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)
                    if quant_cfg is None:
                        mdl = mdl.to(device)
                self._hf_model = mdl
                self._hf_tokenizer = tok
                self._hf_model_key = model_key
            mdl = self._hf_model
            tok = self._hf_tokenizer
            full = (system + "\n\n" if system else "") + prompt
            ids = tok(full, return_tensors="pt").to(mdl.device)
            out = mdl.generate(**ids, max_new_tokens=int(max_tokens or self.cfg.max_tokens), do_sample=True, top_p=0.9, temperature=0.7)
            txt = tok.decode(out[0], skip_special_tokens=True)
            return txt[len(full):].strip() if txt.startswith(full) else txt
        except Exception as e:
            return f"[llm:transformers runtime error] {e}"

    def _gen_llama_cpp(self, prompt: str, system: Optional[str], max_tokens: Optional[int]) -> str:
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            return f"[llm:llama_cpp import error] {e}"
        try:
            if self._llama is None:
                # Load GGUF model from cfg.model
                params: Dict[str, Any] = {
                    "model_path": self.cfg.model,
                    "n_ctx": int(self.cfg.n_ctx),
                }
                if self.cfg.n_gpu_layers and self.cfg.n_gpu_layers > 0:
                    params["n_gpu_layers"] = int(self.cfg.n_gpu_layers)
                self._llama = Llama(**params)
            full = (system + "\n\n" if system else "") + prompt
            out = self._llama(full, max_tokens=int(max_tokens or self.cfg.max_tokens), temperature=0.7, top_p=0.9)
            txt = str(out.get("choices", [{}])[0].get("text", ""))
            return txt.strip()
        except Exception as e:
            return f"[llm:llama_cpp runtime error] {e}"
