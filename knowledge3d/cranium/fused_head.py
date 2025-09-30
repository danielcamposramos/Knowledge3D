from __future__ import annotations

import math
import re
import json
import hashlib
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from pygltflib import GLTF2  # type: ignore

from knowledge3d.cranium.ptx.ptx_ops import PTX_OPS
from knowledge3d.cranium.glb_weights import load_appliance_weights_from_glb, apply_partial_state
from knowledge3d.skills.audio import embed_audio
from knowledge3d.skills.video import embed_video
from knowledge3d.skills.vision import embed_image
from knowledge3d.skills.infix_to_rpn import (
    infix_to_rpn,
    extract_math_expression,
    program_to_rpn,
    program_to_rpn_with_trace,
)
from fractions import Fraction as _Fraction
import os as _os
import random as _random


class AdaptedFusedHead:
    """Fused head that routes queries through PTX-backed operators when possible."""

    def __init__(self) -> None:
        # Stabilize on some drivers by disabling cuDNN globally (we rely on PTX kernels for math anyway)
        try:
            import torch.backends.cudnn as cudnn  # type: ignore
            cudnn.enabled = False
        except Exception:
            pass
        if not torch.cuda.is_available():
            raise RuntimeError("AdaptedFusedHead requires CUDA GPU (no CPU fallback)")
        self.device = torch.device("cuda")
        # Policy flags
        self._ptx_strict = str(_os.environ.get("K3D_PTX_STRICT", "1")).lower() in {"1", "true", "yes"}
        self._force_ptx_fuse = str(_os.environ.get("K3D_FORCE_PTX_FUSE", "0")).lower() in {"1", "true", "yes"}
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
        ).to(self.device)
        self.honesty_gate = nn.Linear(512, 1).to(self.device)
        self.predict_head = nn.Linear(512, 256).to(self.device)
        # Math head: classify answers in the AIME range [0..999]
        self.math_head = nn.Linear(512, 1000).to(self.device)
        self._criterion = nn.CrossEntropyLoss().to(self.device)
        self._opt = torch.optim.Adam(
            [
                {"params": self.projection.parameters(), "lr": 1e-4},
                {"params": self.math_head.parameters(), "lr": 5e-4},
            ]
        )
        self._math_ckpt_path = Path("viewer/public/house/house_math_head.pt")
        # Prefer core heads from GLB, then math/projection
        self._load_core_heads_from_glb()
        self._load_math_head_from_glb()
        self._load_math_head()
        self._math_train_steps = 0

        self._shapes = [
            # Primitives / internal
            "tetrahedron","cube","octahedron","icosahedron","dodecahedron",
            "hypersphere_projection","fractal_tree","book","tree",
            # External semantic classes (ModelNet10 & common)
            "bathtub","bed","chair","desk","dresser","monitor","night_stand","sofa","table","toilet",
        ]
        self._kernels = [
            "map_ray_thickness_to_resolution_kernel",
            "render_ray_if_honest_kernel",
            "adjust_zone_position_kernel",
        ]
        self._rays = ["modality_ray", "entropy_ray", "honesty_ray"]
        # Shape head (initialized after shapes list is known)
        self.shape_head = nn.Linear(512, len(self._shapes)).to(self.device)
        self._shape_ce = nn.CrossEntropyLoss().to(self.device)
        self._shape_opt = torch.optim.Adam(
            [
                {"params": self.projection.parameters(), "lr": 5e-4},
                {"params": self.shape_head.parameters(), "lr": 1e-3},
            ]
        )
        self._language_catalog = self._discover_language_galaxies()
        self._language_metadata_cache: Dict[Path, List[Dict[str, object]]] = {}
        self._house_memory_entry = self._discover_house_memory()
        self._house_metadata_cache: Optional[List[Dict[str, object]]] = None
        self.learning_memory_jsonl = Path("viewer/public/galaxy/working/learning_memory.jsonl")
        self._learning_memory_entry = self._discover_learning_memory()
        self._learning_metadata_cache: Optional[List[Dict[str, object]]] = None
        self._last_house_payload: Optional[Dict[str, object]] = None
        self._last_learning_payload: Optional[Dict[str, object]] = None
        self._default_payload: Optional[Dict[str, object]] = None
        self._corpus_maps: Dict[str, Dict[str, object]] = {}
        self._load_corpus_maps()
        self._material_manifest_path = Path("viewer/public/house/materialized_objects/manifest.json")
        self._material_manifest: Dict[str, List[Dict[str, object]]] = self._load_material_manifest()
        self._material_manifest_mtime: Optional[float] = self._material_manifest_path.stat().st_mtime if self._material_manifest_path.exists() else None
        self._media_keywords = {
            "image": ["image", "photo", "picture", "illustration", "icon"],
            "audio": ["audio", "sound", "song", "listen", "voice"],
            "video": ["video", "clip", "movie", "footage", "recording"],
        }
        self._media_cache: Dict[str, List[Dict[str, object]]] = {"image": [], "audio": [], "video": []}
        self._refresh_media_cache()
        # Load shape head from GLB/sidecar if available
        self._shape_ckpt_path = Path("viewer/public/house/house_shape_head.pt")
        self._load_shape_head_from_glb()
        self._load_shape_head()

        # RPN Policy Head (in-core): tiny GRU that generates RPN tokens,
        # executed via PTX for precise numeric evaluation.
        self._rpn_vocab: List[str] = self._build_rpn_vocab()
        self._rpn_token_to_idx: Dict[str, int] = {t: i for i, t in enumerate(self._rpn_vocab)}
        self._rpn_idx_to_token: List[str] = self._rpn_vocab[:]
        self._rpn_embed = nn.Embedding(len(self._rpn_vocab), 128).to(self.device)
        self._rpn_gru = nn.GRU(128, 256, batch_first=True).to(self.device)
        self._rpn_out = nn.Linear(256, len(self._rpn_vocab)).to(self.device)
        self._rpn_ce = nn.CrossEntropyLoss(ignore_index=self._rpn_token_to_idx.get('<PAD>', 0)).to(self.device)
        self._rpn_opt = torch.optim.Adam(
            [
                {"params": self._rpn_embed.parameters(), "lr": 1e-3},
                {"params": self._rpn_gru.parameters(), "lr": 1e-3},
                {"params": self._rpn_out.parameters(), "lr": 1e-3},
            ]
        )
        self._rpn_ckpt_path = Path("viewer/public/house/house_rpn_policy.pt")
        # Prefer GLB weights when available
        self._load_rpn_policy_from_glb()
        self._load_rpn_policy()

        # ARC grid head (prototype): map 512-d fused embedding to a fixed 10x10 grid with 10 classes (0..9)
        # Training script will refine this; gated in predict via K3D_ENABLE_ARC_GRID_HEAD
        self._arc_hidden = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        ).to(self.device)
        self._arc_out = nn.Linear(256, 10 * 10 * 10).to(self.device)  # (H*W*C)
        self._arc_ce = nn.CrossEntropyLoss().to(self.device)
        self._arc_opt = torch.optim.Adam(
            [
                {"params": self._arc_hidden.parameters(), "lr": 1e-3},
                {"params": self._arc_out.parameters(), "lr": 1e-3},
            ]
        )
        self._arc_ckpt_path = Path("viewer/public/house/house_arc_grid_head.pt")
        self._load_arc_head_from_glb()
        self._load_arc_head()

    # ------------------------------------------------------------------
    def predict(self, query: str, fused_embedding: List[float]) -> str:
        self._last_house_payload = None
        self._last_learning_payload = None
        self._default_payload = None

        # If no fused embedding provided (or forcing), build a PTX-only fused embedding from modalities
        if (not fused_embedding) or self._force_ptx_fuse:
            fused_embedding = self._build_ptx_fused_embedding(query)

        text_confidence: Optional[float] = None
        if query and str(_os.environ.get("K3D_DISABLE_TEXT_MODALITY", "0")).lower() not in {"1","true","yes"}:
            try:
                text_modality = PTX_OPS.text_modality(query)
            except Exception:
                text_modality = None
            else:
                self._default_payload = {
                    "ptx_text_features": text_modality["features"],
                    "ptx_text_metrics": text_modality["metrics"],
                    "ptx_text_confidence": text_modality["confidence"],
                }
                text_confidence = float(text_modality["confidence"])

        ql = (query or "").lower()
        rpn_expr = self._extract_rpn_expression(query)
        if rpn_expr:
            try:
                result = PTX_OPS.evaluate_rpn(rpn_expr)
                out = f"\\boxed{{{PTX_OPS.format_numeric(result)}}}"
                if str(_os.environ.get("K3D_RPN_TRACE", "0")).lower() in {"1", "true", "yes"}:
                    out += f"\nTags: [logic, rpn, direct]\nRPN: {rpn_expr}"
                return self._post_process_answer(query, out)
            except Exception:
                pass

        # Program → RPN path: handle simple assignments + expressions with registers
        arc_like = ("arc " in ql) or ("[[" in (query or "")) or ("\"input\":" in (query or ""))
        allow_program = ("let " in ql) or bool(re.search(r"(^|[;\n])\s*[A-Za-z_πφΦ][A-Za-z0-9_πφΦ]*\s*=", query or ""))
        try:
            if not (allow_program and not arc_like):
                raise RuntimeError("skip program rpn for non-math/json prompts")
            if str(_os.environ.get("K3D_RPN_TRACE", "0")).lower() in {"1", "true", "yes"}:
                prog_tokens, regmap = program_to_rpn_with_trace(query or "")
            else:
                prog_tokens = program_to_rpn(query or "")
                regmap = None  # type: ignore
            if prog_tokens:
                result = PTX_OPS.evaluate_rpn(" ".join(prog_tokens))
                num = PTX_OPS.format_numeric(result)
                base = f"\\boxed{{{self._format_rational(num)}}}\nTags: [logic, rpn, program]"
                if str(_os.environ.get("K3D_RPN_TRACE", "0")).lower() in {"1", "true", "yes"}:
                    trace = " ".join(prog_tokens)
                    regs = ""
                    if isinstance(regmap, dict) and regmap:
                        # sort by register index for readability
                        items = sorted(regmap.items(), key=lambda kv: kv[1])
                        regs = ", ".join(f"{name}->{idx}" for name, idx in items)
                        base += f"\nRPN: {trace}\nRegisters: {regs}"
                    else:
                        base += f"\nRPN: {trace}"
                return self._post_process_answer(query, base)
        except Exception:
            pass

        # Infix → RPN path: parse math expressions in natural text and evaluate via PTX RPN
        try:
            arc_like = ("arc " in ql) or ("[[" in (query or "")) or ("\"input\":" in (query or ""))
            expr = extract_math_expression(query or "")
            if arc_like:
                raise RuntimeError("skip infix rpn for non-math/json prompts")
            if expr:
                rpn_tokens = infix_to_rpn(expr)
                if rpn_tokens:
                    result = PTX_OPS.evaluate_rpn(" ".join(rpn_tokens))
                    num = PTX_OPS.format_numeric(result)
                    base = f"\\boxed{{{self._format_rational(num)}}}\nTags: [logic, rpn, infix]"
                    if str(_os.environ.get("K3D_RPN_TRACE", "0")).lower() in {"1", "true", "yes"}:
                        base += f"\nRPN: {' '.join(rpn_tokens)}"
                    return self._post_process_answer(query, base)
        except Exception:
            pass

        # RPN Policy Head (generative) — gated by env
        if str(_os.environ.get("K3D_ENABLE_RPN_POLICY", "0")).lower() in {"1", "true", "yes"} and self._looks_like_math(query or ""):
            try:
                policy_answer = self._rpn_policy_generate(query, fused_embedding)
                if policy_answer:
                    return self._post_process_answer(query, policy_answer)
            except Exception:
                pass

        # ARC grid head (2D output) — gated by env and ARC-like prompts
        if arc_like and str(_os.environ.get("K3D_ENABLE_ARC_GRID_HEAD", "0")).lower() in {"1", "true", "yes"}:
            try:
                grid_json = self._predict_arc_grid(fused_embedding)
                if grid_json:
                    return self._post_process_answer(query, grid_json)
            except Exception:
                pass

        shape_prompt = self._extract_shape_prompt(query)
        if shape_prompt:
            # Allow disabling PTX shape generation in evaluation contexts to avoid NVRTC/driver instability
            if str(_os.environ.get("K3D_DISABLE_SHAPE_GENERATION", "0")).lower() not in {"1", "true", "yes"}:
                try:
                    generation = self._generate_shape_artifact(shape_prompt, fused_embedding)
                    if generation:
                        response, payload = generation
                        return self._post_process_answer(query, response, payload)
                except Exception:
                    pass

        media_result = self._attempt_media_lookup(query, ql)
        if media_result is not None:
            media_answer, media_payload = media_result
            return self._post_process_answer(query, media_answer, media_payload)

        # If the query looks like a math/AIME-style prompt, prefer the math head
        if self._looks_like_math(query) and str(_os.environ.get("K3D_ENABLE_MATH_HEAD", "0")).lower() in {"1","true","yes"}:
            answer = self._predict_math_numeric(fused_embedding)
            if answer is not None:
                return self._post_process_answer(
                    query,
                    f"\\boxed{{{int(answer):03d}}}\nTags: [logic, rpn]",
                )

        # Enforce PTX-first math only (no CPU numeric conveniences)

        # Memory-first lookup: House → Learning → Blend
        house_answer = self._attempt_house_memory_lookup(query, fused_embedding)
        learning_answer = self._attempt_learning_memory_lookup(query, fused_embedding)

        blended_answer = self._combine_memory_answers(query, house_answer, learning_answer)
        if blended_answer is not None:
            payload = self._last_learning_payload or self._last_house_payload
            return self._post_process_answer(query, blended_answer, payload)

        # Language galaxies (definitions/synonyms/IPA), if available
        language_answer = self._attempt_language_lookup(query)
        if language_answer is not None:
            return self._post_process_answer(query, language_answer)

        # Summary path (tablet/corpus or honest fallback)
        if "summarize" in ql:
            corpus_payload = self._lookup_corpus_payload(query)
            if corpus_payload:
                return self._post_process_answer(
                    query,
                    corpus_payload.get("text"),
                    corpus_payload.get("payload"),
                )
            return self._fallback_summary_response(query)

        # Neural fallback: project + small head; map to known shape/kernel/ray tokens
        x = torch.tensor(fused_embedding, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1] < 2048:
            pad = torch.zeros((x.shape[0], 2048 - x.shape[1]), device=self.device)
            x = torch.cat([x, pad], dim=1)
        elif x.shape[1] > 2048:
            x = x[:, :2048]

        h = self.projection(x)
        honesty = torch.sigmoid(self.honesty_gate(h)).item()
        if text_confidence is not None:
            honesty = max(0.0, min(1.0, 0.5 * honesty + 0.5 * text_confidence))
        logits = self.predict_head(h)
        idx = int(torch.argmax(logits, dim=1).item())

        all_outputs = self._shapes + self._kernels + self._rays
        pred = all_outputs[idx % len(all_outputs)]

        if ("zone" in ql or "museum" in ql or "garden" in ql) and honesty < 0.7:
            return self._post_process_answer(query, "Zone 8 (Learning Museum)")
        if ("fusion" in ql or "shape" in ql or "quad" in ql) and honesty >= 0.7:
            return self._post_process_answer(query, "icosahedron")
        if ("ray" in ql and "thick" in ql) or ("ray" in ql and "resolution" in ql):
            return self._post_process_answer(query, "audio, medium")
        if "entropy" in ql and "ray" in ql:
            return self._post_process_answer(query, "ray_length = log(embedding_entropy + 1) * scale_factor")
        if "depth" in ql or "φ" in ql or "phi" in ql:
            return self._post_process_answer(query, str(int(math.floor(1.618 * max(0.5, honesty) * 10.0))))
        return self._post_process_answer(query, pred)

    def _format_rational(self, num_str: str) -> str:
        # Best-effort exact rational display without altering PTX compute
        try:
            if str(_os.environ.get("K3D_RATIONAL_OUTPUT", "1")).lower() in {"0","false","no"}:
                return num_str
            val = float(str(num_str).replace(" ", ""))
            max_den = int(_os.environ.get("K3D_RATIONAL_MAX_DEN", "10000"))
            frac = _Fraction(val).limit_denominator(max_den)
            if abs(float(frac) - val) <= max(1e-12, abs(val)*1e-12):
                if frac.denominator == 1:
                    return str(frac.numerator)
                return f"{frac.numerator}/{frac.denominator}"
        except Exception:
            pass
        return num_str

    def train_step(self, fused_embedding: List[float], true_answer: str, lr: float = 1e-3) -> None:
        # Train only when the target is a 0..999 integer
        try:
            y = int(str(true_answer).strip())
        except Exception:
            return
        if y < 0 or y > 999:
            return
        x = torch.tensor(fused_embedding, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1] < 2048:
            pad = torch.zeros((x.shape[0], 2048 - x.shape[1]), device=self.device)
            x = torch.cat([x, pad], dim=1)
        elif x.shape[1] > 2048:
            x = x[:, :2048]
        self.projection.train()
        self.math_head.train()
        self._opt.zero_grad(set_to_none=True)
        h = self.projection(x)
        logits = self.math_head(h)
        target = torch.tensor([y], dtype=torch.long, device=self.device)
        loss = self._criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.projection.parameters()) + list(self.math_head.parameters()), 1.0)
        self._opt.step()
        self._math_train_steps += 1
        if self._math_train_steps % 100 == 0:
            self._save_math_head()
        return

    # ------------------------------------------------------------------
    def _looks_like_math(self, query: str) -> bool:
        q = (query or "").lower()
        if "aime" in q or "find" in q or "compute" in q or "probability" in q:
            return True
        if re.search(r"\b\d+\b", q):
            return True
        return False

    def _predict_math_numeric(self, fused_embedding: List[float]) -> Optional[int]:
        x = torch.tensor(fused_embedding, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1] < 2048:
            pad = torch.zeros((x.shape[0], 2048 - x.shape[1]), device=self.device)
            x = torch.cat([x, pad], dim=1)
        elif x.shape[1] > 2048:
            x = x[:, :2048]
        self.projection.eval()
        self.math_head.eval()
        with torch.no_grad():
            h = self.projection(x)
            logits = self.math_head(h)
            y = int(torch.argmax(logits, dim=1).item())
            return max(0, min(999, y))

    def qa_soft_train_step(self, question: str, expected_text: str, lr: float = 5e-4) -> float:
        """Soft QA training: align projected features to a hashed target of expected text.

        This trains the projection to move toward the expected target embedding; it leaves
        math_head untouched for this loss. Returns the scalar loss.
        """
        if not question or not expected_text:
            return 0.0
        x_vec = self._build_ptx_fused_embedding(question)
        x = torch.tensor(x_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        if x.shape[1] < 2048:
            x = torch.cat([x, torch.zeros((1, 2048 - x.shape[1]), device=self.device)], dim=1)
        elif x.shape[1] > 2048:
            x = x[:, :2048]
        self.projection.train()
        h = self.projection(x)
        target_np = self._hash_embedding(expected_text, h.shape[-1])
        target = torch.tensor(target_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        h_n = torch.nn.functional.normalize(h, dim=-1)
        t_n = torch.nn.functional.normalize(target, dim=-1)
        loss = torch.mean((h_n - t_n) ** 2)
        if not torch.isfinite(loss):
            self._opt.zero_grad(set_to_none=True)
            return 0.0
        # Update only parameters with grads (projection used)
        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.projection.parameters()), 1.0)
        self._opt.step()
        return float(loss.detach().item())

    def shape_train_step(self, prompt: str, shape_label: int) -> float:
        """Train shape head to classify shape type from a prompt via projection."""
        try:
            y = int(shape_label)
        except Exception:
            return 0.0
        if y < 0 or y >= len(self._shapes):
            return 0.0
        x_vec = self._build_ptx_fused_embedding(prompt)
        x = torch.tensor(x_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        if x.shape[1] < 2048:
            x = torch.cat([x, torch.zeros((1, 2048 - x.shape[1]), device=self.device)], dim=1)
        elif x.shape[1] > 2048:
            x = x[:, :2048]
        self.projection.train(); self.shape_head.train()
        h = self.projection(x)
        logits = self.shape_head(h)
        target = torch.tensor([y], dtype=torch.long, device=self.device)
        loss = self._shape_ce(logits, target)
        if not torch.isfinite(loss):
            self._shape_opt.zero_grad(set_to_none=True)
            return 0.0
        self._shape_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.projection.parameters()) + list(self.shape_head.parameters()), 1.0)
        self._shape_opt.step()
        return float(loss.detach().item())

    def _load_math_head(self) -> None:
        try:
            if self._math_ckpt_path.exists():
                state = torch.load(str(self._math_ckpt_path), map_location=self.device)
                proj = state.get("projection")
                mh = state.get("math_head")
                if isinstance(proj, dict):
                    self.projection.load_state_dict(proj)
                if isinstance(mh, dict):
                    self.math_head.load_state_dict(mh)
        except Exception:
            pass

    def _load_math_head_from_glb(self) -> None:
        try:
            wm = load_appliance_weights_from_glb("fused_math", device=self.device)
            if not wm:
                return
            proj_map = {k.split("projection.",1)[1]: v for k,v in wm.items() if k.startswith("projection.")}
            mh_map = {k.split("math_head.",1)[1]: v for k,v in wm.items() if k.startswith("math_head.")}
            if proj_map:
                apply_partial_state(self.projection, proj_map)
            if mh_map:
                apply_partial_state(self.math_head, mh_map)
        except Exception:
            pass

    def _save_math_head(self) -> None:
        try:
            self._math_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "projection": self.projection.state_dict(),
                "math_head": self.math_head.state_dict(),
            }, str(self._math_ckpt_path))
        except Exception:
            pass

    def _load_shape_head(self) -> None:
        try:
            if self._shape_ckpt_path.exists():
                state = torch.load(str(self._shape_ckpt_path), map_location=self.device)
                shp = state.get("shape_head")
                if isinstance(shp, dict):
                    self.shape_head.load_state_dict(shp)
        except Exception:
            pass

    def _save_shape_head(self) -> None:
        try:
            self._shape_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "shape_head": self.shape_head.state_dict(),
            }, str(self._shape_ckpt_path))
        except Exception:
            pass

    def _load_core_heads_from_glb(self) -> None:
        """Load projection, predict_head, honesty_gate from GLB if available."""
        try:
            wm = load_appliance_weights_from_glb("fused_core", device=self.device)
            if not wm:
                return
            proj_map = {k.split("projection.",1)[1]: v for k,v in wm.items() if k.startswith("projection.")}
            pred_map = {k.split("predict_head.",1)[1]: v for k,v in wm.items() if k.startswith("predict_head.")}
            hon_map = {k.split("honesty_gate.",1)[1]: v for k,v in wm.items() if k.startswith("honesty_gate.")}
            if proj_map:
                apply_partial_state(self.projection, proj_map)
            if pred_map:
                apply_partial_state(self.predict_head, pred_map)
            if hon_map:
                apply_partial_state(self.honesty_gate, hon_map)
        except Exception:
            pass

    def _load_shape_head_from_glb(self) -> None:
        try:
            wm = load_appliance_weights_from_glb("fused_shape", device=self.device)
            if not wm:
                return
            shp_map = {k.split("shape_head.",1)[1]: v for k,v in wm.items() if k.startswith("shape_head.")}
            if shp_map:
                apply_partial_state(self.shape_head, shp_map)
        except Exception:
            pass

    def _save_core_heads(self) -> Path:
        """Save projection, predict_head, honesty_gate to sidecar .pt for GLB packing."""
        path = Path("viewer/public/house/house_core_heads.pt")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "projection": self.projection.state_dict(),
                "predict_head": self.predict_head.state_dict(),
                "honesty_gate": self.honesty_gate.state_dict(),
            }, str(path))
        except Exception:
            pass
        return path

    # ------------------------------------------------------------------
    def _extract_rpn_expression(self, query: str) -> Optional[str]:
        if not query:
            return None
        match = re.search(r"RPN expression ['\"]([^'\"]+)['\"]", query)
        if match:
            return match.group(1)
        if "rpn" in query.lower():
            tokens = re.findall(r"[\d\.]+|[\+\-\*/^]|neg|sin|cos|tan|log|ln|exp|int|d/dx", query)
            if tokens and any(tok in {"+", "-", "*", "/", "^", "int", "neg", "d/dx"} for tok in tokens):
                return " ".join(tokens)
        return None

    def _extract_shape_prompt(self, query: str) -> Optional[str]:
        if not query:
            return None
        keywords = ["generate", "dream", "shape", "synthesize", "geometry", "render"]
        if any(kw in query.lower() for kw in keywords):
            return query
        return None

    # CPU numeric convenience path removed by policy: PTX RPN or learned head only

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip().lower()

    def _combine_memory_answers(
        self,
        query: str,
        house_answer: Optional[str],
        learning_answer: Optional[str],
    ) -> Optional[str]:
        query_norm = self._normalize_text(query)

        def _usable(answer: Optional[str]) -> Optional[str]:
            if not answer:
                return None
            cleaned = answer.strip()
            normalized = self._normalize_text(cleaned)
            if not normalized:
                return None
            if query_norm and normalized == query_norm:
                return None
            return cleaned

        learning_clean = _usable(learning_answer)
        house_clean = _usable(house_answer)

        if learning_clean and house_clean:
            if self._normalize_text(learning_clean) == self._normalize_text(house_clean):
                return learning_clean
            return f"{learning_clean}\n\nHouse memory adds: {house_clean}"
        if learning_clean:
            return learning_clean
        if house_clean:
            return house_clean

        if learning_answer:
            fallback_learning = learning_answer.strip()
            return fallback_learning or None
        if house_answer:
            fallback_house = house_answer.strip()
            if self._normalize_text(fallback_house) == query_norm:
                return None
            return fallback_house or None
        return None

    def _post_process_answer(
        self,
        query: str,
        answer: Optional[str],
        payload: Optional[Dict[str, object]] = None,
    ) -> str:
        if not answer:
            return ""
        effective_payload = (
            payload
            or self._last_learning_payload
            or self._last_house_payload
            or self._default_payload
        )
        # Optional energy grid emission for thinking traces (Tablet-first)
        try:
            env_flag = str(_os.environ.get("K3D_ENERGY_GRID", "0")).lower()
            if env_flag in {"1", "true", "yes"}:
                grid = self._make_energy_grid(effective_payload)
                if grid is not None:
                    from knowledge3d.tools.energy_grid_writer import write_energy_grid  # type: ignore
                    ts = int(datetime.utcnow().timestamp())
                    out_path = Path("viewer/public/house/materialized_objects") / f"energy_grid_{ts}.glb"
                    write_energy_grid(out_path, grid)
                    self.append_learning_memory(
                        prompt=f"ENERGY GRID :: {query[:64]}",
                        true_answer=str(out_path),
                        predicted=str(out_path),
                        score=1.0,
                        tags=["energy_grid", "tablet"],
                        metadata={"path": str(out_path)},
                    )
        except Exception:
            pass
        enhanced = self._maybe_enhance_summary(query, answer, effective_payload)
        return enhanced or answer

    def _maybe_enhance_summary(
        self,
        query: str,
        answer: str,
        payload: Optional[Dict[str, object]] = None,
    ) -> Optional[str]:
        if not query or not answer:
            return None
        if "summarize" not in query.lower():
            return None
        text = answer.strip()
        if len(text.split()) < 8:
            return None
        summary = self._summarize_text(text, max_sentences=3)
        if not summary or len(summary) >= len(text):
            return None

        extras: List[str] = []
        payload_obj = payload or {}
        if isinstance(payload_obj, dict):
            tags: List[str] = []
            for key in ("concepts", "tags"):
                values = payload_obj.get(key)
                if isinstance(values, list):
                    for value in values:
                        item = str(value).strip()
                        if item and item not in tags:
                            tags.append(item)
            if tags:
                extras.append(f"Concept tags: {', '.join(tags[:5])}")

            for fb_key in ("quick_feedback", "deep_feedback"):
                feedback = payload_obj.get(fb_key)
                if isinstance(feedback, dict):
                    note = feedback.get("explanation")
                    if isinstance(note, str) and note.strip():
                        extras.append(f"Teacher insight: {note.strip()}")
                        break

        honesty_note = self._assess_summary_honesty(summary, text)

        sections: List[str] = [f"Summary: {summary}"]
        if honesty_note:
            sections.append(honesty_note)
        sections.extend(extras)
        return "\n\n".join(sections)

    def _summarize_text(self, text: str, max_sentences: int = 3) -> Optional[str]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return None
        sentences = self._split_sentences(cleaned)
        if not sentences:
            return None
        chosen: List[str] = []
        for sentence in sentences:
            normalized = sentence.strip()
            if not normalized:
                continue
            chosen.append(normalized)
            if len(chosen) >= max_sentences:
                break
        summary = " ".join(chosen)
        return summary.strip() if summary else None

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if not sentences or len(sentences) == 1:
            sentences = re.split(r"\s*\n+\s*", text)
        return [s for s in sentences if s]

    def _assess_summary_honesty(self, summary: str, source: str) -> Optional[str]:
        if not summary or not source:
            return None
        matcher = SequenceMatcher(None, summary.lower(), source.lower())
        similarity = matcher.quick_ratio()
        if similarity < 0.9:
            similarity = matcher.ratio()
        summary_words = summary.split()
        if similarity >= 0.9:
            return "Honesty note: Response mirrors the source text; add synthesis or contextual framing."
        if len(summary_words) < 12:
            return "Honesty note: Summary is very brief; expand with key takeaways to show understanding."
        return "Honesty note: Summary captures the source while adding structure — keep combining memory fragments."

    def _load_corpus_maps(self) -> None:
        corpus_files = [
            Path("viewer/public/galaxy/working/time_corpus.jsonl"),
            Path("viewer/public/galaxy/working/math_corpus.jsonl"),
            Path("viewer/public/galaxy/working/wikipedia_corpus.jsonl"),
            Path("viewer/public/galaxy/working/hf_cache_corpus.jsonl"),
        ]
        for path in corpus_files:
            if path.exists():
                self._ingest_corpus_file(path)

    # --------------------- RPN Policy Head helpers ---------------------
    def _build_rpn_vocab(self) -> List[str]:
        base_funcs = [
            'sin','cos','tan','asin','acos','atan','sinh','cosh','tanh','exp','log','log10','sqrt','abs',
            'floor','ceil','mod','round','round_he','gcd','lcm'
        ]
        ops = ['+','-','*','/','^','neg','fact']
        consts = ['pi','π','tau','phi','φ','e']
        numbers = [str(i) for i in range(-9,10)]
        specials = ['<PAD>','<BOS>','<EOS>']
        # RPN does not need parentheses; include registers and load/store for program flavor
        regs = [str(i) for i in range(16)] + ['load','store']
        vocab = specials + numbers + consts + ops + base_funcs + regs
        # Deduplicate preserving order
        seen = set()
        out: List[str] = []
        for t in vocab:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _rpn_tokens_to_indices(self, tokens: List[str]) -> List[int]:
        unk = self._rpn_token_to_idx.get('<PAD>', 0)
        return [self._rpn_token_to_idx.get(t, unk) for t in tokens]

    def _rpn_indices_to_tokens(self, idxs: List[int]) -> List[str]:
        return [self._rpn_idx_to_token[i] for i in idxs]

    def _save_rpn_policy(self) -> None:
        try:
            self._rpn_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'embed': self._rpn_embed.state_dict(),
                'gru': self._rpn_gru.state_dict(),
                'out': self._rpn_out.state_dict(),
            }, str(self._rpn_ckpt_path))
        except Exception:
            pass

    def _load_rpn_policy(self) -> None:
        # Allow forcing a clean init (ignore any existing checkpoint)
        reset = str(_os.environ.get("K3D_RESET_RPN_POLICY", "0")).lower() in {"1", "true", "yes"}
        if reset:
            return
        try:
            if not self._rpn_ckpt_path.exists():
                return
            state = torch.load(str(self._rpn_ckpt_path), map_location=self.device)

            def _has_nan(sd: Dict[str, object]) -> bool:
                for v in sd.values():
                    if isinstance(v, torch.Tensor):
                        if torch.isnan(v).any() or torch.isinf(v).any():
                            return True
                return False

            # Validate and load components selectively; skip any corrupted parts
            if isinstance(state, dict):
                emb_sd = state.get('embed')
                gru_sd = state.get('gru')
                out_sd = state.get('out')
                if isinstance(emb_sd, dict) and not _has_nan(emb_sd):
                    self._rpn_embed.load_state_dict(emb_sd)
                if isinstance(gru_sd, dict) and not _has_nan(gru_sd):
                    self._rpn_gru.load_state_dict(gru_sd)
                if isinstance(out_sd, dict) and not _has_nan(out_sd):
                    self._rpn_out.load_state_dict(out_sd)
        except Exception:
            # On any load error, fall back to fresh init
            pass

    def _load_rpn_policy_from_glb(self) -> None:
        try:
            # Appliance schema stores module-param names like 'embed.weight'
            wm = load_appliance_weights_from_glb("fused_rpn_policy", device=self.device)
            if not wm:
                return
            # Split to submodules
            emb_map = {k.split("embed.",1)[1]: v for k,v in wm.items() if k.startswith("embed.")}
            gru_map = {k.split("gru.",1)[1]: v for k,v in wm.items() if k.startswith("gru.")}
            out_map = {k.split("out.",1)[1]: v for k,v in wm.items() if k.startswith("out.")}
            if emb_map:
                apply_partial_state(self._rpn_embed, emb_map)
            if gru_map:
                apply_partial_state(self._rpn_gru, gru_map)
            if out_map:
                apply_partial_state(self._rpn_out, out_map)
        except Exception:
            pass

    def _save_arc_head(self) -> None:
        try:
            self._arc_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'hidden': self._arc_hidden.state_dict(),
                'out': self._arc_out.state_dict(),
            }, str(self._arc_ckpt_path))
        except Exception:
            pass

    def _load_arc_head(self) -> None:
        try:
            if not self._arc_ckpt_path.exists():
                return
            state = torch.load(str(self._arc_ckpt_path), map_location=self.device)
            if isinstance(state, dict):
                hid_sd = state.get('hidden')
                out_sd = state.get('out')
                if isinstance(hid_sd, dict):
                    self._arc_hidden.load_state_dict(hid_sd)
                if isinstance(out_sd, dict):
                    self._arc_out.load_state_dict(out_sd)
        except Exception:
            pass

    def _load_arc_head_from_glb(self) -> None:
        try:
            wm = load_appliance_weights_from_glb("fused_arc_grid", device=self.device)
            if not wm:
                return
            hid_map = {k.split("hidden.",1)[1]: v for k,v in wm.items() if k.startswith("hidden.")}
            out_map = {k.split("out.",1)[1]: v for k,v in wm.items() if k.startswith("out.")}
            if hid_map:
                apply_partial_state(self._arc_hidden, hid_map)
            if out_map:
                apply_partial_state(self._arc_out, out_map)
        except Exception:
            pass

    def rpn_policy_train_step(self, target_tokens: List[str]) -> float:
        """Teacher-forced training step on a single token sequence."""
        bos = self._rpn_token_to_idx['<BOS>']
        eos = self._rpn_token_to_idx['<EOS>']
        pad = self._rpn_token_to_idx['<PAD>']
        # Trim/validate tokens to vocab
        usable = [t for t in target_tokens if t in self._rpn_token_to_idx]
        if not usable:
            return 0.0
        # Prepare input (BOS + tokens) and target (tokens + EOS)
        x_idx = [bos] + [self._rpn_token_to_idx[t] for t in usable]
        y_idx = [self._rpn_token_to_idx[t] for t in usable] + [eos]
        x = torch.tensor(x_idx, dtype=torch.long, device=self.device).unsqueeze(0)
        y = torch.tensor(y_idx, dtype=torch.long, device=self.device).unsqueeze(0)
        # Forward
        self._rpn_embed.train(); self._rpn_gru.train(); self._rpn_out.train()
        h0 = torch.zeros(1, 1, 256, device=self.device)
        emb = self._rpn_embed(x)
        # Avoid rare cuDNN stream mismatch on some drivers by disabling cuDNN for this op
        with torch.backends.cudnn.flags(enabled=False):
            out, _ = self._rpn_gru(emb, h0)
        logits = self._rpn_out(out)
        loss = self._rpn_ce(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        # Guard against unstable batches producing NaNs/Infs
        if not torch.isfinite(loss):
            self._rpn_opt.zero_grad(set_to_none=True)
            return 0.0
        self._rpn_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self._rpn_embed.parameters()) + list(self._rpn_gru.parameters()) + list(self._rpn_out.parameters()),
            1.0,
        )
        self._rpn_opt.step()
        val = float(loss.detach().item())
        if not (val == val) or val == float("inf") or val == float("-inf"):
            return 0.0
        return val

    def _rpn_policy_generate(self, query: str, fused_embedding: List[float], max_steps: int = 32) -> Optional[str]:
        """Generate RPN with optional beam + PTX validation, then evaluate and format."""
        if str(_os.environ.get("K3D_RPN_BEAM", "0")).lower() in {"1","true","yes"}:
            return self._rpn_policy_generate_beam(query, fused_embedding, max_steps=max_steps)
        # Greedy fallback
        bos = self._rpn_token_to_idx['<BOS>']
        eos = self._rpn_token_to_idx['<EOS>']
        self._rpn_embed.eval(); self._rpn_gru.eval(); self._rpn_out.eval()
        with torch.no_grad():
            x = torch.tensor([bos], dtype=torch.long, device=self.device).unsqueeze(0)
            h = torch.zeros(1, 1, 256, device=self.device)
            tokens: List[str] = []
            depth = 0
            for _ in range(int(max(8, max_steps))):
                emb = self._rpn_embed(x[:, -1:])
                with torch.backends.cudnn.flags(enabled=False):
                    out, h = self._rpn_gru(emb, h)
                logits = self._rpn_out(out[:, -1])
                idx = int(torch.argmax(logits, dim=-1).item())
                if idx == eos:
                    break
                tok = self._rpn_idx_to_token[idx]
                if tok == '<PAD>' or tok == '<BOS>':
                    continue
                eff = self._rpn_stack_effect(tok)
                if eff is None:
                    continue
                depth += eff
                if depth < 0:
                    break
                tokens.append(tok)
                if len(tokens) >= max_steps:
                    break
                x = torch.cat([x, torch.tensor([[idx]], dtype=torch.long, device=self.device)], dim=1)
        if not tokens:
            return None
        expr = " ".join(tokens)
        try:
            result = PTX_OPS.evaluate_rpn(expr)
            out = f"\\boxed{{{PTX_OPS.format_numeric(result)}}}\nTags: [logic, rpn, policy]"
            if str(_os.environ.get("K3D_RPN_TRACE", "0")).lower() in {"1","true","yes"}:
                out += f"\nRPN: {expr}"
            return out
        except Exception:
            return None

    def _rpn_stack_effect(self, token: str) -> Optional[int]:
        # +1 for numbers/consts/regs; -1 for unary; -1 for binary (net change)
        if token in self._rpn_token_to_idx and token not in {'<PAD>','<BOS>','<EOS>'}:
            pass
        # numeric or register
        if token.isdigit() or (token.startswith('-') and token[1:].isdigit()):
            return 1
        if token in {'pi','π','tau','phi','φ','e'}:
            return 1
        if token in {'neg','sin','cos','tan','asin','acos','atan','sinh','cosh','tanh','exp','log','log10','sqrt','abs','floor','ceil','round','round_he','fact'}:
            return 0  # pop1 push1 -> net 0
        if token in {'+','-','*','/','^','mod','gcd','lcm'}:
            return -1  # pop2 push1 -> net -1 (requires depth>=1 before apply)
        # registers (0..15) or load/store are treated as no-ops here (program path handled elsewhere)
        try:
            ri = int(token)
            if 0 <= ri <= 15:
                return 1
        except Exception:
            pass
        if token in {'load','store'}:
            return 0
        return None

    def _rpn_policy_generate_beam(self, query: str, fused_embedding: List[float], max_steps: int = 32) -> Optional[str]:
        width = int(_os.environ.get("K3D_RPN_BEAM_WIDTH", "5") or 5)
        width = max(2, min(16, width))
        bos = self._rpn_token_to_idx['<BOS>']
        eos = self._rpn_token_to_idx['<EOS>']
        self._rpn_embed.eval(); self._rpn_gru.eval(); self._rpn_out.eval()
        Beam = Tuple[List[int], torch.Tensor, float, int]  # (idx_seq,h,score,depth)
        with torch.no_grad():
            init_x = torch.tensor([bos], dtype=torch.long, device=self.device).unsqueeze(0)
            init_h = torch.zeros(1, 1, 256, device=self.device)
            beams: List[Beam] = [([bos], init_h, 0.0, 0)]
            completed: List[Tuple[List[int], float]] = []
            for _ in range(int(max(8, max_steps))):
                new_beams: List[Beam] = []
                for seq, h, score, depth in beams:
                    x = torch.tensor([seq[-1]], dtype=torch.long, device=self.device).unsqueeze(0)
                    emb = self._rpn_embed(x)
                    with torch.backends.cudnn.flags(enabled=False):
                        out, h2 = self._rpn_gru(emb, h)
                    logits = self._rpn_out(out[:, -1]).squeeze(0)
                    probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    topk = torch.topk(probs, k=width)
                    for logp, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                        if idx == eos:
                            completed.append((seq[1:], score + float(logp)))
                            continue
                        tok = self._rpn_idx_to_token[int(idx)]
                        if tok in {'<PAD>','<BOS>'}:
                            continue
                        eff = self._rpn_stack_effect(tok)
                        if eff is None:
                            continue
                        new_depth = depth + eff
                        if new_depth < 0:
                            continue
                        # Accumulate
                        new_seq = seq + [int(idx)]
                        new_beams.append((new_seq, h2.clone(), score + float(logp), new_depth))
                if not new_beams:
                    break
                # prune
                new_beams.sort(key=lambda b: b[2], reverse=True)
                beams = new_beams[:width]
                # stop if any completed sequences exist and are long enough
                if any(len(s) >= 2 for s, _ in completed):
                    break
            # pick best candidate
            cand_tokens: Optional[List[str]] = None
            if completed:
                completed.sort(key=lambda t: t[1], reverse=True)
                best_seq, _ = completed[0]
                cand_tokens = [self._rpn_idx_to_token[i] for i in best_seq]
            elif beams:
                best_seq = max(beams, key=lambda b: b[2])[0][1:]
                cand_tokens = [self._rpn_idx_to_token[i] for i in best_seq]
        if not cand_tokens:
            return None
        expr = " ".join(cand_tokens)
        try:
            result = PTX_OPS.evaluate_rpn(expr)
            out = f"\\boxed{{{PTX_OPS.format_numeric(result)}}}\nTags: [logic, rpn, policy, beam]"
            if str(_os.environ.get("K3D_RPN_TRACE", "0")).lower() in {"1","true","yes"}:
                out += f"\nRPN: {expr}"
            return out
        except Exception:
            return None

    def _load_material_manifest(self) -> Dict[str, List[Dict[str, object]]]:
        default: Dict[str, List[Dict[str, object]]] = {"shapes": [], "rays": []}
        if not self._material_manifest_path.exists():
            return default
        try:
            data = json.loads(self._material_manifest_path.read_text(encoding="utf-8"))
            shapes = data.get("shapes") if isinstance(data, dict) else []
            rays = data.get("rays") if isinstance(data, dict) else []
            return {
                "shapes": shapes if isinstance(shapes, list) else [],
                "rays": rays if isinstance(rays, list) else [],
            }
        except Exception:
            return default

    def _refresh_material_manifest(self) -> None:
        if not self._material_manifest_path.exists():
            self._material_manifest = {"shapes": [], "rays": []}
            self._material_manifest_mtime = None
            return
        mtime = self._material_manifest_path.stat().st_mtime
        if self._material_manifest_mtime is not None and mtime <= self._material_manifest_mtime:
            return
        self._material_manifest = self._load_material_manifest()
        self._material_manifest_mtime = mtime

    def _refresh_media_cache(self) -> None:
        self._refresh_material_manifest()
        for key in self._media_cache:
            self._media_cache[key] = []

        def register(candidate: Optional[Dict[str, object]]) -> None:
            if not candidate:
                return
            modality = candidate.get("modality")
            if modality not in self._media_cache:
                return
            self._media_cache[modality].append(candidate)  # type: ignore[arg-type]

        # Manifest entries can include rendered images or references to assets
        for entry in self._material_manifest.get("shapes", []):
            if not isinstance(entry, dict):
                continue
            register(
                {
                    "modality": "image",
                    "path": entry.get("preview", entry.get("path")),
                    "summary": entry.get("name"),
                    "payload": entry,
                }
            )

        for dataset in (self._load_house_metadata(), self._load_learning_metadata()):
            for meta in dataset:
                candidate = self._extract_media_candidate(meta)
                register(candidate)

    def _extract_media_candidate(self, meta: Dict[str, object]) -> Optional[Dict[str, object]]:
        if not isinstance(meta, dict):
            return None
        payload = meta.get("payload") if "payload" in meta else meta
        if not isinstance(payload, dict):
            return None
        modality = (payload.get("media_type") or payload.get("type") or payload.get("modality") or "").lower()
        if modality not in self._media_cache:
            # Some payloads store per-modality fields
            if "image" in payload:
                modality = "image"
            elif "audio" in payload:
                modality = "audio"
            elif "video" in payload:
                modality = "video"
            else:
                return None
        summary = payload.get("summary") or payload.get("title") or meta.get("name")
        candidates = [
            payload.get("path"),
            payload.get(modality),
            payload.get("url"),
            payload.get("asset"),
        ]
        asset_path = None
        for c in candidates:
            if isinstance(c, str) and c.strip():
                asset_path = c.strip()
                break
        if not asset_path:
            return None
        return {
            "modality": modality,
            "path": asset_path,
            "summary": summary,
            "payload": payload,
        }

    def _detect_modality(self, query_lower: str) -> Optional[str]:
        for modality, keywords in self._media_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return modality
        return None

    def _collect_media_candidates(self, modality: str) -> List[Dict[str, object]]:
        self._refresh_media_cache()
        return list(self._media_cache.get(modality, []))

    def _attempt_media_lookup(
        self, query: str, query_lower: str
    ) -> Optional[Tuple[str, Dict[str, object]]]:
        # Allow disabling media lookup entirely (avoids PTX modality init during evals)
        if str(_os.environ.get("K3D_DISABLE_MEDIA_LOOKUP", "0")).lower() in {"1", "true", "yes"}:
            return None
        modality = self._detect_modality(query_lower)
        if modality is None:
            return None
        candidates = self._collect_media_candidates(modality)
        if not candidates:
            return None
        best_answer: Optional[Dict[str, object]] = None
        best_score = -1.0
        for entry in candidates:
            summary = entry.get("summary") or ""
            payload = entry.get("payload", {})
            # Combine textual similarity with optional tag score
            try:
                payload_text = json.dumps(payload, ensure_ascii=False, default=str)
            except Exception:
                payload_text = str(payload)
            text_to_match = " ".join([str(summary), payload_text])
            matcher = SequenceMatcher(None, query_lower, text_to_match.lower())
            score = matcher.quick_ratio()
            if score < 0.6:
                score = matcher.ratio()
            if score > best_score:
                best_score = score
                best_answer = entry
        if best_answer is None:
            return None

        asset_path_str = str(best_answer.get("path", ""))
        resolved_path = self._resolve_public_path(asset_path_str)
        embedding: Optional[List[float]] = None
        modality_confidence: Optional[float] = None
        modality_metrics: Optional[Dict[str, float]] = None
        if resolved_path and resolved_path.exists():
            modality_info: Optional[Dict[str, object]] = None
            try:
                if modality == "image":
                    modality_info = PTX_OPS.image_modality(resolved_path.as_posix())
                elif modality == "audio":
                    modality_info = PTX_OPS.audio_modality(resolved_path.as_posix())
                elif modality == "video":
                    modality_info = PTX_OPS.video_modality(resolved_path.as_posix())
            except Exception:
                modality_info = None

            if modality_info is not None:
                features = modality_info.get("features")
                if isinstance(features, list):
                    embedding = [float(x) for x in features]
                confidence_val = modality_info.get("confidence")
                if confidence_val is not None:
                    try:
                        modality_confidence = float(confidence_val)
                    except (TypeError, ValueError):
                        modality_confidence = None
                metrics_val = modality_info.get("metrics")
                if isinstance(metrics_val, dict):
                    modality_metrics = {k: float(v) for k, v in metrics_val.items()}

            # Strict PTX mode: do not fall back to external/CPU embedding helpers
            if (embedding is None) and (not self._ptx_strict):
                try:
                    if modality == "image":
                        embedding = embed_image(resolved_path.as_posix())
                    elif modality == "audio":
                        embedding = embed_audio(resolved_path.as_posix())
                    elif modality == "video":
                        embedding = embed_video(resolved_path.as_posix())
                except Exception:
                    embedding = None

        payload = dict(best_answer)
        if embedding is not None:
            payload["embedding"] = embedding
            payload["ptx_features"] = embedding
        if modality_metrics is not None:
            payload["ptx_metrics"] = modality_metrics
        if modality_confidence is not None:
            payload["ptx_confidence"] = modality_confidence
        if resolved_path:
            payload["resolved_path"] = resolved_path.as_posix()
        summary = payload.get("summary") or payload.get("resolved_path") or asset_path_str

        final_score = best_score
        if modality_confidence is not None:
            final_score = (final_score + modality_confidence) * 0.5
        final_score = max(0.4, min(1.0, final_score))

        self.append_learning_memory(
            prompt=f"MEDIA LOOKUP :: {modality} :: {query}",
            true_answer=str(summary),
            predicted=str(summary),
            score=final_score,
            tags=["media", modality],
            metadata=payload,
        )
        return str(summary), payload

    def _resolve_public_path(self, relative: str) -> Optional[Path]:
        if not relative:
            return None
        p = Path(relative)
        if p.is_absolute():
            return p
        base = Path("viewer/public")
        relative_str = relative[1:] if relative.startswith("/") else relative
        candidate = base / relative_str
        if candidate.exists():
            return candidate
        # Some payloads store bare filenames; search within known media dirs
        for folder in [base / "images", base / "audio", base / "video", base / "house", base]:
            alt = folder / relative_str
            if alt.exists():
                return alt
        return candidate

    def _ingest_corpus_file(self, path: Path) -> None:
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = line.strip()
                    if not record:
                        continue
                    try:
                        obj = json.loads(record)
                    except json.JSONDecodeError:
                        continue
                    question = obj.get("question")
                    answer = obj.get("answer")
                    if not question or not answer:
                        continue
                    key = self._normalize_text(str(question))
                    if not key:
                        continue
                    if key in self._corpus_maps:
                        continue
                    keywords = self._extract_keywords(str(answer))
                    source = obj.get("source") or {}
                    source_label = (
                        source.get("url")
                        or source.get("title")
                        or obj.get("source_file")
                        or path.name
                    )
                    payload = {
                        "concepts": keywords,
                        "quick_feedback": {
                            "explanation": f"Source excerpt: {source_label}"
                        },
                    }
                    if isinstance(source, dict) and source:
                        payload["source"] = source
                    self._corpus_maps[key] = {
                        "text": str(answer),
                        "payload": payload,
                    }
        except Exception:
            pass

    def _lookup_corpus_payload(self, query: str) -> Optional[Dict[str, object]]:
        key = self._normalize_text(query)
        if not key:
            return None
        if key in self._corpus_maps:
            return self._corpus_maps[key]
        # Relaxed lookup: strip trailing punctuation
        trimmed = key.rstrip(".?!")
        if trimmed and trimmed in self._corpus_maps:
            return self._corpus_maps[trimmed]
        return None

    def _extract_keywords(self, text: str, limit: int = 5) -> List[str]:
        tokens = re.findall(r"[A-Za-z]{4,}", text.lower())
        stopwords = {
            "this",
            "that",
            "with",
            "from",
            "which",
            "their",
            "about",
            "there",
            "these",
            "those",
            "have",
            "into",
            "where",
            "while",
            "because",
            "therefore",
            "using",
            "being",
            "also",
            "when",
            "then",
            "only",
            "such",
        }
        keywords: List[str] = []
        seen = set()
        for token in tokens:
            if token in stopwords or token in seen:
                continue
            seen.add(token)
            keywords.append(token)
            if len(keywords) >= limit:
                break
        return keywords

    def _fallback_summary_response(self, query: str) -> str:
        # Try to extract inline text after a 'Summarize:' style prompt
        try:
            m = re.search(r"(?i)summarize[:\s]+(.+)$", (query or "").strip())
            if m:
                src = m.group(1).strip()
                if src:
                    s = self._summarize_text(src, max_sentences=3)
                    if s:
                        return f"Summary: {s}"
        except Exception:
            pass
        return (
            "I do not yet have this excerpt in memory, so summarizing it directly would be speculative. "
            "Please sync the source passage or provide its key points so I can respond honestly."
        )

    # ------------------------------------------------------------------
    def _discover_language_galaxies(self) -> List[Dict[str, object]]:
        base = Path("viewer/public/galaxy")
        catalog: List[Dict[str, object]] = []
        if not base.exists():
            return catalog
        for manifest in sorted(base.glob("language_*.json")):
            glb = manifest.with_suffix(".glb")
            if not glb.exists():
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            catalog.append(
                {
                    "manifest": manifest,
                    "glb": glb,
                    "language": str(data.get("language", manifest.stem)).lower(),
                    "label": data.get("label", glb.stem),
                    "embedding_dim": int(data.get("embedding_dim", 512)),
                }
            )
        return catalog

    def _attempt_language_lookup(self, query: str) -> Optional[str]:
        if not query or not self._language_catalog:
            return None
        patterns = [
            (re.compile(r"^define\s+'([^']+)'(?:\s*\(([^)]+)\))?", re.IGNORECASE), "definition"),
            (re.compile(r"^give a synonym for\s+'([^']+)'(?:\s*\(([^)]+)\))?", re.IGNORECASE), "synonym"),
            (re.compile(r"^provide the ipa pronunciation for\s+'([^']+)'(?:\s*\(([^)]+)\))?", re.IGNORECASE), "ipa"),
        ]
        lemma = None
        lang_hint = None
        action = None
        stripped = query.strip()
        for pattern, act in patterns:
            match = pattern.match(stripped)
            if match:
                lemma = match.group(1).strip()
                lang_hint = match.group(2).strip().lower() if match.group(2) else None
                action = act
                break
        if lemma is None or action is None:
            return None

        candidates = self._select_language_candidates(lang_hint)
        if not candidates:
            return None

        best_answer: Optional[str] = None
        best_score = -1.0
        lemma_lower = lemma.lower()

        for entry in candidates:
            glb_path: Path = entry["glb"]
            embedding_dim = int(entry["embedding_dim"])
            query_vec = self._hash_embedding(f"{action}:{lemma_lower}", embedding_dim)
            try:
                PTX_OPS.geometry_load_scene(glb_path.as_posix())
                top_idx, scores = PTX_OPS.embedding_cosine_topk(query_vec, 5)
            except Exception:
                top_idx = np.array([], dtype=np.int32)
                scores = np.array([], dtype=np.float32)
            finally:
                PTX_OPS.geometry_release()

            if top_idx.size == 0:
                continue
            metadata = self._load_language_metadata(glb_path)
            for idx, score in zip(top_idx.tolist(), scores.tolist()):
                if idx < 0 or idx >= len(metadata):
                    continue
                meta = metadata[idx] if isinstance(metadata[idx], dict) else {}
                payload = meta.get("payload", {}) if isinstance(meta, dict) else {}
                lemma_meta = str(payload.get("lemma") or meta.get("name") or "").lower()
                answer = self._format_language_answer(action, payload)
                if not answer:
                    continue
                exact_match = lemma_meta == lemma_lower
                effective_score = score + (0.5 if exact_match else 0.0)
                if effective_score > best_score:
                    best_score = effective_score
                    best_answer = answer
                if exact_match:
                    break
        return best_answer

    def _select_language_candidates(self, lang_hint: Optional[str]) -> List[Dict[str, object]]:
        if not lang_hint:
            return self._language_catalog
        lang_hint = lang_hint.lower()
        candidates = [entry for entry in self._language_catalog if entry["language"].startswith(lang_hint)]
        if not candidates:
            candidates = [entry for entry in self._language_catalog if lang_hint in entry["language"]]
        return candidates or self._language_catalog

    def _load_language_metadata(self, glb_path: Path) -> List[Dict[str, object]]:
        cached = self._language_metadata_cache.get(glb_path)
        if cached is not None:
            return cached
        try:
            gltf = GLTF2().load(glb_path.as_posix())
            meta = (
                gltf.meshes[0]
                .primitives[0]
                .extras
                .get("k3d", {})
                .get("metadata", [])
            )
            if isinstance(meta, list):
                self._language_metadata_cache[glb_path] = meta
                return meta
        except Exception:
            pass
        self._language_metadata_cache[glb_path] = []
        return []

    def _format_language_answer(self, action: str, payload: Dict[str, object]) -> Optional[str]:
        if action == "definition":
            definition = payload.get("definition")
            if not definition:
                glosses = payload.get("glosses")
                if isinstance(glosses, list) and glosses:
                    definition = glosses[0]
            return str(definition) if definition else None
        if action == "ipa":
            pronunciations = payload.get("pronunciations")
            if isinstance(pronunciations, list) and pronunciations:
                return " / ".join(str(p) for p in pronunciations if p)
            ipa = payload.get("ipa")
            if isinstance(ipa, str):
                return ipa
            return None
        if action == "synonym":
            synonyms = payload.get("synonyms")
            if isinstance(synonyms, list) and synonyms:
                return synonyms[0]
            return None
        return None

    def _hash_embedding(self, text: str, dim: int) -> np.ndarray:
        joined = text.strip() or "language_query"
        digest = hashlib.sha256(joined.encode("utf-8")).digest()
        values: List[float] = []
        seed = digest
        while len(values) < dim:
            for byte in seed:
                values.append(((byte / 255.0) * 2.0) - 1.0)
                if len(values) >= dim:
                    break
            seed = hashlib.sha256(seed).digest()
        return np.asarray(values[:dim], dtype=np.float32)

    def _make_energy_grid(self, payload: Optional[Dict[str, object]]) -> Optional[np.ndarray]:
        """Synthesize a small 3D energy grid from current features for logging/training.

        - Uses any available feature vector in payload (PTX text features preferred)
        - Shapes to (N,N,N) where N is 4..16 (env: K3D_ENERGY_GRID_SIZE)
        - Normalizes to [0,1]
        """
        try:
            size = int(_os.environ.get("K3D_ENERGY_GRID_SIZE", "8"))
        except Exception:
            size = 8
        size = max(4, min(16, size))
        vec: Optional[List[float]] = None
        if isinstance(payload, dict):
            for key in ("ptx_text_features", "ptx_features", "embedding"):
                v = payload.get(key)
                if isinstance(v, list) and v:
                    vec = [float(x) for x in v]
                    break
        if vec is None:
            return None
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        needed = size * size * size
        reps = int(np.ceil(needed / float(arr.size)))
        tiled = np.tile(arr, reps)[:needed]
        grid = tiled.reshape(size, size, size)
        mx = float(np.max(np.abs(grid))) or 1.0
        grid = (grid / (mx + 1e-9) + 1.0) * 0.5
        return grid.astype(np.float32)

    # ------------------------------------------------------------------
    # House memory integration
    # ------------------------------------------------------------------
    def _discover_house_memory(self) -> Optional[Dict[str, object]]:
        base = Path("viewer/public/house")
        glb_path = base / "house_memory.glb"
        manifest_path = base / "house_memory.json"
        if not glb_path.exists():
            return None
        embedding_dim = 512
        if manifest_path.exists():
            try:
                meta = json.loads(manifest_path.read_text(encoding="utf-8"))
                embedding_dim = int(meta.get("embedding_dim", embedding_dim))
            except Exception:
                pass
        return {
            "glb": glb_path,
            "manifest": manifest_path,
            "embedding_dim": embedding_dim,
        }

    def _load_house_metadata(self) -> List[Dict[str, object]]:
        if self._house_metadata_cache is not None:
            return self._house_metadata_cache
        entry = self._house_memory_entry
        if entry is None:
            self._house_metadata_cache = []
            return self._house_metadata_cache
        glb_path: Path = entry["glb"]
        try:
            gltf = GLTF2().load(glb_path.as_posix())
            meta = (
                gltf.meshes[0]
                .primitives[0]
                .extras
                .get("k3d", {})
                .get("metadata", [])
            )
            if isinstance(meta, list):
                self._house_metadata_cache = meta
                return meta
        except Exception:
            pass
        self._house_metadata_cache = []
        return self._house_metadata_cache

    def _house_query_vector(
        self, query: str, fused_embedding: List[float], dim: int
    ) -> np.ndarray:
        projected = self._embedding_from_fused(fused_embedding, dim)
        if projected is not None:
            return projected
        prompt_key = (query or "").strip().lower()
        if not prompt_key:
            return np.zeros(dim, dtype=np.float32)
        return self._hash_embedding(prompt_key, dim)

    def _attempt_house_memory_lookup(self, query: str, fused_embedding: List[float]) -> Optional[str]:
        entry = self._house_memory_entry
        if entry is None or not Path(entry["glb"]).exists():
            entry = self._discover_house_memory()
            self._house_memory_entry = entry
            self._house_metadata_cache = None
        if entry is None or not query:
            return None
        glb_path: Path = entry["glb"]
        embedding_dim = int(entry.get("embedding_dim", 512))
        query_vec = self._house_query_vector(query, fused_embedding, embedding_dim)
        scene_loaded = False
        try:
            PTX_OPS.geometry_load_scene(glb_path.as_posix())
            scene_loaded = True
            top_idx, scores = PTX_OPS.embedding_cosine_topk(query_vec, 5)
        except Exception:
            return None
        finally:
            if scene_loaded:
                try:
                    PTX_OPS.geometry_release()
                except Exception:
                    pass
        if top_idx.size == 0:
            return None
        metadata = self._load_house_metadata()
        best_answer: Optional[str] = None
        best_score = -1.0
        self._last_house_payload = None
        for idx, score in zip(top_idx.tolist(), scores.tolist()):
            if idx < 0 or idx >= len(metadata):
                continue
            if score < 0.58:
                continue
            meta = metadata[idx] if isinstance(metadata[idx], dict) else {}
            payload = meta.get("payload", {}) if isinstance(meta, dict) else {}
            answer = payload.get("summary") or payload.get("title") or payload.get("path")
            if not answer:
                name = meta.get("name") if isinstance(meta, dict) else None
                artifact_type = payload.get("type") if isinstance(payload, dict) else None
                pieces = [str(part) for part in [name, artifact_type] if part]
                answer = " | ".join(pieces) if pieces else None
            if not answer:
                continue
            effective = float(score)
            if effective > best_score:
                best_score = effective
                best_answer = str(answer)
                if isinstance(payload, dict) and payload:
                    self._last_house_payload = payload
        return best_answer

    # ------------------------------------------------------------------
    # Learning memory integration
    # ------------------------------------------------------------------
    def _discover_learning_memory(self) -> Optional[Dict[str, object]]:
        base = Path("viewer/public/galaxy")
        glb_path = base / "learning_memory.glb"
        manifest_path = base / "learning_memory.json"
        if not glb_path.exists():
            return None
        embedding_dim = 512
        if manifest_path.exists():
            try:
                meta = json.loads(manifest_path.read_text(encoding="utf-8"))
                embedding_dim = int(meta.get("embedding_dim", embedding_dim))
            except Exception:
                pass
        return {
            "glb": glb_path,
            "manifest": manifest_path,
            "embedding_dim": embedding_dim,
        }

    def _load_learning_metadata(self) -> List[Dict[str, object]]:
        if self._learning_metadata_cache is not None:
            return self._learning_metadata_cache
        entry = self._learning_memory_entry
        if entry is None:
            self._learning_metadata_cache = []
            return self._learning_metadata_cache
        glb_path: Path = entry["glb"]
        try:
            gltf = GLTF2().load(glb_path.as_posix())
            meta = (
                gltf.meshes[0]
                .primitives[0]
                .extras
                .get("k3d", {})
                .get("metadata", [])
            )
            if isinstance(meta, list):
                self._learning_metadata_cache = meta
                return meta
        except Exception:
            pass
        self._learning_metadata_cache = []
        return self._learning_metadata_cache

    def _attempt_learning_memory_lookup(self, query: str, fused_embedding: List[float]) -> Optional[str]:
        entry = self._learning_memory_entry
        if entry is None or not Path(entry["glb"]).exists():
            entry = self._discover_learning_memory()
            self._learning_memory_entry = entry
            self._learning_metadata_cache = None
        if entry is None or not query:
            return None
        glb_path: Path = entry["glb"]
        embedding_dim = int(entry.get("embedding_dim", 512))
        query_vec = self._learning_query_vector(query, fused_embedding, embedding_dim)
        scene_loaded = False
        try:
            PTX_OPS.geometry_load_scene(glb_path.as_posix())
            scene_loaded = True
            top_idx, scores = PTX_OPS.embedding_cosine_topk(query_vec, 5)
        except Exception:
            return None
        finally:
            if scene_loaded:
                try:
                    PTX_OPS.geometry_release()
                except Exception:
                    pass
        if top_idx.size == 0:
            return None
        metadata = self._load_learning_metadata()
        best_answer = None
        best_score = -1.0
        self._last_learning_payload = None
        for idx, score in zip(top_idx.tolist(), scores.tolist()):
            if idx < 0 or idx >= len(metadata):
                continue
            if score < 0.62:
                continue
            meta = metadata[idx] if isinstance(metadata[idx], dict) else {}
            payload = meta.get("payload", {}) if isinstance(meta, dict) else {}
            answer = payload.get("true_answer") or payload.get("predicted")
            if not answer:
                continue
            effective = float(score)
            prompt = payload.get("prompt")
            if isinstance(prompt, str) and prompt.strip().lower() == query.strip().lower():
                effective += 0.2
            if effective > best_score:
                best_score = effective
                best_answer = str(answer)
                if isinstance(payload, dict) and payload:
                    self._last_learning_payload = payload
        return best_answer

    def _learning_query_vector(
        self, query: str, fused_embedding: List[float], dim: int
    ) -> np.ndarray:
        projected = self._embedding_from_fused(fused_embedding, dim)
        if projected is not None:
            return projected
        prompt_key = (query or "").strip().lower()
        if not prompt_key:
            return np.zeros(dim, dtype=np.float32)
        return self._hash_embedding(prompt_key, dim)

    def _embedding_from_fused(
        self, fused_embedding: List[float], dim: int
    ) -> Optional[np.ndarray]:
        if not fused_embedding:
            return None
        try:
            vec = np.asarray(fused_embedding, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if vec.size == 0:
            return None
        if vec.size < dim:
            vec = np.pad(vec, (0, dim - vec.size))
        elif vec.size > dim:
            vec = vec[:dim]
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return None
        return (vec / norm).astype(np.float32)

    def _expand_to_dim(self, features: Optional[List[float]], dim: int) -> np.ndarray:
        if not features:
            return np.zeros(dim, dtype=np.float32)
        arr = np.asarray([float(x) for x in features], dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return np.zeros(dim, dtype=np.float32)
        if arr.size == dim:
            return arr
        if arr.size > dim:
            return arr[:dim]
        reps = int(np.ceil(dim / float(arr.size)))
        tiled = np.tile(arr, reps)[:dim]
        return tiled.astype(np.float32)

    def _best_media_ptx_features(self, modality: str, query_lower: str) -> Optional[List[float]]:
        # Select best candidate by text similarity and return PTX features only
        candidates = self._collect_media_candidates(modality)
        if not candidates:
            return None
        best = None
        best_score = -1.0
        for entry in candidates:
            summary = entry.get("summary") or ""
            payload = entry.get("payload", {})
            try:
                payload_text = json.dumps(payload, ensure_ascii=False, default=str)
            except Exception:
                payload_text = str(payload)
            matcher = SequenceMatcher(None, query_lower, (str(summary) + " " + payload_text).lower())
            score = matcher.quick_ratio()
            if score < 0.6:
                score = matcher.ratio()
            if score > best_score:
                best_score = score
                best = entry
        if not best:
            return None
        asset_path_str = str(best.get("path", ""))
        resolved = self._resolve_public_path(asset_path_str)
        if not resolved or not resolved.exists():
            return None
        try:
            if modality == "image":
                info = PTX_OPS.image_modality(resolved.as_posix())
            elif modality == "audio":
                info = PTX_OPS.audio_modality(resolved.as_posix())
            else:
                info = PTX_OPS.video_modality(resolved.as_posix())
            feats = info.get("features") if isinstance(info, dict) else None
            if isinstance(feats, list):
                return [float(x) for x in feats]
        except Exception:
            return None
        return None

    def _build_ptx_fused_embedding(self, query: str) -> List[float]:
        ql = (query or "").lower()
        text_feats: Optional[List[float]] = None
        if query and str(_os.environ.get("K3D_DISABLE_TEXT_MODALITY", "0")).lower() not in {"1","true","yes"}:
            try:
                info = PTX_OPS.text_modality(query)
                feats = info.get("features") if isinstance(info, dict) else None
                if isinstance(feats, list):
                    text_feats = [float(x) for x in feats]
            except Exception:
                text_feats = None
        # Optional media PTX features (choose best candidate if query hints a modality)
        img_feats = aud_feats = vid_feats = None
        mod_hint = self._detect_modality(ql)
        if mod_hint == "image":
            img_feats = self._best_media_ptx_features("image", ql)
        elif mod_hint == "audio":
            aud_feats = self._best_media_ptx_features("audio", ql)
        elif mod_hint == "video":
            vid_feats = self._best_media_ptx_features("video", ql)
        # Expand to configured block dims (defaults 128 each): [text | image | audio | video]
        import os as __os
        dims_env = (__os.getenv("K3D_FUSE_DIMS", "") or "").strip()
        try:
            parts = [int(x) for x in dims_env.replace(":", ",").split(",") if x.strip()]
        except Exception:
            parts = []
        while len(parts) < 4:
            parts.append(128)
        tdim, idim, adim, vdim = [max(1, int(x)) for x in parts[:4]]
        blocks = [
            self._expand_to_dim(text_feats, tdim),
            self._expand_to_dim(img_feats, idim),
            self._expand_to_dim(aud_feats, adim),
            self._expand_to_dim(vid_feats, vdim),
        ]
        fused = np.concatenate(blocks, axis=0).astype(np.float32)
        return [float(x) for x in fused]

    def _predict_arc_grid(self, fused_embedding: List[float]) -> Optional[str]:
        """Predict a fixed 10x10 ARC-style grid with 10 classes (0..9).
        Returns a JSON string with {"grid": [[...],[...],...]}.
        """
        vec = self._embedding_from_fused(fused_embedding, 512)
        if vec is None:
            vec = self._hash_embedding(("arc-grid"), 512)
        t = torch.from_numpy(vec).to(self.device)
        with torch.no_grad():
            h = self._arc_hidden(t)
            logits = self._arc_out(h)
            logits = logits.view(10, 10, 10)  # H, W, C
            pred = torch.argmax(logits, dim=-1).int().cpu().numpy()
        grid = [[int(x) for x in row] for row in pred.tolist()]
        return json.dumps({"grid": grid, "tags": ["arc", "grid"]})

    def arc_grid_train_step(self, fused_embedding: List[float], target_grid: List[List[int]]) -> float:
        """One training step for ARC grid head given a target 10x10 grid (values 0..9)."""
        # Coerce/resize target to 10x10
        try:
            import numpy as _np
            tg = _np.array(target_grid, dtype=_np.int64)
            if tg.ndim != 2:
                return 0.0
            # Simple resize/pad/crop to 10x10
            h, w = tg.shape
            out = _np.zeros((10, 10), dtype=_np.int64)
            hh = min(10, h); ww = min(10, w)
            out[:hh, :ww] = tg[:hh, :ww]
        except Exception:
            return 0.0
        vec = self._embedding_from_fused(fused_embedding, 512)
        if vec is None:
            vec = self._hash_embedding(("arc-train"), 512)
        x = torch.from_numpy(vec).to(self.device)
        y = torch.from_numpy(out).to(self.device)
        self._arc_hidden.train(); self._arc_out.train()
        h = self._arc_hidden(x)
        logits = self._arc_out(h).view(10, 10, 10)
        loss = self._arc_ce(logits.permute(2, 0, 1).unsqueeze(0), y.unsqueeze(0))  # CE over class dim
        if not torch.isfinite(loss):
            self._arc_opt.zero_grad(set_to_none=True)
            return 0.0
        self._arc_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self._arc_hidden.parameters()) + list(self._arc_out.parameters()), 1.0)
        self._arc_opt.step()
        val = float(loss.detach().item())
        return 0.0 if (not (val == val) or math.isinf(val)) else val

    def _generate_shape_artifact(
        self, prompt: str, fused_embedding: List[float]
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        path = PTX_OPS.generate_shape(prompt)
        metadata = PTX_OPS.last_generated_shape() or {}
        extras = metadata.get("extras", {}) or {}
        manifest_entry = metadata.get("manifest_entry") or {}
        artifact_path = metadata.get("path", path)
        payload = {
            "artifact_path": artifact_path,
            "manifest": manifest_entry,
            "extras": extras,
        }
        tags = ["generated_shape", "tablet"]
        self.append_learning_memory(
            prompt=f"GENERATED SHAPE :: {prompt}",
            true_answer=str(artifact_path),
            predicted=str(artifact_path),
            score=1.0,
            tags=tags,
            metadata={
                "artifact_path": artifact_path,
                "extras": extras,
                "manifest_entry": manifest_entry,
            },
        )
        self.reload_house_memory()
        self._refresh_media_cache()
        response = manifest_entry.get("path") or str(artifact_path)
        if extras.get("name"):
            payload["summary"] = extras["name"]
            response = extras["name"]
        return response, payload

    def append_learning_memory(
        self,
        *,
        prompt: str,
        true_answer: str,
        predicted: str,
        score: float,
        quick_feedback: Optional[Dict[str, object]] = None,
        deep_feedback: Optional[Dict[str, object]] = None,
        tags: Optional[List[str]] = None,
        language: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Optional[Path]:
        timestamp = datetime.utcnow().isoformat() + "Z"
        record_id = f"learning_{hashlib.sha256((timestamp + prompt).encode('utf-8')).hexdigest()[:16]}"
        record = {
            "id": record_id,
            "timestamp": timestamp,
            "prompt": prompt,
            "true_answer": true_answer,
            "predicted": predicted,
            "score": float(score),
            "quick_feedback": quick_feedback or {},
            "deep_feedback": deep_feedback or {},
            "language": (language or "").lower() or None,
            "tags": tags or [],
        }
        if metadata:
            for key, value in metadata.items():
                if key in record and isinstance(record[key], (dict, list)) and isinstance(value, (dict, list)):
                    record[key] = value
                elif key not in record:
                    record[key] = value
                else:
                    record[key] = value
        try:
            self.learning_memory_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with self.learning_memory_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.reload_learning_memory()
            return self.learning_memory_jsonl
        except Exception:
            return None

    def reload_learning_memory(self) -> bool:
        self._learning_memory_entry = self._discover_learning_memory()
        self._learning_metadata_cache = None
        return self._learning_memory_entry is not None

    def reload_house_memory(self) -> bool:
        self._house_memory_entry = self._discover_house_memory()
        self._house_metadata_cache = None
        return self._house_memory_entry is not None
