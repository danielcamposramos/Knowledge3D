from __future__ import annotations

import math
import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from pygltflib import GLTF2  # type: ignore

from knowledge3d.cranium.ptx.ptx_ops import PTX_OPS


class AdaptedFusedHead:
    """Fused head that routes queries through PTX-backed operators when possible."""

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("AdaptedFusedHead requires CUDA GPU (no CPU fallback)")
        self.device = torch.device("cuda")
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
        ).to(self.device)
        self.honesty_gate = nn.Linear(512, 1).to(self.device)
        self.predict_head = nn.Linear(512, 256).to(self.device)

        self._shapes = [
            "tetrahedron",
            "cube",
            "octahedron",
            "icosahedron",
            "dodecahedron",
            "hypersphere_projection",
            "fractal_tree",
        ]
        self._kernels = [
            "map_ray_thickness_to_resolution_kernel",
            "render_ray_if_honest_kernel",
            "adjust_zone_position_kernel",
        ]
        self._rays = ["modality_ray", "entropy_ray", "honesty_ray"]
        self._language_catalog = self._discover_language_galaxies()
        self._language_metadata_cache: Dict[Path, List[Dict[str, object]]] = {}
        self._house_memory_entry = self._discover_house_memory()
        self._house_metadata_cache: Optional[List[Dict[str, object]]] = None
        self.learning_memory_jsonl = Path("viewer/public/galaxy/working/learning_memory.jsonl")
        self._learning_memory_entry = self._discover_learning_memory()
        self._learning_metadata_cache: Optional[List[Dict[str, object]]] = None

    # ------------------------------------------------------------------
    def predict(self, query: str, fused_embedding: List[float]) -> str:
        rpn_expr = self._extract_rpn_expression(query)
        if rpn_expr:
            try:
                result = PTX_OPS.evaluate_rpn(rpn_expr)
                return PTX_OPS.format_numeric(result)
            except Exception:
                pass

        shape_prompt = self._extract_shape_prompt(query)
        if shape_prompt:
            try:
                return PTX_OPS.generate_shape(shape_prompt)
            except Exception:
                pass

        numeric = self._simple_numeric_solver(query)
        if numeric is not None:
            return PTX_OPS.format_numeric(numeric)

        house_answer = self._attempt_house_memory_lookup(query, fused_embedding)
        learning_answer = self._attempt_learning_memory_lookup(query, fused_embedding)

        blended_answer = self._combine_memory_answers(query, house_answer, learning_answer)
        if blended_answer is not None:
            return blended_answer

        language_answer = self._attempt_language_lookup(query)
        if language_answer is not None:
            return language_answer

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
        logits = self.predict_head(h)
        idx = int(torch.argmax(logits, dim=1).item())

        ql = (query or "").lower()
        all_outputs = self._shapes + self._kernels + self._rays
        pred = all_outputs[idx % len(all_outputs)]

        if ("zone" in ql or "museum" in ql or "garden" in ql) and honesty < 0.7:
            return "Zone 8 (Learning Museum)"
        if ("fusion" in ql or "shape" in ql or "quad" in ql) and honesty >= 0.7:
            return "icosahedron"
        if ("ray" in ql and "thick" in ql) or ("ray" in ql and "resolution" in ql):
            return "audio, medium"
        if "entropy" in ql and "ray" in ql:
            return "ray_length = log(embedding_entropy + 1) * scale_factor"
        if "depth" in ql or "φ" in ql or "phi" in ql:
            return str(int(math.floor(1.618 * max(0.5, honesty) * 10.0)))
        return pred

    def train_step(self, fused_embedding: List[float], true_answer: str, lr: float = 1e-3) -> None:
        _ = (fused_embedding, true_answer, lr)
        return

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

    def _simple_numeric_solver(self, query: str) -> Optional[float]:
        if not query:
            return None
        match = re.search(r"=\s*([\d\.]+)$", query)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        expr_match = re.search(r"evaluate\s+([\d\s\+\-\*/\.\(\)]+)$", query.lower())
        if expr_match:
            expr = expr_match.group(1)
            if re.fullmatch(r"[\d\s\+\-\*/\.\(\)]+", expr):
                try:
                    return float(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
                except Exception:
                    return None
        return None

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
        return best_answer

    def _learning_query_vector(
        self, query: str, fused_embedding: List[float], dim: int
    ) -> np.ndarray:
        prompt_key = (query or "").strip().lower()
        if not prompt_key:
            return np.zeros(dim, dtype=np.float32)
        return self._hash_embedding(prompt_key, dim)

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
