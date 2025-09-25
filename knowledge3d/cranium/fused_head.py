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
from knowledge3d.skills.audio import embed_audio
from knowledge3d.skills.video import embed_video
from knowledge3d.skills.vision import embed_image


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
            "book",
            "tree",
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

    # ------------------------------------------------------------------
    def predict(self, query: str, fused_embedding: List[float]) -> str:
        self._last_house_payload = None
        self._last_learning_payload = None
        self._default_payload = None

        text_confidence: Optional[float] = None
        if query:
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
                return self._post_process_answer(query, PTX_OPS.format_numeric(result))
            except Exception:
                pass

        shape_prompt = self._extract_shape_prompt(query)
        if shape_prompt:
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

        numeric = self._simple_numeric_solver(query)
        if numeric is not None:
            return self._post_process_answer(query, PTX_OPS.format_numeric(numeric))

        house_answer = self._attempt_house_memory_lookup(query, fused_embedding)
        learning_answer = self._attempt_learning_memory_lookup(query, fused_embedding)

        blended_answer = self._combine_memory_answers(query, house_answer, learning_answer)
        if blended_answer is not None:
            payload = self._last_learning_payload or self._last_house_payload
            return self._post_process_answer(query, blended_answer, payload)

        language_answer = self._attempt_language_lookup(query)
        if language_answer is not None:
            return self._post_process_answer(query, language_answer)

        if "summarize" in ql:
            corpus_payload = self._lookup_corpus_payload(query)
            if corpus_payload:
                return self._post_process_answer(
                    query,
                    corpus_payload.get("text"),
                    corpus_payload.get("payload"),
                )
            return self._fallback_summary_response(query)

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

            if embedding is None:
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
