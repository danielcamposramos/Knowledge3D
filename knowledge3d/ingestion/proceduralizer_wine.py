"""Thin WINE-style capture bridge for proceduralizer model calls.

This bridge owns transport, capture, and receipt generation. Deterministic
knowledge normalization happens after this boundary.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from typing import Any

from knowledge3d.ingestion.ollama_manager import OllamaManager

from .proceduralizer_contract import (
    PROCEDURALIZER_BUNDLE_JSON_SCHEMA,
    PROCEDURALIZER_MODEL_PROFILES,
    ProceduralizerReceipt,
    ProceduralizerRequest,
    extract_json_object,
    parse_bundle,
    proceduralizer_system_prompt,
    request_hash,
    response_hash,
)


def _request_user_message(request: ProceduralizerRequest) -> str:
    parts = [
        f"source_kind={request.source_kind}",
        f"source_id={request.source_id}",
        f"source_path={request.source_path or 'unknown'}",
        f"domain_hint={request.domain_hint or 'General'}",
        f"ingest_mode={request.ingest_mode or 'augment'}",
        f"mode={request.mode or 'standard'}",
        "",
        "Response schema:",
        json.dumps(PROCEDURALIZER_BUNDLE_JSON_SCHEMA, ensure_ascii=False, sort_keys=True),
        "",
    ]
    if request.existing_ref_menu:
        parts.extend([request.existing_ref_menu.strip(), "", "---", ""])
    if request.context_chunks:
        parts.append("Context chunks:")
        for index, chunk in enumerate(request.context_chunks, start=1):
            parts.append(f"[context {index}] {str(chunk).strip()[:1600]}")
        parts.extend(["", "---", ""])
    if request.peer_content_sample:
        parts.append("Peer content samples:")
        for index, sample in enumerate(list(request.peer_content_sample)[:3], start=1):
            parts.append(f"[peer {index}] {str(sample).strip()[:1200]}")
        parts.extend(["", "---", ""])
    if request.web_evidence:
        parts.append("Web evidence:")
        for index, item in enumerate(list(request.web_evidence)[:8], start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            parts.append(f"[evidence {index}] title={title}")
            parts.append(f"[evidence {index}] url={url}")
            if snippet:
                parts.append(f"[evidence {index}] snippet={snippet[:800]}")
        parts.extend(["", "---", ""])
    parts.append("Source content:")
    parts.append(str(request.content or "").strip()[:12000])
    return "\n".join(parts).strip()


class ProceduralizerWineBridge:
    """Stable request/receipt boundary around Ollama-backed model execution."""

    def __init__(
        self,
        *,
        provider: str = "ollama",
        default_timeout: float = 120.0,
        capture_dir: str | Path | None = None,
        ollama: OllamaManager | None = None,
    ) -> None:
        self.provider = str(provider or "ollama").strip().lower() or "ollama"
        self.default_timeout = float(default_timeout)
        self.capture_dir = Path(capture_dir) if capture_dir is not None else None
        self.ollama = ollama or OllamaManager(default_timeout=self.default_timeout)

    def resolve_model(self, *, model_profile: str = "quality", model: str | None = None) -> str:
        if model:
            return str(model).strip()
        return PROCEDURALIZER_MODEL_PROFILES.get(str(model_profile or "quality").strip().lower(), PROCEDURALIZER_MODEL_PROFILES["quality"])

    def submit(
        self,
        request: ProceduralizerRequest,
        *,
        model_profile: str = "quality",
        model: str | None = None,
        timeout: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> ProceduralizerReceipt:
        resolved_model = self.resolve_model(model_profile=model_profile, model=model)
        payload = request.to_dict()
        req_hash = request_hash(payload)
        request_path, response_path = self._capture_paths(req_hash)
        if request_path is not None:
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        started = time.perf_counter()
        run_timeout = float(timeout if timeout is not None else self.default_timeout)
        if self.provider != "ollama":
            bundle, _, failure_code = parse_bundle("", request)
            return ProceduralizerReceipt(
                status="transport_error",
                provider=self.provider,
                model=resolved_model,
                latency_ms=0,
                request_hash=req_hash,
                response_hash="",
                raw_response_path=str(response_path or ""),
                schema_ok=False,
                failure_code=failure_code or "unsupported_provider",
                retry_after_utc=self._retry_after_utc(failure_code or "unsupported_provider"),
                parsed_bundle=bundle,
            )

        result = self.ollama.chat(
            model=resolved_model,
            messages=[
                {"role": "system", "content": proceduralizer_system_prompt(request)},
                {"role": "user", "content": _request_user_message(request)},
            ],
            timeout=run_timeout,
            temperature=float(dict(options or {}).pop("temperature", 0.1)),
            options=options,
            response_format={"type": "object"}
            if str(request.mode or "standard").strip().lower() == "differentiation"
            else PROCEDURALIZER_BUNDLE_JSON_SCHEMA,
        )
        latency_ms = int((time.perf_counter() - started) * 1000.0)
        raw_output = result.output if result.returncode == 0 else (result.stderr or result.output)
        if response_path is not None:
            response_path.write_text(str(raw_output or ""), encoding="utf-8")
        resp_hash = response_hash(raw_output)

        if result.returncode != 0:
            bundle, _, failure_code = parse_bundle("", request)
            detected_failure = self._detect_failure_code(raw_output) or failure_code or "transport_error"
            return ProceduralizerReceipt(
                status="transport_error",
                provider=self.provider,
                model=resolved_model,
                latency_ms=latency_ms,
                request_hash=req_hash,
                response_hash=resp_hash,
                raw_response_path=str(response_path or ""),
                schema_ok=False,
                failure_code=detected_failure,
                retry_after_utc=self._retry_after_utc(detected_failure),
                parsed_bundle=bundle,
            )

        bundle, schema_ok, failure_code = parse_bundle(raw_output, request)
        detected_failure = self._detect_failure_code(raw_output) or failure_code
        status = "completed" if schema_ok and not detected_failure else "invalid_json"
        return ProceduralizerReceipt(
            status=status,
            provider=self.provider,
            model=resolved_model,
            latency_ms=latency_ms,
            request_hash=req_hash,
            response_hash=resp_hash,
            raw_response_path=str(response_path or ""),
            schema_ok=schema_ok,
            failure_code=detected_failure or "",
            retry_after_utc=self._retry_after_utc(detected_failure or ""),
            parsed_bundle=bundle,
        )

    def write_receipt(self, receipt: ProceduralizerReceipt, *, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(receipt.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    def _capture_paths(self, req_hash: str) -> tuple[Path | None, Path | None]:
        if self.capture_dir is None:
            return None, None
        stem = f"proceduralizer_{req_hash}"
        return self.capture_dir / f"{stem}.request.json", self.capture_dir / f"{stem}.response.txt"

    def _detect_failure_code(self, raw_output: str) -> str:
        lowered = str(raw_output or "").lower()
        if not lowered:
            return ""
        if extract_json_object(raw_output) is not None:
            return ""
        if lowered.strip() in {"timed out", "timeout"}:
            return "timeout"
        if "operation timed out" in lowered or "request timed out" in lowered:
            return "timeout"
        if "rate limit" in lowered:
            return "plan_limit_consumed"
        if "quota" in lowered and any(token in lowered for token in ("exceeded", "reached", "limit", "usage")):
            return "plan_limit_consumed"
        if "plan" in lowered and any(token in lowered for token in ("exhausted", "consumed", "limit", "quota", "reset")):
            return "plan_limit_consumed"
        if "context length" in lowered or "maximum context" in lowered:
            return "context_exhausted"
        return ""

    def _retry_after_utc(self, failure_code: str) -> str:
        if failure_code != "plan_limit_consumed":
            return ""
        return (datetime.now(timezone.utc) + timedelta(hours=5, minutes=1)).isoformat()


__all__ = ["ProceduralizerWineBridge"]
