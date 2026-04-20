"""Thin WINE-style capture bridge for proceduralizer model calls.

This bridge owns transport, capture, and receipt generation. Deterministic
knowledge normalization happens after this boundary.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any

from knowledge3d.ingestion.ollama_manager import OllamaManager

from .proceduralizer_contract import (
    PROCEDURALIZER_BUNDLE_JSON_SCHEMA,
    PROCEDURALIZER_MEANING_RESOLUTION_SCHEMA,
    PROCEDURALIZER_MODEL_PROFILES,
    ProceduralizerReceipt,
    ProceduralizerRequest,
    extract_json_object,
    parse_bundle,
    proceduralizer_system_prompt,
    request_hash,
    response_hash,
)

_LOG = logging.getLogger(__name__)

# Retry envelope for the Ollama call inside submit().
#
# Daniel's specification (2026-04-19): the plan is contracted to this project
# and generous, so prefer to wait and retry rather than fail the worker.
#
# Step 1 (every failure, regardless of kind): up to 3 attempts at 30s spacing.
#        This covers transient network errors AND catches the first brush
#        with a plan_limit_consumed response.
# Step 2 (only if last failure is plan_limit_consumed after step 1): sleep 5h,
#        then 3 attempts at 60s spacing.
# Step 3 (only if still plan_limit_consumed after step 2): sleep 30min, then
#        3 final attempts at 60s spacing.
# After step 3 the final receipt is returned as-is; the driver decides what
# to do with it.
DEFAULT_RETRY_TRANSIENT_ATTEMPTS = 3
DEFAULT_RETRY_TRANSIENT_DELAY_S = 30.0
DEFAULT_RETRY_PLAN_WAVES: tuple[tuple[int, float, float], ...] = (
    (3, 5 * 3600.0, 60.0),   # after 5h, 3 attempts @ 60s
    (3, 30 * 60.0, 60.0),    # after 30min, 3 attempts @ 60s
)

# Failure codes that benefit from a retry (either transient or plan-limit).
# Schema / JSON failures (invalid_json, invalid_schema, language_*, etc.) are
# deterministic given the same prompt and model, so retrying them is waste.
_RETRYABLE_FAILURES: frozenset[str] = frozenset(
    {"transport_error", "timeout", "plan_limit_consumed"}
)


def _request_user_message(request: ProceduralizerRequest) -> str:
    mode = str(request.mode or "standard").strip().lower()
    if mode == "meaning_resolution":
        embedded_schema = PROCEDURALIZER_MEANING_RESOLUTION_SCHEMA
    else:
        embedded_schema = PROCEDURALIZER_BUNDLE_JSON_SCHEMA
    parts = [
        f"source_kind={request.source_kind}",
        f"source_id={request.source_id}",
        f"source_path={request.source_path or 'unknown'}",
        f"domain_hint={request.domain_hint or 'General'}",
        f"ingest_mode={request.ingest_mode or 'augment'}",
        f"mode={request.mode or 'standard'}",
        "",
        "Response schema:",
        json.dumps(embedded_schema, ensure_ascii=False, sort_keys=True),
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
            if isinstance(item.get("hits"), list):
                row_id = str(item.get("row_id") or "").strip()
                query = str(item.get("query") or "").strip()
                if row_id:
                    parts.append(f"[evidence {index}] row_id={row_id}")
                if query:
                    parts.append(f"[evidence {index}] query={query}")
                for hit_index, hit in enumerate(list(item.get("hits") or [])[:5], start=1):
                    if not isinstance(hit, dict):
                        continue
                    title = str(hit.get("title") or "").strip()
                    url = str(hit.get("url") or "").strip()
                    snippet = str(hit.get("snippet") or "").strip()
                    parts.append(f"[evidence {index}.{hit_index}] title={title}")
                    parts.append(f"[evidence {index}.{hit_index}] url={url}")
                    if snippet:
                        parts.append(f"[evidence {index}.{hit_index}] snippet={snippet[:800]}")
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
        retry_transient_attempts: int = DEFAULT_RETRY_TRANSIENT_ATTEMPTS,
        retry_transient_delay_s: float = DEFAULT_RETRY_TRANSIENT_DELAY_S,
        retry_plan_waves: tuple[tuple[int, float, float], ...] = DEFAULT_RETRY_PLAN_WAVES,
        retry_sleep: Any = None,
    ) -> ProceduralizerReceipt:
        resolved_model = self.resolve_model(model_profile=model_profile, model=model)
        payload = request.to_dict()
        req_hash = request_hash(payload)
        request_path, response_path = self._capture_paths(req_hash)
        if request_path is not None:
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

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

        sleep_fn = retry_sleep if retry_sleep is not None else time.sleep

        def _do_one() -> ProceduralizerReceipt:
            return self._one_chat_attempt(
                request=request,
                resolved_model=resolved_model,
                options=options,
                run_timeout=run_timeout,
                req_hash=req_hash,
                response_path=response_path,
            )

        receipt = _do_one()
        # Step 1: transient + first plan-limit burst — up to `retry_transient_attempts`
        # total attempts at `retry_transient_delay_s` spacing.
        attempts = 1
        while (
            receipt.failure_code in _RETRYABLE_FAILURES
            and attempts < int(retry_transient_attempts)
        ):
            _LOG.warning(
                "proceduralizer retry (transient): failure=%s attempt=%s/%s sleeping=%.1fs",
                receipt.failure_code, attempts, retry_transient_attempts, retry_transient_delay_s,
            )
            sleep_fn(float(retry_transient_delay_s))
            receipt = _do_one()
            attempts += 1

        # Step 2+: escalating waves — ONLY when the failure is plan_limit_consumed.
        if receipt.failure_code == "plan_limit_consumed":
            for wave_idx, (wave_attempts, wave_pre_sleep, wave_delay) in enumerate(retry_plan_waves, start=1):
                _LOG.warning(
                    "proceduralizer plan_limit: entering wave %s — pre-sleep=%.0fs attempts=%s delay=%.0fs",
                    wave_idx, wave_pre_sleep, wave_attempts, wave_delay,
                )
                sleep_fn(float(wave_pre_sleep))
                for _ in range(int(wave_attempts)):
                    receipt = _do_one()
                    if receipt.failure_code != "plan_limit_consumed":
                        break
                    sleep_fn(float(wave_delay))
                if receipt.failure_code != "plan_limit_consumed":
                    break

        return receipt

    def _one_chat_attempt(
        self,
        *,
        request: ProceduralizerRequest,
        resolved_model: str,
        options: dict[str, Any] | None,
        run_timeout: float,
        req_hash: str,
        response_path: Path | None,
    ) -> ProceduralizerReceipt:
        started = time.perf_counter()
        mode = str(request.mode or "standard").strip().lower()
        if mode == "differentiation":
            response_format: dict[str, Any] = {"type": "object"}
        elif mode == "meaning_resolution":
            response_format = PROCEDURALIZER_MEANING_RESOLUTION_SCHEMA
        else:
            response_format = PROCEDURALIZER_BUNDLE_JSON_SCHEMA
        attempt_options = dict(options or {})
        temperature = float(attempt_options.pop("temperature", 0.1))
        result = self.ollama.chat(
            model=resolved_model,
            messages=[
                {"role": "system", "content": proceduralizer_system_prompt(request)},
                {"role": "user", "content": _request_user_message(request)},
            ],
            timeout=run_timeout,
            temperature=temperature,
            options=attempt_options,
            response_format=response_format,
        )
        latency_ms = int((time.perf_counter() - started) * 1000.0)
        raw_output = result.output if result.returncode == 0 else (result.stderr or result.output)
        if response_path is not None:
            response_path.write_text(str(raw_output or ""), encoding="utf-8")
        resp_hash = response_hash(raw_output)

        if result.returncode != 0:
            bundle, _, _ = parse_bundle("", request)
            # Prefer the detector's verdict (captures 'timeout', 'plan limit'
            # phrases in stderr) over parse_bundle's invalid_json diagnosis,
            # which is noise on an empty body. Fall back to transport_error
            # so the retry envelope correctly classifies this as retryable.
            detected_failure = self._detect_failure_code(raw_output) or "transport_error"
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
