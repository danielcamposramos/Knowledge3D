"""Ollama model lifecycle manager for clean ingestion tasks."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Any
import urllib.error
import urllib.request


@dataclass
class OllamaQueryResult:
    """Normalized query result payload."""

    model: str
    output: str
    returncode: int
    stderr: str


class OllamaModelManager:
    """Context-managed model loader/unloader for clean per-task context."""

    def __init__(self, default_timeout: float = 120.0):
        self.default_timeout = default_timeout
        self.current_model: str | None = None

    def load_model(self, model_name: str) -> None:
        """Warm a model by issuing a trivial prompt."""
        if self.current_model == model_name:
            return
        self.query(
            model=model_name,
            prompt="READY",
            timeout=self.default_timeout,
        )
        self.current_model = model_name

    def unload_model(self, model_name: str | None = None) -> None:
        """Unload model from Ollama runtime if supported."""
        model = model_name or self.current_model
        if not model:
            return
        subprocess.run(
            ["ollama", "stop", model],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(10.0, self.default_timeout / 2),
        )
        if model == self.current_model:
            self.current_model = None

    def query(
        self,
        model: str,
        prompt: str,
        timeout: float | None = None,
    ) -> OllamaQueryResult:
        """Run a non-conversational model call."""
        run_timeout = timeout if timeout is not None else self.default_timeout
        safe_prompt = self._sanitize_prompt(prompt)
        try:
            proc = subprocess.run(
                ["ollama", "run", model, safe_prompt],
                check=False,
                capture_output=True,
                text=True,
                timeout=run_timeout,
            )
            return OllamaQueryResult(
                model=model,
                output=(proc.stdout or "").strip(),
                returncode=proc.returncode,
                stderr=(proc.stderr or "").strip(),
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return OllamaQueryResult(
                model=model,
                output="",
                returncode=1,
                stderr=str(exc),
            )

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        timeout: float | None = None,
        temperature: float = 0.3,
        options: dict[str, Any] | None = None,
        response_format: str | dict[str, Any] | None = None,
    ) -> OllamaQueryResult:
        """Run a structured chat call through the Ollama HTTP API."""
        run_timeout = timeout if timeout is not None else self.default_timeout
        safe_messages: list[dict[str, str]] = []
        for message in list(messages or []):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user").strip() or "user"
            content = self._sanitize_prompt(str(message.get("content") or ""))
            safe_messages.append({"role": role, "content": content})

        resolved_options = dict(options or {})
        # 'think' must be a top-level key in the Ollama API, not inside options
        think_value = resolved_options.pop("think", None)
        payload: dict[str, Any] = {
            "model": model,
            "messages": safe_messages,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                **resolved_options,
            },
        }
        if response_format is not None:
            payload["format"] = response_format
        if think_value is not None:
            payload["think"] = think_value
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=run_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            msg = data.get("message", {})
            content = str(msg.get("content", "")).strip()
            # If content is empty but thinking is present, use thinking as fallback
            if not content:
                thinking = str(msg.get("thinking", "")).strip()
                if thinking:
                    content = thinking
            return OllamaQueryResult(
                model=model,
                output=content,
                returncode=0,
                stderr="",
            )
        except urllib.error.URLError as exc:
            return OllamaQueryResult(
                model=model,
                output="",
                returncode=1,
                stderr=str(exc),
            )
        except Exception as exc:
            return OllamaQueryResult(
                model=model,
                output="",
                returncode=1,
                stderr=str(exc),
            )

    def _sanitize_prompt(self, prompt: str) -> str:
        """
        Remove null/control bytes that can break subprocess argv handling.

        Keep tabs/newlines for prompt readability while normalizing unsafe
        control characters to spaces.
        """
        text = str(prompt or "")
        if "\x00" in text:
            text = text.replace("\x00", "")
        # Keep printable + newline/tab; replace remaining control chars.
        text = "".join(ch if (ch >= " " or ch in "\n\t") else " " for ch in text)
        # Collapse long whitespace runs but preserve line breaks.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Guard against ill-formed unicode sequences.
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
        return text.strip()

    def __enter__(self) -> "OllamaModelManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unload_model()


OllamaManager = OllamaModelManager
