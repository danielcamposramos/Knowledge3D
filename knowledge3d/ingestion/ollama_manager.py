"""Ollama model lifecycle manager for clean ingestion tasks."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


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
        try:
            proc = subprocess.run(
                ["ollama", "run", model, prompt],
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

    def __enter__(self) -> "OllamaModelManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unload_model()
