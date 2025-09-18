from __future__ import annotations

import json
import subprocess
from typing import Dict

from ...tools.phase10.teacher_prompt import TEACHER_SYSTEM_PROMPT  # type: ignore


class TeacherEvaluator:
    def __init__(
        self,
        ollama_url: str = "http://192.168.0.4:11434",
        *,
        initial_timeout: int = 300,
        timeout: int = 150,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.initial_timeout = max(initial_timeout, timeout)
        self.timeout = max(timeout, 1)
        self._model_warmups: Dict[str, bool] = {}
        try:
            result = subprocess.run(
                [
                    "curl",
                    "-s",
                    "--connect-timeout",
                    "5",
                    f"{self.ollama_url}/api/tags",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(f"❌ Ollama unreachable at {self.ollama_url} — aborting.")
            print(f"✅ Ollama confirmed reachable at {self.ollama_url}")
        except Exception as e:
            raise RuntimeError(f"❌ Ollama unreachable at {self.ollama_url} — aborting. Error: {e}")

    def evaluate_response(
        self,
        ai_response: str,
        *,
        model: str = "exaone3.5:latest",
        question: str | None = None,
        expected_answer: str | None = None,
    ) -> Dict:
        """Evaluate AI response with RLWHF scoring via Ollama (non-stream)."""
        if not isinstance(ai_response, str) or not ai_response.strip():
            return {"score": -1.0, "explanation": "Empty response"}

        model_name = str(model)
        warm = self._model_warmups.get(model_name, False)
        current_timeout = self.timeout if warm else self.initial_timeout

        if question or expected_answer:
            prompt = (
                f"{TEACHER_SYSTEM_PROMPT}\n\n"
                f"Question: {question or 'Unknown question provided.'}\n"
                f"Expected Answer: {expected_answer or 'Not supplied.'}\n"
                f"Student Answer: \"{ai_response}\""
            )
        else:
            prompt = f"{TEACHER_SYSTEM_PROMPT}\n\nAI Response: \"{ai_response}\""
        try:
            result = subprocess.run(
                [
                    "curl",
                    "-s",
                    f"{self.ollama_url}/api/generate",
                    "-d",
                    json.dumps(
                        {
                            "model": model_name,
                            "prompt": prompt,
                            "stream": False,
                            "keep_alive": "0s",
                        }
                    ),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=current_timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "score": -1.0,
                "explanation": (
                    f"Ollama model '{model_name}' timed out after {current_timeout}s; "
                    "warm the model manually or increase initial timeout."
                ),
            }
        except Exception as e:  # pragma: no cover
            return {"score": -1.0, "explanation": f"Ollama invocation failed: {e}"}

        if result.returncode != 0:
            return {"score": -1.0, "explanation": f"Ollama failed: {result.stderr.strip()}"}

        try:
            payload = json.loads(result.stdout)
            evaluation = str(payload.get("response", "")).strip()
        except Exception:
            evaluation = result.stdout.strip()

        if result.returncode == 0:
            self._model_warmups[model_name] = True
        score = self.extract_score(evaluation)
        explanation = self.extract_explanation(evaluation)
        return {"score": score, "explanation": explanation}

    def extract_score(self, evaluation: str) -> float:
        """Extract score sentinel tokens from evaluation text."""
        if "❌ -1" in evaluation:
            return -1.0
        if "🚫 -0.5" in evaluation or "-0.5 point" in evaluation:
            return -0.5
        if "⚠️ +0.5" in evaluation or "+0.5" in evaluation:
            return 0.5
        if "🛑 0" in evaluation or "0 point" in evaluation:
            return 0.0
        if "✅ +1" in evaluation or "+1" in evaluation:
            return 1.0
        # Fallback: penalize unclear judgments
        return -1.0

    def extract_explanation(self, evaluation: str) -> str:
        """Strip leading score glyph if present and return the rest."""
        out = evaluation
        for prefix in ("❌ -1 point. ", "🚫 -0.5 point. ", "⚠️ +0.5 point. ", "🛑 0 point. ", "✅ +1 point. "):
            if out.startswith(prefix):
                return out[len(prefix) :]
        return out
