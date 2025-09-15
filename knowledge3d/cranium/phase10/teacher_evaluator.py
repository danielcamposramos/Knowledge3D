from __future__ import annotations

import json
import subprocess
from typing import Dict

from ...tools.phase10.teacher_prompt import TEACHER_SYSTEM_PROMPT  # type: ignore


class TeacherEvaluator:
    def __init__(self, ollama_url: str = "http://192.168.0.4:11434"):
        self.ollama_url = ollama_url.rstrip("/")

    def evaluate_response(self, ai_response: str, model: str = "exaone3.5:latest") -> Dict:
        """Evaluate AI response with RLWHF scoring via Ollama (non-stream)."""
        if not isinstance(ai_response, str) or not ai_response.strip():
            return {"score": -1.0, "explanation": "Empty response"}

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
                            "model": str(model),
                            "prompt": prompt,
                            "stream": False,
                            "keep_alive": "0s",
                        }
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception as e:  # pragma: no cover
            return {"score": -1.0, "explanation": f"Ollama invocation failed: {e}"}

        if result.returncode != 0:
            return {"score": -1.0, "explanation": f"Ollama failed: {result.stderr.strip()}"}

        try:
            payload = json.loads(result.stdout)
            evaluation = str(payload.get("response", "")).strip()
        except Exception:
            evaluation = result.stdout.strip()

        score = self.extract_score(evaluation)
        explanation = self.extract_explanation(evaluation)
        return {"score": score, "explanation": explanation}

    def extract_score(self, evaluation: str) -> float:
        """Extract score sentinel tokens from evaluation text."""
        if "❌ -1" in evaluation:
            return -1.0
        if "⚠️ +0.5" in evaluation or "+0.5" in evaluation:
            return 0.5
        if "✅ +1" in evaluation or "+1" in evaluation:
            return 1.0
        # Fallback: penalize unclear judgments
        return -1.0

    def extract_explanation(self, evaluation: str) -> str:
        """Strip leading score glyph if present and return the rest."""
        out = evaluation
        for prefix in ("❌ -1 point. ", "⚠️ +0.5 point. ", "✅ +1 point. "):
            if out.startswith(prefix):
                return out[len(prefix) :]
        return out

