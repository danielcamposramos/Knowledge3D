"""Live voice interaction loop for K3D ingestion-path audio I/O."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile
import time

from .augmentation_providers import AugmentationProvider, AugmentationResult, create_provider
from .audio_pipeline import preprocess_audio
from .modal_providers import (
    SpeechResult,
    SpeechToTextProvider,
    TextToSpeechProvider,
    TranscriptionResult,
    create_stt_provider,
    create_tts_provider,
)


class VoiceSession:
    """Live voice interaction loop: audio -> text -> augmentation -> speech."""

    def __init__(
        self,
        *,
        stt_provider: SpeechToTextProvider | None = None,
        tts_provider: TextToSpeechProvider | None = None,
        augmentation_provider: AugmentationProvider | None = None,
        language: str = "en",
        output_dir: str | None = None,
    ) -> None:
        self.stt = stt_provider or create_stt_provider()
        self.tts = tts_provider or create_tts_provider()
        self.augmentation = augmentation_provider or create_provider()
        self.language = str(language).strip() or "en"
        self.output_dir = Path(output_dir or tempfile.mkdtemp(prefix="k3d_voice_"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.health_log = self.output_dir / "health_log.jsonl"
        self._running = False

    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        processed = preprocess_audio(audio_path, output_dir=str(self.output_dir))
        return self.stt.transcribe(processed, language=self.language)

    def query_and_respond(self, question: str) -> str:
        result = self.augmentation.augment(
            question,
            {
                "name": "voice_query",
                "domain_hint": "General",
                "source": "voice",
            },
        )
        answer = str(result.summary).strip() or "I do not know yet."
        self._log_interaction(question, answer, result)
        return answer

    def speak(self, text: str) -> str:
        timestamp_ms = int(time.time() * 1000)
        output_path = self.output_dir / f"response_{timestamp_ms}.mp3"
        result = self.tts.synthesize(str(text), output_path=str(output_path))
        return result.audio_path

    def process_audio_query(self, audio_path: str) -> tuple[str, str]:
        transcript = self.transcribe_file(audio_path)
        answer = self.query_and_respond(transcript.text)
        response_audio = self.speak(answer)
        return answer, response_audio

    def start_voice_loop(self, *, listen_seconds: float = 10.0) -> None:
        self._running = True
        while self._running:
            chunk_path = self._record_chunk(listen_seconds)
            if not chunk_path:
                continue
            try:
                _, audio_path = self.process_audio_query(chunk_path)
                self._play_audio(audio_path)
            except Exception:
                pass
            finally:
                Path(chunk_path).unlink(missing_ok=True)

    def stop(self) -> None:
        self._running = False

    def _record_chunk(self, seconds: float) -> str | None:
        path = self.output_dir / f"chunk_{int(time.time())}.wav"
        try:
            subprocess.run(
                [
                    "arecord",
                    "-f",
                    "S16_LE",
                    "-r",
                    "16000",
                    "-c",
                    "1",
                    "-d",
                    str(max(1, int(seconds))),
                    str(path),
                ],
                check=True,
                capture_output=True,
                timeout=float(seconds) + 5.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return None
        if not path.is_file() or path.stat().st_size <= 100:
            return None
        return str(path)

    def _play_audio(self, path: str) -> None:
        for player in ("paplay", "aplay", "ffplay"):
            try:
                subprocess.run(
                    [player, str(path)],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                return
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                continue

    def _log_interaction(self, question: str, answer: str, result: AugmentationResult) -> None:
        question_id = f"voice_{int(time.time() * 1000)}"
        entry = {
            "timestamp": time.time(),
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "question_id": question_id,
            "suite": "voice",
            "source": "voice",
            "question": str(question),
            "answer": str(answer),
            "expected": None,
            "correct": None,
            "elapsed_s": 0.0,
            "domain": result.domain,
            "confidence": float(result.confidence),
            "provider": result.provider,
        }
        with self.health_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


__all__ = ["VoiceSession"]
