from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import wave

from knowledge3d.tools.augmentation_providers import AugmentationResult
from knowledge3d.tools.modal_providers import SpeechResult, TranscriptionResult
from knowledge3d.tools.voice_session import VoiceSession


class _MockAugProvider:
    provider_name = "mock_aug"

    def augment(self, content: str, context: dict[str, object]) -> AugmentationResult:
        return AugmentationResult(
            summary=f"Answer: {content}",
            entities=[],
            relationships=[],
            domain="General",
            meaning_rpn_hint="GENERAL CONTENT ENTRY",
            taxonomy_refs=["concept_language"],
            surface_forms={"en": "Mock Answer", "pt": "Resposta Mock"},
            confidence=0.9,
            provider=self.provider_name,
            raw_response="{}",
        )

    def classify(self, content: str) -> str:
        return "General"

    def is_available(self) -> bool:
        return True


class _MockSTTProvider:
    provider_name = "mock_stt"

    def transcribe(self, audio_path: str, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(
            text="What is water?",
            language=language or "en",
            segments=[],
            confidence=1.0,
            duration_seconds=1.0,
            provider=self.provider_name,
            raw_response="{}",
        )

    def is_available(self) -> bool:
        return True


class _MockTTSProvider:
    provider_name = "mock_tts"

    def synthesize(self, text: str, *, output_path: str, voice: str | None = None) -> SpeechResult:
        Path(output_path).write_bytes(b"ID3")
        return SpeechResult(audio_path=str(output_path), format="mp3", duration_seconds=1.0, provider=self.provider_name)

    def is_available(self) -> bool:
        return True


def _create_test_wav(path: Path, *, rate: int = 16000, channels: int = 1, duration_s: float = 0.25) -> Path:
    frames = int(rate * duration_s)
    amplitude = 12000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        for index in range(frames):
            sample = int(amplitude * math.sin((2.0 * math.pi * 440.0 * index) / rate))
            frame = struct.pack("<h", sample)
            handle.writeframesraw(frame * channels)
    return path


def test_voice_session_creates(tmp_path: Path) -> None:
    session = VoiceSession(output_dir=str(tmp_path))
    assert session.output_dir == tmp_path


def test_voice_session_logs_interaction(tmp_path: Path) -> None:
    session = VoiceSession(
        stt_provider=_MockSTTProvider(),
        tts_provider=_MockTTSProvider(),
        augmentation_provider=_MockAugProvider(),
        output_dir=str(tmp_path),
    )

    answer = session.query_and_respond("What is water?")

    assert answer == "Answer: What is water?"
    assert session.health_log.is_file()
    entry = json.loads(session.health_log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["source"] == "voice"
    assert entry["suite"] == "voice"
    assert entry["question"] == "What is water?"


def test_voice_session_process_audio_query_end_to_end(tmp_path: Path) -> None:
    audio_path = _create_test_wav(tmp_path / "question.wav")
    session = VoiceSession(
        stt_provider=_MockSTTProvider(),
        tts_provider=_MockTTSProvider(),
        augmentation_provider=_MockAugProvider(),
        output_dir=str(tmp_path),
    )

    answer, response_audio = session.process_audio_query(str(audio_path))

    assert answer.startswith("Answer:")
    assert Path(response_audio).is_file()
