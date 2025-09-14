from __future__ import annotations

from dataclasses import dataclass
from typing import List
from pathlib import Path
from datetime import datetime
import io
import math
import wave


@dataclass
class TTSEngine:
    """Minimal, dependency-free synthesizer for MVP.

    Generates a simple sine-wave voice with pitch, speed, and volume controls.
    """
    sample_rate: int = 22050

    def synthesize(self, text: str, pitch: float = 120.0, speed: float = 1.0, volume: float = 0.7) -> bytes:
        # Duration proportional to text length and inverse speed
        dur = max(0.2, min(8.0, len(text) / 12.0)) / max(0.25, min(2.0, speed))
        n = int(self.sample_rate * dur)
        freq = max(60.0, min(300.0, float(pitch)))
        vol = max(0.0, min(1.0, float(volume))) * 0.8
        data = bytearray()
        for i in range(n):
            t = i / self.sample_rate
            # Simple formant-like modulation
            f = freq * (1.0 + 0.02 * math.sin(2 * math.pi * 3.0 * t))
            s = math.sin(2 * math.pi * f * t)
            # Envelope (attack/decay)
            env = min(1.0, i / (0.05 * self.sample_rate)) * (1.0 - min(1.0, i / (0.9 * n)))
            val = int(32767 * vol * env * s)
            data += int(val).to_bytes(2, byteorder='little', signed=True)
        # Write WAV to memory
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(bytes(data))
        return buf.getvalue()


class VoiceChat:
    def __init__(self) -> None:
        self.tts_engine = TTSEngine()

    def speak(self, text: str, embedding: List[float]) -> bytes:
        # Map embedding head to prosody
        e0 = float(embedding[0]) if len(embedding) > 0 else 0.0
        e1 = float(embedding[1]) if len(embedding) > 1 else 0.0
        e2 = float(embedding[2]) if len(embedding) > 2 else 0.0
        pitch = 120.0 + e0 * 50.0
        speed = 1.0 + e1 * 0.5
        volume = 0.5 + e2 * 0.5
        return self.tts_engine.synthesize(text, pitch=pitch, speed=speed, volume=volume)

    def speak_to_file(self, text: str, embedding: List[float], out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        data = self.speak(text, embedding)
        ts = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
        path = out_dir / f'utter-{ts}.wav'
        path.write_bytes(data)
        return path

