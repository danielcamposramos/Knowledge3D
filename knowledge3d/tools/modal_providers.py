"""Multi-modal provider layer for ingestion-path I/O conversions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib.util
import json
import mimetypes
import os
from pathlib import Path
import time
from typing import Any, Callable
import urllib.error
import urllib.request
import uuid


class ModalProvider(ABC):
    """Marker base for all modal conversion providers."""

    provider_name = "modal"
    modality = "unknown"

    @abstractmethod
    def is_available(self) -> bool:
        """Whether provider credentials/runtime are available."""


@dataclass
class TranscriptionResult:
    """Result of a speech-to-text conversion."""

    text: str
    language: str
    segments: list[dict[str, Any]]
    confidence: float
    duration_seconds: float
    provider: str
    raw_response: str


@dataclass
class SpeechResult:
    """Result of a text-to-speech conversion."""

    audio_path: str
    format: str
    duration_seconds: float
    provider: str


@dataclass
class Mesh3DResult:
    """Result of a 3D generation task."""

    mesh_path: str
    format: str
    task_id: str
    provider: str
    raw_response: str


class SpeechToTextProvider(ModalProvider):
    """Base for audio/video → text transcription."""

    modality = "stt"

    @abstractmethod
    def transcribe(self, audio_path: str, *, language: str | None = None) -> TranscriptionResult:
        """Transcribe audio/video into text."""


class TextToSpeechProvider(ModalProvider):
    """Base for text → speech synthesis."""

    modality = "tts"

    @abstractmethod
    def synthesize(self, text: str, *, output_path: str, voice: str | None = None) -> SpeechResult:
        """Synthesize speech to a target audio path."""


class TextTo3DProvider(ModalProvider):
    """Base for text description → 3D generation."""

    modality = "text_to_3d"

    @abstractmethod
    def generate(self, prompt: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        """Generate 3D mesh from a text prompt."""


class ImageTo3DProvider(ModalProvider):
    """Base for image → 3D generation."""

    modality = "image_to_3d"

    @abstractmethod
    def generate(self, image_path: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        """Generate 3D mesh from a reference image."""


def _segment_value(segment: Any, key: str, default: Any = None) -> Any:
    if isinstance(segment, dict):
        return segment.get(key, default)
    return getattr(segment, key, default)


def _rest_request(
    method: str,
    url: str,
    *,
    api_key: str,
    json_data: dict[str, Any] | None = None,
    files: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Make authenticated REST request and return parsed JSON."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data: bytes | None = None
    if files:
        data, content_type = _build_multipart_body(fields=json_data or {}, files=files)
        headers["Content-Type"] = content_type
    elif json_data is not None:
        data = json.dumps(json_data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"REST request failed [{exc.code}] {url}: {body}") from exc
    text = payload.decode("utf-8", errors="ignore").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def _rest_request_binary(
    method: str,
    url: str,
    *,
    api_key: str,
    json_data: dict[str, Any] | None = None,
    files: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> bytes:
    """Make authenticated REST request and return raw bytes."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data: bytes | None = None
    if files:
        data, content_type = _build_multipart_body(fields=json_data or {}, files=files)
        headers["Content-Type"] = content_type
    elif json_data is not None:
        data = json.dumps(json_data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"REST binary request failed [{exc.code}] {url}: {body}") from exc


def _download_file(url: str, output_path: str | Path, *, timeout: float = 60.0) -> Path:
    """Download a file to disk."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        target.write_bytes(response.read())
    return target


def _build_multipart_body(
    *,
    fields: dict[str, Any],
    files: dict[str, str],
) -> tuple[bytes, str]:
    boundary = f"----K3DFormBoundary{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8")
        )
    for field_name, file_path in files.items():
        path = Path(file_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            (
                f'Content-Disposition: form-data; name="{field_name}"; filename="{path.name}"\r\n'
                f"Content-Type: {mime_type}\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(path.read_bytes())
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _poll_until_complete(
    check_fn: Callable[[], dict[str, Any]],
    *,
    timeout: float = 300.0,
    interval: float = 5.0,
    terminal_statuses: tuple[str, ...] = ("SUCCEEDED", "completed", "done"),
    failure_statuses: tuple[str, ...] = ("FAILED", "failed", "error", "expired"),
) -> dict[str, Any]:
    """Poll until a 3D generation job completes."""
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        result = check_fn()
        status = str(result.get("status", "")).lower()
        if any(token.lower() in status for token in terminal_statuses):
            return result
        if any(token.lower() in status for token in failure_statuses):
            raise RuntimeError(f"3D generation failed: {result}")
        time.sleep(interval)
    raise TimeoutError(f"3D generation timed out after {timeout}s")


class WhisperLocalProvider(SpeechToTextProvider):
    """Local faster-whisper transcription."""

    provider_name = "whisper_local"

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        **_: Any,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(self, audio_path: str, *, language: str | None = None) -> TranscriptionResult:
        model = self._get_model()
        segments_iter, info = model.transcribe(
            audio_path,
            beam_size=5,
            language=language,
            word_timestamps=True,
        )
        segments: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for segment in segments_iter:
            seg_text = str(getattr(segment, "text", "")).strip()
            segments.append(
                {
                    "start": float(getattr(segment, "start", 0.0)),
                    "end": float(getattr(segment, "end", 0.0)),
                    "text": seg_text,
                }
            )
            if seg_text:
                text_parts.append(seg_text)
        language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        return TranscriptionResult(
            text=" ".join(text_parts).strip(),
            language=str(getattr(info, "language", None) or language or "unknown"),
            segments=segments,
            confidence=max(0.0, min(1.0, language_probability)),
            duration_seconds=max(0.0, duration),
            provider=self.provider_name,
            raw_response="",
        )

    def is_available(self) -> bool:
        return importlib.util.find_spec("faster_whisper") is not None


class VibeServerProvider(SpeechToTextProvider):
    """OpenAI-compatible local Vibe server."""

    provider_name = "vibe_server"

    def __init__(self, base_url: str = "http://localhost:3022", **_: Any) -> None:
        self.base_url = base_url.rstrip("/")

    def transcribe(self, audio_path: str, *, language: str | None = None) -> TranscriptionResult:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("openai package is not installed.") from exc
        client = OpenAI(api_key="local", base_url=self.base_url)
        with Path(audio_path).open("rb") as handle:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=handle,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        raw_segments = getattr(response, "segments", None) or []
        segments = [
            {
                "start": float(_segment_value(segment, "start", 0.0) or 0.0),
                "end": float(_segment_value(segment, "end", 0.0) or 0.0),
                "text": str(_segment_value(segment, "text", "")).strip(),
            }
            for segment in raw_segments
        ]
        return TranscriptionResult(
            text=str(getattr(response, "text", "") or response).strip(),
            language=str(getattr(response, "language", None) or language or "unknown"),
            segments=segments,
            confidence=0.8,
            duration_seconds=float(getattr(response, "duration", 0.0) or 0.0),
            provider=self.provider_name,
            raw_response=str(response),
        )

    def is_available(self) -> bool:
        try:
            request = urllib.request.Request(f"{self.base_url}/docs", method="HEAD")
            with urllib.request.urlopen(request, timeout=2):
                return True
        except Exception:
            return False


class OpenAIWhisperProvider(SpeechToTextProvider):
    """OpenAI Whisper cloud STT."""

    provider_name = "openai_whisper"

    def __init__(self, model: str = "whisper-1", **_: Any) -> None:
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip()

    def transcribe(self, audio_path: str, *, language: str | None = None) -> TranscriptionResult:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("openai package is not installed.") from exc
        client = OpenAI(api_key=self.api_key)
        with Path(audio_path).open("rb") as handle:
            response = client.audio.transcriptions.create(
                model=self.model,
                file=handle,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        raw_segments = getattr(response, "segments", None) or []
        segments = [
            {
                "start": float(_segment_value(segment, "start", 0.0) or 0.0),
                "end": float(_segment_value(segment, "end", 0.0) or 0.0),
                "text": str(_segment_value(segment, "text", "")).strip(),
            }
            for segment in raw_segments
        ]
        return TranscriptionResult(
            text=str(getattr(response, "text", "") or response).strip(),
            language=str(getattr(response, "language", None) or language or "unknown"),
            segments=segments,
            confidence=0.9,
            duration_seconds=float(getattr(response, "duration", 0.0) or 0.0),
            provider=self.provider_name,
            raw_response=str(response),
        )

    def is_available(self) -> bool:
        return bool(self.api_key) and importlib.util.find_spec("openai") is not None


class DeepgramSTTProvider(SpeechToTextProvider):
    """Deepgram STT provider."""

    provider_name = "deepgram"

    def __init__(self, model: str = "nova-3", **_: Any) -> None:
        self.model = model
        self.api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()

    def transcribe(self, audio_path: str, *, language: str | None = None) -> TranscriptionResult:
        try:
            from deepgram import DeepgramClient, FileSource, PrerecordedOptions
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("deepgram package is not installed.") from exc
        client = DeepgramClient(self.api_key)
        with Path(audio_path).open("rb") as handle:
            payload: FileSource = {"buffer": handle.read()}
        options = PrerecordedOptions(
            model=self.model,
            smart_format=True,
            language=language,
            utterances=True,
        )
        response = client.listen.rest.v("1").transcribe_file(payload, options)
        result = response.results
        channel = result.channels[0] if getattr(result, "channels", None) else None
        alternative = channel.alternatives[0] if channel and getattr(channel, "alternatives", None) else None
        transcript = str(getattr(alternative, "transcript", "") or "")
        segments = []
        for word in getattr(alternative, "words", []) or []:
            segments.append(
                {
                    "start": float(getattr(word, "start", 0.0) or 0.0),
                    "end": float(getattr(word, "end", 0.0) or 0.0),
                    "text": str(getattr(word, "word", "")).strip(),
                }
            )
        return TranscriptionResult(
            text=transcript,
            language=str(getattr(channel, "detected_language", None) or language or "unknown"),
            segments=segments,
            confidence=float(getattr(alternative, "confidence", 0.0) or 0.0),
            duration_seconds=float(getattr(getattr(response, "metadata", None), "duration", 0.0) or 0.0),
            provider=self.provider_name,
            raw_response=str(response),
        )

    def is_available(self) -> bool:
        return bool(self.api_key) and importlib.util.find_spec("deepgram") is not None


class AssemblyAISTTProvider(SpeechToTextProvider):
    """AssemblyAI STT provider."""

    provider_name = "assemblyai"

    def __init__(self, **_: Any) -> None:
        self.api_key = os.environ.get("ASSEMBLYAI_API_KEY", "").strip()

    def transcribe(self, audio_path: str, *, language: str | None = None) -> TranscriptionResult:
        try:
            import assemblyai as aai
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("assemblyai package is not installed.") from exc
        aai.settings.api_key = self.api_key
        config = aai.TranscriptionConfig(language_code=language) if language else aai.TranscriptionConfig()
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path, config=config)
        segments = [
            {
                "start": float(getattr(utterance, "start", 0) or 0) / 1000.0,
                "end": float(getattr(utterance, "end", 0) or 0) / 1000.0,
                "text": str(getattr(utterance, "text", "")).strip(),
            }
            for utterance in getattr(transcript, "utterances", []) or []
        ]
        return TranscriptionResult(
            text=str(getattr(transcript, "text", "") or ""),
            language=str(language or "unknown"),
            segments=segments,
            confidence=float(getattr(transcript, "confidence", 0.0) or 0.0),
            duration_seconds=float(getattr(transcript, "audio_duration", 0) or 0) / 1000.0,
            provider=self.provider_name,
            raw_response=str(getattr(transcript, "json_response", transcript)),
        )

    def is_available(self) -> bool:
        return bool(self.api_key) and importlib.util.find_spec("assemblyai") is not None


class GoogleSTTProvider(SpeechToTextProvider):
    """Google Cloud Speech-to-Text provider."""

    provider_name = "google_stt"

    def __init__(self, **_: Any) -> None:
        self.credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

    def transcribe(self, audio_path: str, *, language: str | None = None) -> TranscriptionResult:
        try:
            from google.cloud import speech_v2 as speech
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("google-cloud-speech package is not installed.") from exc
        client = speech.SpeechClient()
        with Path(audio_path).open("rb") as handle:
            audio_content = handle.read()
        config = speech.RecognitionConfig(
            auto_decoding_config=speech.AutoDetectDecodingConfig(),
            language_codes=[language or "en-US"],
            model="long",
        )
        request = speech.RecognizeRequest(config=config, content=audio_content)
        response = client.recognize(request=request)
        full_text_parts: list[str] = []
        for result in getattr(response, "results", []) or []:
            alternatives = getattr(result, "alternatives", []) or []
            if alternatives:
                full_text_parts.append(str(getattr(alternatives[0], "transcript", "")).strip())
        return TranscriptionResult(
            text=" ".join(part for part in full_text_parts if part).strip(),
            language=str(language or "en-US"),
            segments=[],
            confidence=0.85,
            duration_seconds=0.0,
            provider=self.provider_name,
            raw_response=str(response),
        )

    def is_available(self) -> bool:
        return (
            bool(self.credentials_path)
            and Path(self.credentials_path).is_file()
            and importlib.util.find_spec("google.cloud.speech_v2") is not None
        )


class OpenAITTSProvider(TextToSpeechProvider):
    """OpenAI TTS provider."""

    provider_name = "openai_tts"

    def __init__(self, model: str = "tts-1", voice: str = "nova", **_: Any) -> None:
        self.model = model
        self.default_voice = voice
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip()

    def synthesize(self, text: str, *, output_path: str, voice: str | None = None) -> SpeechResult:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("openai package is not installed.") from exc
        client = OpenAI(api_key=self.api_key)
        fmt = Path(output_path).suffix.lstrip(".") or "mp3"
        response = client.audio.speech.create(
            model=self.model,
            voice=voice or self.default_voice,
            input=text,
            response_format=fmt,
        )
        response.stream_to_file(output_path)
        return SpeechResult(
            audio_path=str(output_path),
            format=fmt,
            duration_seconds=0.0,
            provider=self.provider_name,
        )

    def is_available(self) -> bool:
        return bool(self.api_key) and importlib.util.find_spec("openai") is not None


class ElevenLabsTTSProvider(TextToSpeechProvider):
    """ElevenLabs TTS provider."""

    provider_name = "elevenlabs"

    def __init__(self, model: str = "eleven_v3", voice: str = "Rachel", **_: Any) -> None:
        self.model = model
        self.default_voice = voice
        self.api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()

    def synthesize(self, text: str, *, output_path: str, voice: str | None = None) -> SpeechResult:
        try:
            from elevenlabs import ElevenLabs, save
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("elevenlabs package is not installed.") from exc
        client = ElevenLabs(api_key=self.api_key)
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=voice or self.default_voice,
            model_id=self.model,
            output_format="mp3_44100_128",
        )
        save(audio, output_path)
        return SpeechResult(
            audio_path=str(output_path),
            format="mp3",
            duration_seconds=0.0,
            provider=self.provider_name,
        )

    def is_available(self) -> bool:
        return bool(self.api_key) and importlib.util.find_spec("elevenlabs") is not None


class MeshyText3DProvider(TextTo3DProvider):
    """Meshy text-to-3D provider."""

    provider_name = "meshy_text3d"
    API_BASE = "https://api.meshy.ai/openapi/v2"

    def __init__(self, timeout: float = 300.0, **_: Any) -> None:
        self.timeout = float(timeout)
        self.api_key = os.environ.get("MESHY_API_KEY", "").strip()

    def generate(self, prompt: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        create_resp = _rest_request(
            "POST",
            f"{self.API_BASE}/text-to-3d",
            api_key=self.api_key,
            json_data={"mode": "preview", "prompt": prompt, "art_style": "realistic"},
        )
        task_id = str(create_resp.get("result") or create_resp.get("task_id") or "")

        def check() -> dict[str, Any]:
            return _rest_request("GET", f"{self.API_BASE}/text-to-3d/{task_id}", api_key=self.api_key)

        result = _poll_until_complete(check, timeout=self.timeout)
        model_urls = result.get("model_urls", {}) if isinstance(result.get("model_urls"), dict) else {}
        model_url = str(model_urls.get(format) or model_urls.get("glb") or "")
        _download_file(model_url, output_path)
        return Mesh3DResult(str(output_path), format, task_id, self.provider_name, str(result))

    def is_available(self) -> bool:
        return bool(self.api_key)


class TripoText3DProvider(TextTo3DProvider):
    """Tripo text-to-3D provider."""

    provider_name = "tripo_text3d"
    API_BASE = "https://api.tripo3d.ai/v2/openapi"

    def __init__(self, timeout: float = 300.0, **_: Any) -> None:
        self.timeout = float(timeout)
        self.api_key = os.environ.get("TRIPO_API_KEY", "").strip()

    def generate(self, prompt: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        create_resp = _rest_request(
            "POST",
            f"{self.API_BASE}/task",
            api_key=self.api_key,
            json_data={"type": "text_to_model", "prompt": prompt},
        )
        task_id = str((create_resp.get("data") or {}).get("task_id", ""))

        def check() -> dict[str, Any]:
            return _rest_request("GET", f"{self.API_BASE}/task/{task_id}", api_key=self.api_key)

        result = _poll_until_complete(check, timeout=self.timeout)
        model_url = str((((result.get("data") or {}).get("output") or {}).get("model")) or "")
        _download_file(model_url, output_path)
        return Mesh3DResult(str(output_path), format, task_id, self.provider_name, str(result))

    def is_available(self) -> bool:
        return bool(self.api_key)


class RodinText3DProvider(TextTo3DProvider):
    """Hyper3D Rodin text-to-3D provider."""

    provider_name = "rodin_text3d"
    API_BASE = "https://api.hyper3d.com/api/v2"

    def __init__(self, timeout: float = 300.0, **_: Any) -> None:
        self.timeout = float(timeout)
        self.api_key = os.environ.get("HYPER3D_API_KEY", "").strip()

    def generate(self, prompt: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        create_resp = _rest_request(
            "POST",
            f"{self.API_BASE}/rodin",
            api_key=self.api_key,
            json_data={"prompt": prompt},
        )
        task_id = str(create_resp.get("task_id", ""))

        def check() -> dict[str, Any]:
            return _rest_request("GET", f"{self.API_BASE}/rodin/{task_id}", api_key=self.api_key)

        result = _poll_until_complete(check, timeout=self.timeout)
        output = result.get("output", {}) if isinstance(result.get("output"), dict) else {}
        model_url = str(output.get(format) or output.get("glb") or "")
        _download_file(model_url, output_path)
        return Mesh3DResult(str(output_path), format, task_id, self.provider_name, str(result))

    def is_available(self) -> bool:
        return bool(self.api_key)


class CSMText3DProvider(TextTo3DProvider):
    """Common Sense Machines text-to-3D provider."""

    provider_name = "csm_text3d"

    def __init__(self, timeout: float = 300.0, **_: Any) -> None:
        self.timeout = float(timeout)
        self.api_key = os.environ.get("CSM_API_KEY", "").strip()

    def generate(self, prompt: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        try:
            from csm import CSMClient
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("csm package is not installed.") from exc
        client = CSMClient(api_key=self.api_key)
        session = client.text_to_3d(prompt=prompt)
        session.download(format=format, path=output_path)
        return Mesh3DResult(str(output_path), format, str(getattr(session, "id", "")), self.provider_name, str(session))

    def is_available(self) -> bool:
        return bool(self.api_key) and importlib.util.find_spec("csm") is not None


class MeshyImage3DProvider(ImageTo3DProvider):
    """Meshy image-to-3D provider."""

    provider_name = "meshy_image3d"
    API_BASE = "https://api.meshy.ai/openapi/v2"

    def __init__(self, timeout: float = 300.0, **_: Any) -> None:
        self.timeout = float(timeout)
        self.api_key = os.environ.get("MESHY_API_KEY", "").strip()

    def generate(self, image_path: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        create_resp = _rest_request(
            "POST",
            f"{self.API_BASE}/image-to-3d",
            api_key=self.api_key,
            files={"image": image_path},
        )
        task_id = str(create_resp.get("result") or create_resp.get("task_id") or "")

        def check() -> dict[str, Any]:
            return _rest_request("GET", f"{self.API_BASE}/image-to-3d/{task_id}", api_key=self.api_key)

        result = _poll_until_complete(check, timeout=self.timeout)
        model_urls = result.get("model_urls", {}) if isinstance(result.get("model_urls"), dict) else {}
        model_url = str(model_urls.get(format) or model_urls.get("glb") or "")
        _download_file(model_url, output_path)
        return Mesh3DResult(str(output_path), format, task_id, self.provider_name, str(result))

    def is_available(self) -> bool:
        return bool(self.api_key)


class TripoImage3DProvider(ImageTo3DProvider):
    """Tripo image-to-3D provider."""

    provider_name = "tripo_image3d"
    API_BASE = "https://api.tripo3d.ai/v2/openapi"

    def __init__(self, timeout: float = 300.0, **_: Any) -> None:
        self.timeout = float(timeout)
        self.api_key = os.environ.get("TRIPO_API_KEY", "").strip()

    def generate(self, image_path: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        upload_resp = _rest_request(
            "POST",
            f"{self.API_BASE}/upload",
            api_key=self.api_key,
            files={"file": image_path},
        )
        image_token = str((upload_resp.get("data") or {}).get("image_token", ""))
        create_resp = _rest_request(
            "POST",
            f"{self.API_BASE}/task",
            api_key=self.api_key,
            json_data={"type": "image_to_model", "file": {"type": "png", "file_token": image_token}},
        )
        task_id = str((create_resp.get("data") or {}).get("task_id", ""))

        def check() -> dict[str, Any]:
            return _rest_request("GET", f"{self.API_BASE}/task/{task_id}", api_key=self.api_key)

        result = _poll_until_complete(check, timeout=self.timeout)
        model_url = str((((result.get("data") or {}).get("output") or {}).get("model")) or "")
        _download_file(model_url, output_path)
        return Mesh3DResult(str(output_path), format, task_id, self.provider_name, str(result))

    def is_available(self) -> bool:
        return bool(self.api_key)


class StabilityImage3DProvider(ImageTo3DProvider):
    """Stability SF3D provider."""

    provider_name = "stability_image3d"
    API_BASE = "https://api.stability.ai"

    def __init__(self, **_: Any) -> None:
        self.api_key = os.environ.get("STABILITY_API_KEY", "").strip()

    def generate(self, image_path: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        response_bytes = _rest_request_binary(
            "POST",
            f"{self.API_BASE}/v2beta/3d/stable-fast-3d",
            api_key=self.api_key,
            files={"image": image_path},
        )
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response_bytes)
        return Mesh3DResult(str(output_path), "glb", "sync", self.provider_name, "")

    def is_available(self) -> bool:
        return bool(self.api_key)


class RodinImage3DProvider(ImageTo3DProvider):
    """Hyper3D Rodin image-to-3D provider."""

    provider_name = "rodin_image3d"
    API_BASE = "https://api.hyper3d.com/api/v2"

    def __init__(self, timeout: float = 300.0, **_: Any) -> None:
        self.timeout = float(timeout)
        self.api_key = os.environ.get("HYPER3D_API_KEY", "").strip()

    def generate(self, image_path: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        create_resp = _rest_request(
            "POST",
            f"{self.API_BASE}/rodin",
            api_key=self.api_key,
            files={"images": image_path},
        )
        task_id = str(create_resp.get("task_id", ""))

        def check() -> dict[str, Any]:
            return _rest_request("GET", f"{self.API_BASE}/rodin/{task_id}", api_key=self.api_key)

        result = _poll_until_complete(check, timeout=self.timeout)
        output = result.get("output", {}) if isinstance(result.get("output"), dict) else {}
        model_url = str(output.get(format) or output.get("glb") or "")
        _download_file(model_url, output_path)
        return Mesh3DResult(str(output_path), format, task_id, self.provider_name, str(result))

    def is_available(self) -> bool:
        return bool(self.api_key)


class CSMImage3DProvider(ImageTo3DProvider):
    """Common Sense Machines image-to-3D provider."""

    provider_name = "csm_image3d"

    def __init__(self, timeout: float = 300.0, **_: Any) -> None:
        self.timeout = float(timeout)
        self.api_key = os.environ.get("CSM_API_KEY", "").strip()

    def generate(self, image_path: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        try:
            from csm import CSMClient
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("csm package is not installed.") from exc
        client = CSMClient(api_key=self.api_key)
        session = client.image_to_3d(image_path=image_path)
        session.download(format=format, path=output_path)
        return Mesh3DResult(str(output_path), format, str(getattr(session, "id", "")), self.provider_name, str(session))

    def is_available(self) -> bool:
        return bool(self.api_key) and importlib.util.find_spec("csm") is not None


def create_stt_provider(name: str | None = None, **kwargs: Any) -> SpeechToTextProvider:
    """Create speech-to-text provider by name or auto-detect."""
    providers: dict[str, type[SpeechToTextProvider]] = {
        "whisper_local": WhisperLocalProvider,
        "vibe_server": VibeServerProvider,
        "openai_whisper": OpenAIWhisperProvider,
        "deepgram": DeepgramSTTProvider,
        "assemblyai": AssemblyAISTTProvider,
        "google_stt": GoogleSTTProvider,
    }
    requested = str(name or "auto").strip().lower()
    if requested and requested != "auto":
        cls = providers.get(requested)
        if cls is None:
            raise ValueError(f"Unknown STT provider: {name}. Available: {sorted(providers.keys())}")
        return cls(**kwargs)
    for candidate in ("whisper_local", "vibe_server", "openai_whisper", "deepgram", "assemblyai", "google_stt"):
        provider = providers[candidate](**kwargs)
        if provider.is_available():
            return provider
    return WhisperLocalProvider(**kwargs)


def create_tts_provider(name: str | None = None, **kwargs: Any) -> TextToSpeechProvider:
    """Create text-to-speech provider by name or auto-detect."""
    providers: dict[str, type[TextToSpeechProvider]] = {
        "openai_tts": OpenAITTSProvider,
        "elevenlabs": ElevenLabsTTSProvider,
    }
    requested = str(name or "auto").strip().lower()
    if requested and requested != "auto":
        cls = providers.get(requested)
        if cls is None:
            raise ValueError(f"Unknown TTS provider: {name}. Available: {sorted(providers.keys())}")
        return cls(**kwargs)
    for candidate in ("openai_tts", "elevenlabs"):
        provider = providers[candidate](**kwargs)
        if provider.is_available():
            return provider
    return OpenAITTSProvider(**kwargs)


def create_text3d_provider(name: str | None = None, **kwargs: Any) -> TextTo3DProvider:
    """Create text-to-3D provider by name or auto-detect."""
    providers: dict[str, type[TextTo3DProvider]] = {
        "meshy_text3d": MeshyText3DProvider,
        "tripo_text3d": TripoText3DProvider,
        "rodin_text3d": RodinText3DProvider,
        "csm_text3d": CSMText3DProvider,
    }
    requested = str(name or "auto").strip().lower()
    if requested and requested != "auto":
        cls = providers.get(requested)
        if cls is None:
            raise ValueError(f"Unknown text-to-3D provider: {name}. Available: {sorted(providers.keys())}")
        return cls(**kwargs)
    for candidate in ("meshy_text3d", "tripo_text3d", "rodin_text3d", "csm_text3d"):
        provider = providers[candidate](**kwargs)
        if provider.is_available():
            return provider
    return MeshyText3DProvider(**kwargs)


def create_image3d_provider(name: str | None = None, **kwargs: Any) -> ImageTo3DProvider:
    """Create image-to-3D provider by name or auto-detect."""
    providers: dict[str, type[ImageTo3DProvider]] = {
        "meshy_image3d": MeshyImage3DProvider,
        "tripo_image3d": TripoImage3DProvider,
        "stability_image3d": StabilityImage3DProvider,
        "rodin_image3d": RodinImage3DProvider,
        "csm_image3d": CSMImage3DProvider,
    }
    requested = str(name or "auto").strip().lower()
    if requested and requested != "auto":
        cls = providers.get(requested)
        if cls is None:
            raise ValueError(f"Unknown image-to-3D provider: {name}. Available: {sorted(providers.keys())}")
        return cls(**kwargs)
    for candidate in ("stability_image3d", "meshy_image3d", "tripo_image3d", "rodin_image3d", "csm_image3d"):
        provider = providers[candidate](**kwargs)
        if provider.is_available():
            return provider
    return MeshyImage3DProvider(**kwargs)


__all__ = [
    "AssemblyAISTTProvider",
    "CSMImage3DProvider",
    "CSMText3DProvider",
    "DeepgramSTTProvider",
    "ElevenLabsTTSProvider",
    "GoogleSTTProvider",
    "ImageTo3DProvider",
    "Mesh3DResult",
    "MeshyImage3DProvider",
    "MeshyText3DProvider",
    "ModalProvider",
    "OpenAITTSProvider",
    "OpenAIWhisperProvider",
    "RodinImage3DProvider",
    "RodinText3DProvider",
    "SpeechResult",
    "SpeechToTextProvider",
    "StabilityImage3DProvider",
    "TextTo3DProvider",
    "TextToSpeechProvider",
    "TranscriptionResult",
    "TripoImage3DProvider",
    "TripoText3DProvider",
    "VibeServerProvider",
    "WhisperLocalProvider",
    "_download_file",
    "_poll_until_complete",
    "_rest_request",
    "_rest_request_binary",
    "create_image3d_provider",
    "create_stt_provider",
    "create_text3d_provider",
    "create_tts_provider",
]
