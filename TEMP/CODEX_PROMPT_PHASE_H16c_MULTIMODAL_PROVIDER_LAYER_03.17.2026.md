# Phase H16c — Multi-Modal Provider Layer

**Depends on:** H16b (Multi-Provider Extension)
**Creates:** `knowledge3d/tools/modal_providers.py`
**Tests:** `tests/test_modal_providers.py`

---

## Objective

Build a **bidirectional multi-modal I/O layer** alongside the existing text augmentation providers. K3D is a multi-modal AI — it must consume FROM and produce TO every major external standard:

| Direction | Modality | Purpose |
|-----------|----------|---------|
| **Input** | Audio → Text | Transcribe audio/video into knowledge content |
| **Input** | Image → 3D | Generate 3D assets from reference images |
| **Input** | Text → 3D | Generate 3D assets from descriptions |
| **Output** | Text → Speech | Produce audio from knowledge content |
| **Output** | Knowledge → 3D GLB | Already done (export_house.py) |
| **Output** | Knowledge → DOM | Already done (domOps) |

This is the **backwards compatibility bridge** — K3D's internal Galaxy representation can interface with every external format out of the box.

---

## Architecture

New file: `knowledge3d/tools/modal_providers.py`

Four abstract base classes, each with concrete provider implementations. Same patterns as `augmentation_providers.py`: lazy imports, env-var API keys, ingestion-path only, `is_available()` checks.

```
ModalProvider (marker base)
├── SpeechToTextProvider (ABC)
│   ├── WhisperLocalProvider      — faster-whisper (native Python, GPU)
│   ├── VibeServerProvider        — Vibe --server (OpenAI-compatible on localhost:3022)
│   ├── OpenAIWhisperProvider     — OpenAI Whisper API
│   ├── DeepgramSTTProvider       — Deepgram nova-3
│   ├── AssemblyAISTTProvider     — AssemblyAI
│   └── GoogleSTTProvider         — Google Cloud Speech-to-Text
│
├── TextToSpeechProvider (ABC)
│   ├── OpenAITTSProvider         — OpenAI TTS (tts-1, gpt-4o-mini-tts)
│   └── ElevenLabsTTSProvider     — ElevenLabs (eleven_v3)
│
├── TextTo3DProvider (ABC)
│   ├── MeshyText3DProvider       — Meshy AI text-to-3D
│   ├── TripoText3DProvider       — Tripo AI text-to-3D
│   ├── RodinText3DProvider       — Hyper3D Rodin text-to-3D
│   └── CSMText3DProvider         — Common Sense Machines text-to-3D
│
└── ImageTo3DProvider (ABC)
    ├── MeshyImage3DProvider      — Meshy AI image-to-3D
    ├── TripoImage3DProvider      — Tripo AI image-to-3D
    ├── StabilityImage3DProvider  — Stability AI SF3D (synchronous, GLB)
    ├── RodinImage3DProvider      — Hyper3D Rodin image-to-3D
    └── CSMImage3DProvider        — Common Sense Machines image-to-3D
```

---

## Base Classes

### ModalProvider (marker)

```python
class ModalProvider(ABC):
    """Marker base for all modal conversion providers."""
    provider_name = "modal"
    modality = "unknown"  # "stt", "tts", "text_to_3d", "image_to_3d"

    @abstractmethod
    def is_available(self) -> bool:
        """Whether provider credentials/runtime are available."""
```

### SpeechToTextProvider

```python
@dataclass
class TranscriptionResult:
    """Result of a speech-to-text conversion."""
    text: str                          # Full transcript
    language: str                      # Detected language code (e.g., "en", "pt")
    segments: list[dict[str, Any]]     # [{start: float, end: float, text: str}, ...]
    confidence: float                  # 0.0-1.0
    duration_seconds: float            # Audio duration
    provider: str                      # Provider name
    raw_response: str                  # Raw API response for debugging

class SpeechToTextProvider(ModalProvider):
    """Base for audio/video → text transcription."""
    modality = "stt"

    @abstractmethod
    def transcribe(self, audio_path: str, *, language: str | None = None) -> TranscriptionResult:
        """Transcribe audio/video file to text with timestamps."""
```

### TextToSpeechProvider

```python
@dataclass
class SpeechResult:
    """Result of a text-to-speech conversion."""
    audio_path: str          # Path to generated audio file
    format: str              # "mp3", "wav", "opus", etc.
    duration_seconds: float  # Approximate duration
    provider: str

class TextToSpeechProvider(ModalProvider):
    """Base for text → audio speech synthesis."""
    modality = "tts"

    @abstractmethod
    def synthesize(self, text: str, *, output_path: str, voice: str | None = None) -> SpeechResult:
        """Synthesize speech from text, write to output_path."""
```

### TextTo3DProvider / ImageTo3DProvider

```python
@dataclass
class Mesh3DResult:
    """Result of a 3D generation task."""
    mesh_path: str           # Path to downloaded GLB/OBJ file
    format: str              # "glb", "fbx", "obj"
    task_id: str             # Provider's task/job ID (for async polling)
    provider: str
    raw_response: str

class TextTo3DProvider(ModalProvider):
    """Base for text description → 3D mesh generation."""
    modality = "text_to_3d"

    @abstractmethod
    def generate(self, prompt: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        """Generate 3D mesh from text prompt. Blocks until complete (handles polling internally)."""

class ImageTo3DProvider(ModalProvider):
    """Base for image → 3D mesh generation."""
    modality = "image_to_3d"

    @abstractmethod
    def generate(self, image_path: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        """Generate 3D mesh from reference image. Blocks until complete."""
```

---

## Speech-to-Text Providers

### 1. WhisperLocalProvider (faster-whisper, GPU-native)

```python
class WhisperLocalProvider(SpeechToTextProvider):
    """Local GPU-accelerated transcription via faster-whisper."""
    provider_name = "whisper_local"

    def __init__(self, model_size: str = "large-v3", device: str = "cuda",
                 compute_type: str = "float16", **_: Any) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None  # Lazy-loaded

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device=self.device,
                                        compute_type=self.compute_type)
        return self._model

    def transcribe(self, audio_path: str, *, language: str | None = None) -> TranscriptionResult:
        model = self._get_model()
        segments_iter, info = model.transcribe(audio_path, beam_size=5,
                                                language=language,
                                                word_timestamps=True)
        segments = []
        full_text_parts = []
        for seg in segments_iter:
            segments.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
            full_text_parts.append(seg.text.strip())
        return TranscriptionResult(
            text=" ".join(full_text_parts),
            language=info.language or language or "unknown",
            segments=segments,
            confidence=1.0 - (info.language_probability or 0.0),  # Approximate
            duration_seconds=info.duration or 0.0,
            provider=self.provider_name,
            raw_response="",
        )

    def is_available(self) -> bool:
        return importlib.util.find_spec("faster_whisper") is not None
```

**Install:** `pip install faster-whisper` (uses CTranslate2, CUDA 12 + cuDNN 9, ~4x faster than openai-whisper, less VRAM)

### 2. VibeServerProvider (Vibe --server, OpenAI-compatible)

Daniel's machine has Vibe v3.0.11 installed at `/usr/bin/vibe`. Its `--server` mode exposes an OpenAI-compatible transcription API on port 3022.

```python
class VibeServerProvider(SpeechToTextProvider):
    """Vibe local server — OpenAI-compatible whisper API on localhost."""
    provider_name = "vibe_server"

    def __init__(self, base_url: str = "http://localhost:3022", **_: Any) -> None:
        self.base_url = base_url

    def transcribe(self, audio_path: str, *, language: str | None = None) -> TranscriptionResult:
        from openai import OpenAI
        client = OpenAI(api_key="local", base_url=self.base_url)
        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        # Parse verbose_json response
        segments = []
        if hasattr(response, "segments"):
            for seg in response.segments:
                segments.append({"start": seg.get("start", 0), "end": seg.get("end", 0),
                                 "text": seg.get("text", "").strip()})
        return TranscriptionResult(
            text=response.text if hasattr(response, "text") else str(response),
            language=getattr(response, "language", language or "unknown"),
            segments=segments,
            confidence=0.8,  # Vibe doesn't return confidence
            duration_seconds=getattr(response, "duration", 0.0),
            provider=self.provider_name,
            raw_response=str(response),
        )

    def is_available(self) -> bool:
        """Check if Vibe server is responding on its port."""
        import urllib.request
        try:
            req = urllib.request.Request(f"{self.base_url}/docs", method="HEAD")
            urllib.request.urlopen(req, timeout=2)
            return True
        except Exception:
            return False
```

**Usage:** User starts Vibe server manually: `WEBKIT_DISABLE_COMPOSITING_MODE=1 /usr/bin/vibe --server`
**Note:** Vibe accepts one transcription at a time (429 on concurrent). Queue externally if needed.

### 3. OpenAIWhisperProvider (cloud)

```python
class OpenAIWhisperProvider(SpeechToTextProvider):
    """OpenAI Whisper cloud API."""
    provider_name = "openai_whisper"

    def __init__(self, model: str = "whisper-1", **_: Any) -> None:
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip()

    def transcribe(self, audio_path: str, *, language: str | None = None) -> TranscriptionResult:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model=self.model,
                file=f,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        segments = []
        if hasattr(response, "segments"):
            for seg in response.segments:
                segments.append({"start": seg.get("start", 0), "end": seg.get("end", 0),
                                 "text": seg.get("text", "").strip()})
        return TranscriptionResult(
            text=response.text if hasattr(response, "text") else str(response),
            language=getattr(response, "language", language or "unknown"),
            segments=segments,
            confidence=0.9,
            duration_seconds=getattr(response, "duration", 0.0),
            provider=self.provider_name,
            raw_response=str(response),
        )

    def is_available(self) -> bool:
        return bool(self.api_key) and importlib.util.find_spec("openai") is not None
```

### 4. DeepgramSTTProvider

```python
class DeepgramSTTProvider(SpeechToTextProvider):
    """Deepgram nova-3 speech-to-text."""
    provider_name = "deepgram"

    def __init__(self, model: str = "nova-3", **_: Any) -> None:
        self.model = model
        self.api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()

    def transcribe(self, audio_path: str, *, language: str | None = None) -> TranscriptionResult:
        from deepgram import DeepgramClient, PrerecordedOptions, FileSource
        client = DeepgramClient(self.api_key)
        with open(audio_path, "rb") as f:
            payload: FileSource = {"buffer": f.read()}
        options = PrerecordedOptions(model=self.model, smart_format=True,
                                      language=language, utterances=True)
        response = client.listen.rest.v("1").transcribe_file(payload, options)
        result = response.results
        transcript = result.channels[0].alternatives[0].transcript if result.channels else ""
        segments = []
        if hasattr(result.channels[0].alternatives[0], "words"):
            # Group words into segments by utterance or sentence
            for word in result.channels[0].alternatives[0].words:
                segments.append({"start": word.start, "end": word.end, "text": word.word})
        confidence = result.channels[0].alternatives[0].confidence if result.channels else 0.0
        return TranscriptionResult(
            text=transcript,
            language=result.channels[0].detected_language if result.channels else language or "unknown",
            segments=segments,
            confidence=confidence,
            duration_seconds=response.metadata.duration if hasattr(response, "metadata") else 0.0,
            provider=self.provider_name,
            raw_response=str(response),
        )

    def is_available(self) -> bool:
        return bool(self.api_key) and importlib.util.find_spec("deepgram") is not None
```

### 5. AssemblyAISTTProvider

```python
class AssemblyAISTTProvider(SpeechToTextProvider):
    """AssemblyAI speech-to-text."""
    provider_name = "assemblyai"

    def __init__(self, **_: Any) -> None:
        self.api_key = os.environ.get("ASSEMBLYAI_API_KEY", "").strip()

    def transcribe(self, audio_path: str, *, language: str | None = None) -> TranscriptionResult:
        import assemblyai as aai
        aai.settings.api_key = self.api_key
        config = aai.TranscriptionConfig(language_code=language) if language else aai.TranscriptionConfig()
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path, config=config)
        segments = []
        if transcript.utterances:
            for utt in transcript.utterances:
                segments.append({"start": utt.start / 1000.0, "end": utt.end / 1000.0,
                                 "text": utt.text})
        return TranscriptionResult(
            text=transcript.text or "",
            language=language or "unknown",
            segments=segments,
            confidence=transcript.confidence or 0.0,
            duration_seconds=(transcript.audio_duration or 0) / 1000.0,
            provider=self.provider_name,
            raw_response=str(transcript.json_response),
        )

    def is_available(self) -> bool:
        return bool(self.api_key) and importlib.util.find_spec("assemblyai") is not None
```

### 6. GoogleSTTProvider

```python
class GoogleSTTProvider(SpeechToTextProvider):
    """Google Cloud Speech-to-Text v2."""
    provider_name = "google_stt"

    def __init__(self, **_: Any) -> None:
        self.credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

    def transcribe(self, audio_path: str, *, language: str | None = None) -> TranscriptionResult:
        from google.cloud import speech_v2 as speech
        client = speech.SpeechClient()
        with open(audio_path, "rb") as f:
            audio_content = f.read()
        config = speech.RecognitionConfig(
            auto_decoding_config=speech.AutoDetectDecodingConfig(),
            language_codes=[language or "en-US"],
            model="long",
        )
        request = speech.RecognizeRequest(config=config, content=audio_content)
        response = client.recognize(request=request)
        full_text = ""
        segments = []
        for result in response.results:
            alt = result.alternatives[0] if result.alternatives else None
            if alt:
                full_text += alt.transcript + " "
        return TranscriptionResult(
            text=full_text.strip(),
            language=language or "en-US",
            segments=segments,
            confidence=0.85,
            duration_seconds=0.0,
            provider=self.provider_name,
            raw_response=str(response),
        )

    def is_available(self) -> bool:
        return bool(self.credentials_path) and Path(self.credentials_path).is_file() and \
               importlib.util.find_spec("google.cloud.speech_v2") is not None
```

---

## Text-to-Speech Providers

### 7. OpenAITTSProvider

```python
class OpenAITTSProvider(TextToSpeechProvider):
    """OpenAI text-to-speech."""
    provider_name = "openai_tts"

    def __init__(self, model: str = "tts-1", voice: str = "nova", **_: Any) -> None:
        self.model = model
        self.default_voice = voice
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip()

    def synthesize(self, text: str, *, output_path: str, voice: str | None = None) -> SpeechResult:
        from openai import OpenAI
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
            audio_path=output_path,
            format=fmt,
            duration_seconds=0.0,  # Not returned by API; caller can probe
            provider=self.provider_name,
        )

    def is_available(self) -> bool:
        return bool(self.api_key) and importlib.util.find_spec("openai") is not None
```

### 8. ElevenLabsTTSProvider

```python
class ElevenLabsTTSProvider(TextToSpeechProvider):
    """ElevenLabs text-to-speech."""
    provider_name = "elevenlabs"

    def __init__(self, model: str = "eleven_v3", voice: str = "Rachel", **_: Any) -> None:
        self.model = model
        self.default_voice = voice
        self.api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()

    def synthesize(self, text: str, *, output_path: str, voice: str | None = None) -> SpeechResult:
        from elevenlabs import ElevenLabs, save
        client = ElevenLabs(api_key=self.api_key)
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=voice or self.default_voice,
            model_id=self.model,
            output_format="mp3_44100_128",
        )
        save(audio, output_path)
        return SpeechResult(
            audio_path=output_path,
            format="mp3",
            duration_seconds=0.0,
            provider=self.provider_name,
        )

    def is_available(self) -> bool:
        return bool(self.api_key) and importlib.util.find_spec("elevenlabs") is not None
```

---

## 3D Generation Providers — Shared Async Polling Helper

Most 3D generation APIs are async (POST creates task → poll until done → download GLB). Factor this into a helper:

```python
def _poll_until_complete(
    check_fn: Callable[[], dict[str, Any]],
    *,
    timeout: float = 300.0,
    interval: float = 5.0,
    terminal_statuses: tuple[str, ...] = ("SUCCEEDED", "completed", "done"),
    failure_statuses: tuple[str, ...] = ("FAILED", "failed", "error", "expired"),
) -> dict[str, Any]:
    """Poll check_fn() until status is terminal. Returns final response dict."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check_fn()
        status = str(result.get("status", "")).lower()
        for ts in terminal_statuses:
            if ts.lower() in status:
                return result
        for fs in failure_statuses:
            if fs.lower() in status:
                raise RuntimeError(f"3D generation failed: {result}")
        time.sleep(interval)
    raise TimeoutError(f"3D generation timed out after {timeout}s")
```

### Shared REST Helper

All 3D APIs use simple REST with Bearer auth. Factor a tiny helper:

```python
def _rest_request(
    method: str,
    url: str,
    *,
    api_key: str,
    json_data: dict | None = None,
    files: dict | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Make authenticated REST request, return parsed JSON."""
    import urllib.request
    import urllib.error
    # Implementation uses urllib.request to avoid adding requests as dependency.
    # For multipart/form-data (file uploads), use email.mime or a minimal multipart encoder.
    ...
```

**Important:** Use `urllib.request` (stdlib) for REST calls. Do NOT add `requests` as a dependency. For file uploads (image-to-3D), implement minimal multipart encoding with `email.mime` or manual boundary construction.

---

## Text-to-3D Providers

### 9. MeshyText3DProvider

```python
class MeshyText3DProvider(TextTo3DProvider):
    """Meshy AI text-to-3D."""
    provider_name = "meshy_text3d"

    API_BASE = "https://api.meshy.ai/openapi/v2"

    def __init__(self, timeout: float = 300.0, **_: Any) -> None:
        self.timeout = timeout
        self.api_key = os.environ.get("MESHY_API_KEY", "").strip()

    def generate(self, prompt: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        # POST /text-to-3d → task_id
        create_resp = _rest_request("POST", f"{self.API_BASE}/text-to-3d",
                                     api_key=self.api_key,
                                     json_data={"mode": "preview", "prompt": prompt,
                                                 "art_style": "realistic"})
        task_id = create_resp.get("result") or create_resp.get("task_id", "")

        # Poll GET /text-to-3d/{task_id}
        def check():
            return _rest_request("GET", f"{self.API_BASE}/text-to-3d/{task_id}",
                                  api_key=self.api_key)

        result = _poll_until_complete(check, timeout=self.timeout)
        model_url = result.get("model_urls", {}).get(format) or result.get("model_urls", {}).get("glb", "")
        # Download GLB
        _download_file(model_url, output_path)
        return Mesh3DResult(mesh_path=output_path, format=format, task_id=task_id,
                            provider=self.provider_name, raw_response=str(result))

    def is_available(self) -> bool:
        return bool(self.api_key)
```

### 10. TripoText3DProvider

```python
class TripoText3DProvider(TextTo3DProvider):
    """Tripo AI text-to-3D."""
    provider_name = "tripo_text3d"

    API_BASE = "https://api.tripo3d.ai/v2/openapi"

    def __init__(self, timeout: float = 300.0, **_: Any) -> None:
        self.timeout = timeout
        self.api_key = os.environ.get("TRIPO_API_KEY", "").strip()

    def generate(self, prompt: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        create_resp = _rest_request("POST", f"{self.API_BASE}/task",
                                     api_key=self.api_key,
                                     json_data={"type": "text_to_model", "prompt": prompt})
        task_id = create_resp.get("data", {}).get("task_id", "")

        def check():
            return _rest_request("GET", f"{self.API_BASE}/task/{task_id}",
                                  api_key=self.api_key)

        result = _poll_until_complete(check, timeout=self.timeout)
        model_url = result.get("data", {}).get("output", {}).get("model", "")
        _download_file(model_url, output_path)
        return Mesh3DResult(mesh_path=output_path, format=format, task_id=task_id,
                            provider=self.provider_name, raw_response=str(result))

    def is_available(self) -> bool:
        return bool(self.api_key)
```

### 11. RodinText3DProvider

```python
class RodinText3DProvider(TextTo3DProvider):
    """Hyper3D Rodin text-to-3D."""
    provider_name = "rodin_text3d"

    API_BASE = "https://api.hyper3d.com/api/v2"

    def __init__(self, timeout: float = 300.0, **_: Any) -> None:
        self.timeout = timeout
        self.api_key = os.environ.get("HYPER3D_API_KEY", "").strip()

    def generate(self, prompt: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        # POST /rodin with text prompt only (no images) → text-to-3D mode
        create_resp = _rest_request("POST", f"{self.API_BASE}/rodin",
                                     api_key=self.api_key,
                                     json_data={"prompt": prompt})
        task_id = create_resp.get("task_id", "")

        def check():
            return _rest_request("GET", f"{self.API_BASE}/rodin/{task_id}",
                                  api_key=self.api_key)

        result = _poll_until_complete(check, timeout=self.timeout)
        model_url = result.get("output", {}).get(format, "") or result.get("output", {}).get("glb", "")
        _download_file(model_url, output_path)
        return Mesh3DResult(mesh_path=output_path, format=format, task_id=task_id,
                            provider=self.provider_name, raw_response=str(result))

    def is_available(self) -> bool:
        return bool(self.api_key)
```

### 12. CSMText3DProvider

```python
class CSMText3DProvider(TextTo3DProvider):
    """Common Sense Machines text-to-3D."""
    provider_name = "csm_text3d"

    def __init__(self, timeout: float = 300.0, **_: Any) -> None:
        self.timeout = timeout
        self.api_key = os.environ.get("CSM_API_KEY", "").strip()

    def generate(self, prompt: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        from csm import CSMClient
        client = CSMClient(api_key=self.api_key)
        session = client.text_to_3d(prompt=prompt)
        # SDK handles polling internally
        mesh = session.download(format=format, path=output_path)
        return Mesh3DResult(mesh_path=output_path, format=format, task_id=session.id,
                            provider=self.provider_name, raw_response=str(session))

    def is_available(self) -> bool:
        return bool(self.api_key) and importlib.util.find_spec("csm") is not None
```

---

## Image-to-3D Providers

### 13. MeshyImage3DProvider

```python
class MeshyImage3DProvider(ImageTo3DProvider):
    """Meshy AI image-to-3D."""
    provider_name = "meshy_image3d"

    API_BASE = "https://api.meshy.ai/openapi/v2"

    def __init__(self, timeout: float = 300.0, **_: Any) -> None:
        self.timeout = timeout
        self.api_key = os.environ.get("MESHY_API_KEY", "").strip()

    def generate(self, image_path: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        # Upload image and create task
        create_resp = _rest_request("POST", f"{self.API_BASE}/image-to-3d",
                                     api_key=self.api_key,
                                     files={"image": image_path})
        task_id = create_resp.get("result") or create_resp.get("task_id", "")

        def check():
            return _rest_request("GET", f"{self.API_BASE}/image-to-3d/{task_id}",
                                  api_key=self.api_key)

        result = _poll_until_complete(check, timeout=self.timeout)
        model_url = result.get("model_urls", {}).get(format) or result.get("model_urls", {}).get("glb", "")
        _download_file(model_url, output_path)
        return Mesh3DResult(mesh_path=output_path, format=format, task_id=task_id,
                            provider=self.provider_name, raw_response=str(result))

    def is_available(self) -> bool:
        return bool(self.api_key)
```

### 14. TripoImage3DProvider

```python
class TripoImage3DProvider(ImageTo3DProvider):
    """Tripo AI image-to-3D."""
    provider_name = "tripo_image3d"

    API_BASE = "https://api.tripo3d.ai/v2/openapi"

    def __init__(self, timeout: float = 300.0, **_: Any) -> None:
        self.timeout = timeout
        self.api_key = os.environ.get("TRIPO_API_KEY", "").strip()

    def generate(self, image_path: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        # Upload image first, get image_token
        upload_resp = _rest_request("POST", f"{self.API_BASE}/upload",
                                     api_key=self.api_key,
                                     files={"file": image_path})
        image_token = upload_resp.get("data", {}).get("image_token", "")

        create_resp = _rest_request("POST", f"{self.API_BASE}/task",
                                     api_key=self.api_key,
                                     json_data={"type": "image_to_model",
                                                 "file": {"type": "png", "file_token": image_token}})
        task_id = create_resp.get("data", {}).get("task_id", "")

        def check():
            return _rest_request("GET", f"{self.API_BASE}/task/{task_id}",
                                  api_key=self.api_key)

        result = _poll_until_complete(check, timeout=self.timeout)
        model_url = result.get("data", {}).get("output", {}).get("model", "")
        _download_file(model_url, output_path)
        return Mesh3DResult(mesh_path=output_path, format=format, task_id=task_id,
                            provider=self.provider_name, raw_response=str(result))

    def is_available(self) -> bool:
        return bool(self.api_key)
```

### 15. StabilityImage3DProvider (Synchronous!)

```python
class StabilityImage3DProvider(ImageTo3DProvider):
    """Stability AI Stable Fast 3D — synchronous image-to-GLB."""
    provider_name = "stability_image3d"

    API_BASE = "https://api.stability.ai"

    def __init__(self, **_: Any) -> None:
        self.api_key = os.environ.get("STABILITY_API_KEY", "").strip()

    def generate(self, image_path: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        # POST multipart with image, returns GLB bytes directly (synchronous!)
        resp_bytes = _rest_request_binary("POST",
            f"{self.API_BASE}/v2beta/3d/stable-fast-3d",
            api_key=self.api_key,
            files={"image": image_path},
        )
        Path(output_path).write_bytes(resp_bytes)
        return Mesh3DResult(mesh_path=output_path, format="glb", task_id="sync",
                            provider=self.provider_name, raw_response="")

    def is_available(self) -> bool:
        return bool(self.api_key)
```

**Note:** SF3D is unique — it returns the GLB binary directly in the response (no polling). Need a `_rest_request_binary()` variant that returns raw bytes.

### 16. RodinImage3DProvider

```python
class RodinImage3DProvider(ImageTo3DProvider):
    """Hyper3D Rodin image-to-3D."""
    provider_name = "rodin_image3d"

    API_BASE = "https://api.hyper3d.com/api/v2"

    def __init__(self, timeout: float = 300.0, **_: Any) -> None:
        self.timeout = timeout
        self.api_key = os.environ.get("HYPER3D_API_KEY", "").strip()

    def generate(self, image_path: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        # POST /rodin with image upload → image-to-3D mode (auto-detected)
        create_resp = _rest_request("POST", f"{self.API_BASE}/rodin",
                                     api_key=self.api_key,
                                     files={"images": image_path})
        task_id = create_resp.get("task_id", "")

        def check():
            return _rest_request("GET", f"{self.API_BASE}/rodin/{task_id}",
                                  api_key=self.api_key)

        result = _poll_until_complete(check, timeout=self.timeout)
        model_url = result.get("output", {}).get(format, "") or result.get("output", {}).get("glb", "")
        _download_file(model_url, output_path)
        return Mesh3DResult(mesh_path=output_path, format=format, task_id=task_id,
                            provider=self.provider_name, raw_response=str(result))

    def is_available(self) -> bool:
        return bool(self.api_key)
```

### 17. CSMImage3DProvider

```python
class CSMImage3DProvider(ImageTo3DProvider):
    """Common Sense Machines image-to-3D."""
    provider_name = "csm_image3d"

    def __init__(self, timeout: float = 300.0, **_: Any) -> None:
        self.timeout = timeout
        self.api_key = os.environ.get("CSM_API_KEY", "").strip()

    def generate(self, image_path: str, *, output_path: str, format: str = "glb") -> Mesh3DResult:
        from csm import CSMClient
        client = CSMClient(api_key=self.api_key)
        session = client.image_to_3d(image_path=image_path)
        mesh = session.download(format=format, path=output_path)
        return Mesh3DResult(mesh_path=output_path, format=format, task_id=session.id,
                            provider=self.provider_name, raw_response=str(session))

    def is_available(self) -> bool:
        return bool(self.api_key) and importlib.util.find_spec("csm") is not None
```

---

## Factory Functions

```python
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
    # Auto: local first (free), then cloud
    for candidate in ("whisper_local", "vibe_server", "openai_whisper", "deepgram", "assemblyai", "google_stt"):
        provider = providers[candidate](**kwargs)
        if provider.is_available():
            return provider
    return WhisperLocalProvider(**kwargs)  # Fallback


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
```

---

## Environment Variables Summary

| Provider | Env Var | SDK Required | Modality |
|----------|---------|-------------|----------|
| WhisperLocal | (none) | `faster-whisper` | STT |
| VibeServer | (none — local) | `openai` | STT |
| OpenAI Whisper | `OPENAI_API_KEY` | `openai` | STT |
| Deepgram | `DEEPGRAM_API_KEY` | `deepgram-sdk` | STT |
| AssemblyAI | `ASSEMBLYAI_API_KEY` | `assemblyai` | STT |
| Google STT | `GOOGLE_APPLICATION_CREDENTIALS` | `google-cloud-speech` | STT |
| OpenAI TTS | `OPENAI_API_KEY` | `openai` | TTS |
| ElevenLabs | `ELEVENLABS_API_KEY` | `elevenlabs` | TTS |
| Meshy (text+image) | `MESHY_API_KEY` | (stdlib urllib) | 3D |
| Tripo (text+image) | `TRIPO_API_KEY` | (stdlib urllib) | 3D |
| Stability SF3D | `STABILITY_API_KEY` | (stdlib urllib) | Image→3D |
| Rodin/Hyper3D | `HYPER3D_API_KEY` | (stdlib urllib) | 3D |
| CSM | `CSM_API_KEY` | `csm-ai` | 3D |

**Note:** 3D REST providers use only `urllib.request` (stdlib) — no `requests` dependency. STT/TTS providers use their official Python SDKs for robustness.

---

## Vibe Integration Notes

- Binary: `/usr/bin/vibe` (v3.0.11, ELF 64-bit)
- Library path: `/usr/lib/vibe/` (locales)
- Launch env: `WEBKIT_DISABLE_COMPOSITING_MODE=1`
- Server mode: `WEBKIT_DISABLE_COMPOSITING_MODE=1 /usr/bin/vibe --server` → port 3022
- Uses whisper.cpp GGML models with Vulkan GPU acceleration
- Server is OpenAI-compatible: use `openai` SDK with `base_url="http://localhost:3022"`
- **Limitation:** One transcription at a time (429 on concurrent requests)
- `VibeServerProvider.is_available()` probes `http://localhost:3022/docs` with HEAD request

For batch/pipeline use, prefer `WhisperLocalProvider` (faster-whisper) which has native Python API and handles concurrency better. Vibe server is an alternative when faster-whisper is not installed or for AMD/Intel GPU machines.

---

## Tests

### Structure: `tests/test_modal_providers.py`

```python
# 1. All providers instantiate without error
def test_all_stt_providers_instantiate():
    for cls in (WhisperLocalProvider, VibeServerProvider, OpenAIWhisperProvider,
                DeepgramSTTProvider, AssemblyAISTTProvider, GoogleSTTProvider):
        provider = cls()
        assert isinstance(provider, SpeechToTextProvider)
        assert provider.modality == "stt"

def test_all_tts_providers_instantiate():
    for cls in (OpenAITTSProvider, ElevenLabsTTSProvider):
        provider = cls()
        assert isinstance(provider, TextToSpeechProvider)
        assert provider.modality == "tts"

def test_all_text3d_providers_instantiate():
    for cls in (MeshyText3DProvider, TripoText3DProvider, RodinText3DProvider, CSMText3DProvider):
        provider = cls()
        assert isinstance(provider, TextTo3DProvider)
        assert provider.modality == "text_to_3d"

def test_all_image3d_providers_instantiate():
    for cls in (MeshyImage3DProvider, TripoImage3DProvider, StabilityImage3DProvider,
                RodinImage3DProvider, CSMImage3DProvider):
        provider = cls()
        assert isinstance(provider, ImageTo3DProvider)
        assert provider.modality == "image_to_3d"

# 2. Availability checks (all False without credentials)
def test_cloud_providers_unavailable_without_keys(monkeypatch):
    for var in ("OPENAI_API_KEY", "DEEPGRAM_API_KEY", "ASSEMBLYAI_API_KEY",
                "GOOGLE_APPLICATION_CREDENTIALS", "ELEVENLABS_API_KEY",
                "MESHY_API_KEY", "TRIPO_API_KEY", "STABILITY_API_KEY",
                "HYPER3D_API_KEY", "CSM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert not OpenAIWhisperProvider().is_available()
    assert not DeepgramSTTProvider().is_available()
    assert not ElevenLabsTTSProvider().is_available()
    assert not MeshyText3DProvider().is_available()
    assert not StabilityImage3DProvider().is_available()

# 3. Factory functions
def test_create_stt_provider_by_name():
    provider = create_stt_provider("whisper_local")
    assert isinstance(provider, WhisperLocalProvider)

def test_create_text3d_provider_unknown_raises():
    with pytest.raises(ValueError, match="Unknown text-to-3D"):
        create_text3d_provider("nonexistent")

# 4. Poll helper
def test_poll_until_complete_succeeds():
    calls = [0]
    def check():
        calls[0] += 1
        if calls[0] >= 3:
            return {"status": "SUCCEEDED", "output": {"glb": "http://example.com/model.glb"}}
        return {"status": "PENDING"}
    result = _poll_until_complete(check, timeout=30.0, interval=0.01)
    assert result["status"] == "SUCCEEDED"

def test_poll_until_complete_raises_on_failure():
    def check():
        return {"status": "FAILED", "error": "bad input"}
    with pytest.raises(RuntimeError, match="failed"):
        _poll_until_complete(check, timeout=5.0, interval=0.01)

# 5. Vibe server availability (mock)
def test_vibe_server_unavailable_when_not_running():
    provider = VibeServerProvider(base_url="http://localhost:39999")
    assert not provider.is_available()

# 6. WhisperLocal availability (mock)
def test_whisper_local_available_with_package(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: object() if name == "faster_whisper" else None)
    assert WhisperLocalProvider().is_available()
```

---

## File Changes Summary

| File | Action |
|------|--------|
| `knowledge3d/tools/modal_providers.py` | **NEW** — 4 ABCs, 17 concrete providers, 4 factory functions, polling helper |
| `tests/test_modal_providers.py` | **NEW** — Instantiation, availability, factory, poll helper tests |

---

## Success Criteria

1. All 17 providers instantiate without error
2. All factory functions work by name and raise `ValueError` for unknown names
3. `is_available()` returns `False` when keys/packages missing (no crashes)
4. `_poll_until_complete()` handles success, failure, and timeout
5. `WhisperLocalProvider` uses `faster-whisper` (native Python, GPU)
6. `VibeServerProvider` connects to local Vibe `--server` via OpenAI SDK
7. 3D REST providers use `urllib.request` only (no `requests` dependency)
8. `StabilityImage3DProvider` is synchronous (no polling)
9. All existing H16/H16b tests still pass (non-regression)
10. `compileall` passes on new file

---

## The Bigger Picture

This completes K3D's **multi-modal I/O layer**:

```
EXTERNAL WORLD                          K3D INTERNAL
─────────────────                       ────────────────
Audio files ──[STT]──→ Text ──[Augment]──→ Stars ──→ Galaxy
Images ──────[Image→3D]──→ GLB ──→ House furniture/objects
Text prompts ──[Text→3D]──→ GLB ──→ House furniture/objects
                                        ↓
Stars ──[TTS]──→ Audio files ←───── EXTERNAL WORLD
Stars ──[DOM]──→ HTML elements ←─── EXTERNAL WORLD  (H11)
Stars ──[GLB]──→ 3D files ←──────── EXTERNAL WORLD  (H5)
```

Every external format can flow IN (ingestion) and OUT (production). K3D's Galaxy is the universal hub. The House is the spatial manifestation. The providers are the adapters.
