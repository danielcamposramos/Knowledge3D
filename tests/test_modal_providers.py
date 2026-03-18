from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from knowledge3d.tools.modal_providers import (
    AssemblyAISTTProvider,
    CSMImage3DProvider,
    CSMText3DProvider,
    DeepgramSTTProvider,
    ElevenLabsTTSProvider,
    GoogleSTTProvider,
    ImageTo3DProvider,
    MeshyImage3DProvider,
    MeshyText3DProvider,
    OpenAITTSProvider,
    OpenAIWhisperProvider,
    RodinImage3DProvider,
    RodinText3DProvider,
    SpeechToTextProvider,
    StabilityImage3DProvider,
    TextTo3DProvider,
    TextToSpeechProvider,
    TripoImage3DProvider,
    TripoText3DProvider,
    VibeServerProvider,
    WhisperLocalProvider,
    _poll_until_complete,
    create_image3d_provider,
    create_stt_provider,
    create_text3d_provider,
    create_tts_provider,
)


def test_all_stt_providers_instantiate() -> None:
    for cls in (
        WhisperLocalProvider,
        VibeServerProvider,
        OpenAIWhisperProvider,
        DeepgramSTTProvider,
        AssemblyAISTTProvider,
        GoogleSTTProvider,
    ):
        provider = cls()
        assert isinstance(provider, SpeechToTextProvider)
        assert provider.modality == "stt"


def test_all_tts_providers_instantiate() -> None:
    for cls in (OpenAITTSProvider, ElevenLabsTTSProvider):
        provider = cls()
        assert isinstance(provider, TextToSpeechProvider)
        assert provider.modality == "tts"


def test_all_text3d_providers_instantiate() -> None:
    for cls in (MeshyText3DProvider, TripoText3DProvider, RodinText3DProvider, CSMText3DProvider):
        provider = cls()
        assert isinstance(provider, TextTo3DProvider)
        assert provider.modality == "text_to_3d"


def test_all_image3d_providers_instantiate() -> None:
    for cls in (
        MeshyImage3DProvider,
        TripoImage3DProvider,
        StabilityImage3DProvider,
        RodinImage3DProvider,
        CSMImage3DProvider,
    ):
        provider = cls()
        assert isinstance(provider, ImageTo3DProvider)
        assert provider.modality == "image_to_3d"


def test_cloud_providers_unavailable_without_keys(monkeypatch) -> None:
    for var in (
        "OPENAI_API_KEY",
        "DEEPGRAM_API_KEY",
        "ASSEMBLYAI_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "ELEVENLABS_API_KEY",
        "MESHY_API_KEY",
        "TRIPO_API_KEY",
        "STABILITY_API_KEY",
        "HYPER3D_API_KEY",
        "CSM_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    assert not OpenAIWhisperProvider().is_available()
    assert not DeepgramSTTProvider().is_available()
    assert not ElevenLabsTTSProvider().is_available()
    assert not MeshyText3DProvider().is_available()
    assert not StabilityImage3DProvider().is_available()


def test_create_stt_provider_by_name() -> None:
    provider = create_stt_provider("whisper_local")
    assert isinstance(provider, WhisperLocalProvider)


def test_create_text3d_provider_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown text-to-3D"):
        create_text3d_provider("nonexistent")


def test_create_tts_provider_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown TTS provider"):
        create_tts_provider("nonexistent")


def test_create_image3d_provider_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown image-to-3D provider"):
        create_image3d_provider("nonexistent")


def test_poll_until_complete_succeeds() -> None:
    calls = [0]

    def check():
        calls[0] += 1
        if calls[0] >= 3:
            return {"status": "SUCCEEDED", "output": {"glb": "http://example.com/model.glb"}}
        return {"status": "PENDING"}

    result = _poll_until_complete(check, timeout=30.0, interval=0.01)

    assert result["status"] == "SUCCEEDED"


def test_poll_until_complete_raises_on_failure() -> None:
    def check():
        return {"status": "FAILED", "error": "bad input"}

    with pytest.raises(RuntimeError, match="failed"):
        _poll_until_complete(check, timeout=5.0, interval=0.01)


def test_poll_until_complete_times_out() -> None:
    def check():
        return {"status": "PENDING"}

    with pytest.raises(TimeoutError, match="timed out"):
        _poll_until_complete(check, timeout=0.02, interval=0.01)


def test_vibe_server_unavailable_when_not_running() -> None:
    provider = VibeServerProvider(base_url="http://localhost:39999")
    assert not provider.is_available()


def test_whisper_local_available_with_package(monkeypatch) -> None:
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "faster_whisper":
            return object()
        return original_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert WhisperLocalProvider().is_available()


def test_create_stt_provider_auto_prefers_local_first(monkeypatch) -> None:
    monkeypatch.setattr(WhisperLocalProvider, "is_available", lambda self: False)
    monkeypatch.setattr(VibeServerProvider, "is_available", lambda self: True)
    monkeypatch.setattr(OpenAIWhisperProvider, "is_available", lambda self: True)

    provider = create_stt_provider()

    assert isinstance(provider, VibeServerProvider)


def test_google_stt_requires_credentials_file(monkeypatch, tmp_path: Path) -> None:
    creds = tmp_path / "creds.json"
    creds.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(creds))
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "google.cloud.speech_v2":
            return object()
        return original_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert GoogleSTTProvider().is_available()
