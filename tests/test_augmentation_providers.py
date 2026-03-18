from __future__ import annotations

import importlib.util
import sys
import subprocess
import types

from knowledge3d.tools.augmentation_providers import (
    AugmentationResult,
    ClaudeProvider,
    DeepSeekProvider,
    GeminiProvider,
    GPTProvider,
    GLMProvider,
    GrokProvider,
    KimiProvider,
    OpenAICompatibleProvider,
    OllamaProvider,
    QwenProvider,
    create_provider,
)


class _DummyResponse:
    def __init__(self, output: str, returncode: int = 0, stderr: str = "") -> None:
        self.output = output
        self.returncode = returncode
        self.stderr = stderr


class _DummyOllama:
    def __init__(self, output: str) -> None:
        self.output = output

    def query(self, *, model: str, prompt: str, timeout: float) -> _DummyResponse:
        return _DummyResponse(self.output)


def test_ollama_provider_available_check(monkeypatch) -> None:
    def _fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="NAME", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    provider = OllamaProvider(ollama=_DummyOllama("{}"))

    assert provider.is_available() is True


def test_ollama_provider_parses_json_block() -> None:
    provider = OllamaProvider(
        ollama=_DummyOllama(
            """
            leading text
            {"summary":"Linear algebra notes","entities":[{"type":"concept","name":"matrix","content":"array"}],
             "relationships":[{"from":"matrix","relation":"supports","to":"determinant"}],
             "domain":"Mathematics","meaning_rpn_hint":"MATH MATRIX ENTRY",
             "taxonomy_refs":["concept_mathematics"],"surface_forms":{"en":"Matrix Notes","pt":"Notas de Matriz"},
             "confidence":0.9}
            trailing text
            """
        )
    )

    result = provider.augment("content", {"name": "matrix_notes.txt", "domain_hint": "Mathematics"})

    assert isinstance(result, AugmentationResult)
    assert result.domain == "Mathematics"
    assert result.surface_forms["en"] == "Matrix Notes"


def test_claude_provider_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = ClaudeProvider()
    assert not provider.is_available()


def test_gpt_provider_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = GPTProvider()
    assert not provider.is_available()


def test_openai_compatible_providers_require_their_api_keys(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    assert not DeepSeekProvider().is_available()
    assert not GrokProvider().is_available()
    assert not KimiProvider().is_available()
    assert not QwenProvider().is_available()


def test_gpt_provider_subclasses_openai_compatible() -> None:
    provider = GPTProvider()
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.base_url == "https://api.openai.com/v1"


def test_provider_base_urls_are_wired() -> None:
    assert DeepSeekProvider().base_url == "https://api.deepseek.com"
    assert GrokProvider().base_url == "https://api.x.ai/v1"
    assert KimiProvider().base_url == "https://api.moonshot.cn/v1"
    assert QwenProvider().base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert GLMProvider().base_url == "https://open.bigmodel.cn/api/paas/v4"


def test_gemini_provider_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    provider = GeminiProvider()
    assert not provider.is_available()


def test_create_provider_auto_prefers_first_available(monkeypatch) -> None:
    monkeypatch.setattr(OllamaProvider, "is_available", lambda self: False)
    monkeypatch.setattr(ClaudeProvider, "is_available", lambda self: False)
    monkeypatch.setattr(GPTProvider, "is_available", lambda self: True)
    monkeypatch.setattr(DeepSeekProvider, "is_available", lambda self: True)

    provider = create_provider()

    assert isinstance(provider, GPTProvider)


def test_gemini_provider_uses_image_path_when_present(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake")
    monkeypatch.setenv("GOOGLE_API_KEY", "key")

    class _DummyResponse:
        text = '{"summary":"Image note","entities":[],"relationships":[],"domain":"Visual","meaning_rpn_hint":"VISUAL IMAGE ENTRY","taxonomy_refs":["concept_visual"],"surface_forms":{"en":"Image Note","pt":"Nota de Imagem"},"confidence":0.8}'

    captured: dict[str, object] = {}

    class _DummyModel:
        def __init__(self, *, model_name: str, system_instruction: str | None = None) -> None:
            captured["model_name"] = model_name
            captured["system_instruction"] = system_instruction

        def generate_content(self, parts, generation_config=None):
            captured["parts"] = list(parts)
            return _DummyResponse()

    genai_module = types.ModuleType("google.generativeai")
    genai_module.configure = lambda api_key: captured.setdefault("api_key", api_key)
    genai_module.GenerativeModel = _DummyModel
    genai_module.types = types.SimpleNamespace(
        GenerationConfig=lambda **kwargs: kwargs,
    )
    google_module = types.ModuleType("google")
    google_module.generativeai = genai_module

    class _DummyImage:
        def __init__(self, path):
            self.path = str(path)
            self.closed = False

        def close(self):
            self.closed = True

    dummy_image = _DummyImage(image_path)
    pil_image_module = types.ModuleType("PIL.Image")
    pil_image_module.open = lambda path: dummy_image
    pil_module = types.ModuleType("PIL")
    pil_module.Image = pil_image_module

    original_find_spec = importlib.util.find_spec

    def _fake_find_spec(name: str):
        if name in {"google.generativeai", "openai"}:
            return object()
        return original_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.generativeai", genai_module)
    monkeypatch.setitem(sys.modules, "PIL", pil_module)
    monkeypatch.setitem(sys.modules, "PIL.Image", pil_image_module)

    provider = GeminiProvider()
    result = provider.augment("content", {"name": "picture", "image_path": str(image_path), "domain_hint": "Visual"})

    assert result.domain == "Visual"
    assert len(captured["parts"]) == 2
    assert captured["parts"][1] is dummy_image
    assert dummy_image.closed is True


def test_create_provider_factory() -> None:
    provider = create_provider("ollama", ollama=_DummyOllama("{}"))
    assert isinstance(provider, OllamaProvider)


def test_create_provider_factory_for_new_provider() -> None:
    provider = create_provider("deepseek")
    assert isinstance(provider, DeepSeekProvider)
