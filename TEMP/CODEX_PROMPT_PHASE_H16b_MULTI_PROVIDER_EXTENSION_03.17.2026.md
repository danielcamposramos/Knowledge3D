# Phase H16b — Multi-Provider Augmentation Extension

**Depends on:** H16 (Living Ingestion & Augmentation)
**Modifies:** `knowledge3d/tools/augmentation_providers.py`
**Tests:** `tests/test_augmentation_providers.py` (extend existing)

---

## Objective

Extend `augmentation_providers.py` to support all major AI provider APIs: DeepSeek, Grok (xAI), Kimi (Moonshot), Qwen (DashScope), GLM (Zhipu AI), and Google Gemini (including vision/multimodal). Most of these are **OpenAI-compatible** — a single base class handles them.

---

## Architecture: OpenAI-Compatible Base Class

### Key Insight

DeepSeek, Grok/xAI, Kimi/Moonshot, and Qwen/DashScope all expose OpenAI-compatible chat completion endpoints. They accept the same `openai` Python SDK with a different `base_url` and API key. Rather than duplicating `GPTProvider` six times, introduce one reusable base:

```python
class OpenAICompatibleProvider(AugmentationProvider):
    """Base for any provider exposing an OpenAI-compatible chat API."""

    provider_name = "openai_compatible"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str,
        timeout: float = 120.0,
        **_: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout = float(timeout)
        self.api_key = os.environ.get(api_key_env, "").strip()
        self._api_key_env = api_key_env

    def augment(self, content: str, context: dict[str, Any]) -> AugmentationResult:
        if not self.is_available():
            raise RuntimeError(f"{self.provider_name} provider is not available.")
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": AUGMENTATION_SYSTEM_PROMPT},
                {"role": "user", "content": self._build_prompt(content, context)},
            ],
            max_tokens=2048,
        )
        text = ""
        if response.choices:
            text = str(response.choices[0].message.content or "")
        return self._parse_result(text, self.provider_name, context=context)

    def classify(self, content: str) -> str:
        if not self.is_available():
            return "General"
        try:
            from openai import OpenAI
        except ImportError:
            return "General"
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": self._build_classification_prompt(content)}],
            max_tokens=32,
        )
        text = ""
        if response.choices:
            text = str(response.choices[0].message.content or "")
        return self._normalize_domain(text)

    def is_available(self) -> bool:
        return bool(self.api_key) and importlib.util.find_spec("openai") is not None
```

**Important:** Wrap the `from openai import OpenAI` in a try/except inside `augment()` just like `GPTProvider` does, so the import error becomes a RuntimeError with a helpful message.

---

## Provider Definitions (All Subclass OpenAICompatibleProvider)

### 1. DeepSeekProvider

```python
class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek API — OpenAI-compatible."""
    provider_name = "deepseek"

    def __init__(self, model: str = "deepseek-chat", timeout: float = 120.0, **kw: Any) -> None:
        super().__init__(
            model=model,
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            timeout=timeout,
            **kw,
        )
```

### 2. GrokProvider

```python
class GrokProvider(OpenAICompatibleProvider):
    """xAI Grok API — OpenAI-compatible."""
    provider_name = "grok"

    def __init__(self, model: str = "grok-3-latest", timeout: float = 120.0, **kw: Any) -> None:
        super().__init__(
            model=model,
            base_url="https://api.x.ai/v1",
            api_key_env="XAI_API_KEY",
            timeout=timeout,
            **kw,
        )
```

### 3. KimiProvider

```python
class KimiProvider(OpenAICompatibleProvider):
    """Moonshot AI Kimi API — OpenAI-compatible."""
    provider_name = "kimi"

    def __init__(self, model: str = "moonshot-v1-auto", timeout: float = 120.0, **kw: Any) -> None:
        super().__init__(
            model=model,
            base_url="https://api.moonshot.cn/v1",
            api_key_env="MOONSHOT_API_KEY",
            timeout=timeout,
            **kw,
        )
```

### 4. QwenProvider

```python
class QwenProvider(OpenAICompatibleProvider):
    """Alibaba Qwen via DashScope — OpenAI-compatible mode."""
    provider_name = "qwen"

    def __init__(self, model: str = "qwen-max", timeout: float = 120.0, **kw: Any) -> None:
        super().__init__(
            model=model,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env="DASHSCOPE_API_KEY",
            timeout=timeout,
            **kw,
        )
```

### 5. GLMProvider

```python
class GLMProvider(OpenAICompatibleProvider):
    """Zhipu AI GLM API — OpenAI-compatible mode."""
    provider_name = "glm"

    def __init__(self, model: str = "glm-4-plus", timeout: float = 120.0, **kw: Any) -> None:
        super().__init__(
            model=model,
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key_env="ZHIPUAI_API_KEY",
            timeout=timeout,
            **kw,
        )
```

---

## 6. GeminiProvider (Separate — Google's Own SDK)

Google Gemini uses `google-generativeai` SDK (not OpenAI-compatible). This provider also supports **vision/multimodal** via the same SDK.

```python
class GeminiProvider(AugmentationProvider):
    """Google Gemini API with vision/multimodal support."""

    provider_name = "gemini"

    def __init__(self, model: str = "gemini-2.0-flash", timeout: float = 120.0, **_: Any) -> None:
        self.model = model
        self.timeout = float(timeout)
        self.api_key = os.environ.get("GOOGLE_API_KEY", "").strip()

    def augment(self, content: str, context: dict[str, Any]) -> AugmentationResult:
        if not self.is_available():
            raise RuntimeError("Gemini provider is not available.")
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError("google-generativeai package is not installed.") from exc
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=AUGMENTATION_SYSTEM_PROMPT,
        )
        prompt = self._build_prompt(content, context)

        # Vision path: if context contains an image_path, include it
        parts: list[Any] = [prompt]
        image_path = (context or {}).get("image_path")
        if image_path and Path(image_path).is_file():
            import PIL.Image
            img = PIL.Image.open(image_path)
            parts.append(img)

        response = model.generate_content(
            parts,
            generation_config=genai.types.GenerationConfig(max_output_tokens=2048),
        )
        text = response.text if hasattr(response, "text") else ""
        return self._parse_result(text, self.provider_name, context=context)

    def classify(self, content: str) -> str:
        if not self.is_available():
            return "General"
        try:
            import google.generativeai as genai
        except ImportError:
            return "General"
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(model_name=self.model)
        response = model.generate_content(
            self._build_classification_prompt(content),
            generation_config=genai.types.GenerationConfig(max_output_tokens=32),
        )
        text = response.text if hasattr(response, "text") else ""
        return self._normalize_domain(text)

    def is_available(self) -> bool:
        return bool(self.api_key) and importlib.util.find_spec("google.generativeai") is not None
```

### Vision/Multimodal Notes

- Gemini natively accepts `PIL.Image` objects alongside text in `generate_content()`
- Pass `image_path` in the `context` dict to trigger vision augmentation
- `PIL` (Pillow) is a lazy import — only loaded when an image is present
- Other providers are text-only for now; vision can be added to OpenAI-compatible providers later (GPT-4o, Qwen-VL support it via base64 image messages)

---

## Updated create_provider() Factory

```python
def create_provider(name: str | None = None, **kwargs: Any) -> AugmentationProvider:
    """Create an explicit provider or auto-select the best available one."""
    providers: dict[str, type[AugmentationProvider]] = {
        "ollama": OllamaProvider,
        "claude": ClaudeProvider,
        "gpt": GPTProvider,
        "deepseek": DeepSeekProvider,
        "grok": GrokProvider,
        "kimi": KimiProvider,
        "qwen": QwenProvider,
        "glm": GLMProvider,
        "gemini": GeminiProvider,
    }
    requested = str(name or "auto").strip().lower()
    if requested and requested != "auto":
        cls = providers.get(requested)
        if cls is None:
            raise ValueError(
                f"Unknown provider: {name}. Available: {sorted(providers.keys())}"
            )
        return cls(**kwargs)
    # Auto-detection priority: local first, then cloud providers
    for candidate_name in ("ollama", "claude", "gpt", "deepseek", "gemini", "qwen", "glm", "grok", "kimi"):
        provider = providers[candidate_name](**kwargs)
        if provider.is_available():
            return provider
    return OllamaProvider(**kwargs)
```

**Auto-detection order rationale:**
1. `ollama` — local, free, no API cost
2. `claude` — Anthropic (project's primary AI partner)
3. `gpt` — OpenAI (widely available)
4. `deepseek` — strong reasoning, cost-effective
5. `gemini` — Google (vision support)
6. `qwen` — Alibaba (strong multilingual)
7. `glm` — Zhipu (Chinese market leader)
8. `grok` — xAI
9. `kimi` — Moonshot

---

## Refactor: GPTProvider → OpenAICompatibleProvider

The existing `GPTProvider` should be refactored to subclass `OpenAICompatibleProvider`:

```python
class GPTProvider(OpenAICompatibleProvider):
    """OpenAI GPT-backed augmentation provider."""
    provider_name = "gpt"

    def __init__(self, model: str = "gpt-4o", timeout: float = 120.0, **kw: Any) -> None:
        super().__init__(
            model=model,
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            timeout=timeout,
            **kw,
        )
```

This removes ~40 lines of duplicated code from the current GPTProvider while preserving identical behavior. The `base_url` for OpenAI is `https://api.openai.com/v1` (the openai SDK default, but explicit here for consistency).

---

## Environment Variables Summary

| Provider | Env Var | SDK Required |
|----------|---------|-------------|
| Ollama | (none — local) | (none — subprocess) |
| Claude | `ANTHROPIC_API_KEY` | `anthropic` |
| GPT | `OPENAI_API_KEY` | `openai` |
| DeepSeek | `DEEPSEEK_API_KEY` | `openai` |
| Grok | `XAI_API_KEY` | `openai` |
| Kimi | `MOONSHOT_API_KEY` | `openai` |
| Qwen | `DASHSCOPE_API_KEY` | `openai` |
| GLM | `ZHIPUAI_API_KEY` | `openai` |
| Gemini | `GOOGLE_API_KEY` | `google-generativeai` (+ `Pillow` for vision) |

All API key access is via standard `os.environ.get()` — user sets their own keys. No secrets stored in code.

---

## Updated __all__

```python
__all__ = [
    "ALLOWED_DOMAINS",
    "AUGMENTATION_SYSTEM_PROMPT",
    "AugmentationProvider",
    "AugmentationResult",
    "ClaudeProvider",
    "DeepSeekProvider",
    "GPTProvider",
    "GeminiProvider",
    "GLMProvider",
    "GrokProvider",
    "KimiProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "QwenProvider",
    "create_provider",
]
```

---

## Tests

Extend `tests/test_augmentation_providers.py` with:

### 1. OpenAICompatibleProvider base class tests
```python
def test_openai_compatible_base_not_available_without_key():
    """All OpenAI-compatible providers return is_available()=False without env var."""
    for cls in (DeepSeekProvider, GrokProvider, KimiProvider, QwenProvider, GLMProvider):
        provider = cls()
        assert not provider.is_available()
        assert provider.provider_name in ("deepseek", "grok", "kimi", "qwen", "glm")
```

### 2. GeminiProvider tests
```python
def test_gemini_not_available_without_key():
    provider = GeminiProvider()
    assert not provider.is_available()
    assert provider.provider_name == "gemini"

def test_gemini_classify_fallback():
    provider = GeminiProvider()
    assert provider.classify("some content") == "General"
```

### 3. Factory tests
```python
def test_create_provider_all_names():
    """All provider names are recognized by create_provider()."""
    for name in ("ollama", "claude", "gpt", "deepseek", "grok", "kimi", "qwen", "glm", "gemini"):
        provider = create_provider(name)
        assert provider.provider_name == name

def test_create_provider_unknown_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        create_provider("nonexistent")
```

### 4. GPTProvider refactor non-regression
```python
def test_gpt_provider_is_openai_compatible():
    """GPTProvider is now a subclass of OpenAICompatibleProvider."""
    provider = GPTProvider()
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.provider_name == "gpt"
```

### 5. Provider configuration tests
```python
def test_deepseek_base_url():
    p = DeepSeekProvider()
    assert p.base_url == "https://api.deepseek.com"
    assert p.model == "deepseek-chat"

def test_grok_base_url():
    p = GrokProvider()
    assert p.base_url == "https://api.x.ai/v1"
    assert p.model == "grok-3-latest"

def test_kimi_base_url():
    p = KimiProvider()
    assert p.base_url == "https://api.moonshot.cn/v1"
    assert p.model == "moonshot-v1-auto"

def test_qwen_base_url():
    p = QwenProvider()
    assert p.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert p.model == "qwen-max"

def test_glm_base_url():
    p = GLMProvider()
    assert p.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert p.model == "glm-4-plus"
```

---

## File Changes Summary

| File | Action |
|------|--------|
| `knowledge3d/tools/augmentation_providers.py` | Add `OpenAICompatibleProvider` base, 6 new provider subclasses, refactor `GPTProvider`, update `create_provider()` and `__all__` |
| `tests/test_augmentation_providers.py` | Add tests for all new providers (availability, config, factory) |

---

## Success Criteria

1. All 9 providers instantiate without error
2. `create_provider("deepseek")` etc. returns correct provider type
3. `is_available()` returns `False` when env var is missing (no crashes)
4. `classify()` returns `"General"` when provider unavailable (graceful fallback)
5. `GPTProvider` refactored to subclass `OpenAICompatibleProvider` — existing tests still pass
6. `GeminiProvider` accepts `image_path` in context for vision augmentation
7. All existing H16 tests pass (non-regression)
8. New tests cover every provider's config and availability check
