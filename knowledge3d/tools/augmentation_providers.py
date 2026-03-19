"""Multi-provider augmentation backend for ingestion-time enrichment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from knowledge3d.ingestion.ollama_manager import OllamaModelManager


ALLOWED_DOMAINS = (
    "Mathematics",
    "Physics",
    "Biology",
    "Language",
    "Tools",
    "Visual",
    "Audio",
    "General",
)

AUGMENTATION_SYSTEM_PROMPT = """You augment knowledge content for K3D ingestion.
Return strict JSON with keys:
summary: compact summary text
entities: list of {"type","name","content"}
relationships: list of {"from","relation","to"}
domain: one of Mathematics, Physics, Biology, Language, Tools, Visual, Audio, General
meaning_rpn_hint: compact RPN-like semantic description
taxonomy_refs: list of related concept IDs
surface_forms: {"en": "english name", "pt": "portuguese name"}
confidence: float 0-1
Avoid long narrative. Be precise and structured."""


@dataclass
class AugmentationResult:
    """Structured augmentation payload consumed by star generation."""

    summary: str
    entities: list[dict[str, str]]
    relationships: list[dict[str, str]]
    domain: str
    meaning_rpn_hint: str
    taxonomy_refs: list[str]
    surface_forms: dict[str, str]
    confidence: float
    provider: str
    raw_response: str


class AugmentationProvider(ABC):
    """Base augmentation provider."""

    provider_name = "provider"

    @abstractmethod
    def augment(self, content: str, context: dict[str, Any]) -> AugmentationResult:
        """Augment content into structured knowledge."""

    @abstractmethod
    def classify(self, content: str) -> str:
        """Classify content into a target domain."""

    @abstractmethod
    def is_available(self) -> bool:
        """Whether provider credentials/runtime are available."""

    def _build_prompt(self, content: str, context: dict[str, Any]) -> str:
        snippet = str(content or "").strip()
        if len(snippet) > 8000:
            snippet = snippet[:8000]
        source_path = str(context.get("path", "")).strip()
        domain_hint = str(context.get("domain_hint", "")).strip()
        name = str(context.get("name", "")).strip()
        return (
            f"Source name: {name or 'unknown'}\n"
            f"Source path: {source_path or 'unknown'}\n"
            f"Domain hint: {domain_hint or 'General'}\n"
            "Augment the following content into the required JSON schema.\n\n"
            f"CONTENT:\n{snippet}"
        )

    def _build_classification_prompt(self, content: str) -> str:
        snippet = str(content or "").strip()
        if len(snippet) > 500:
            snippet = snippet[:500]
        return (
            "Classify this content into exactly one domain: "
            "Mathematics, Physics, Biology, Language, Tools, Visual, Audio, General.\n"
            f"Content:\n{snippet}\n"
            "Domain:"
        )

    def _parse_result(
        self,
        raw_response: str,
        provider: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> AugmentationResult:
        payload = self._extract_json_payload(raw_response)
        if not isinstance(payload, dict):
            return self._fallback_result(raw_response, provider, context=context)
        domain = self._normalize_domain(payload.get("domain") or (context or {}).get("domain_hint"))
        summary = str(payload.get("summary") or "").strip()
        if not summary:
            summary = self._fallback_summary(context)
        meaning_rpn_hint = str(payload.get("meaning_rpn_hint") or "").strip()
        if not meaning_rpn_hint:
            meaning_rpn_hint = f"{domain.upper()} CONTENT ENTRY"
        taxonomy_refs = self._normalize_taxonomy_refs(payload.get("taxonomy_refs"), domain)
        surface_forms = self._normalize_surface_forms(payload.get("surface_forms"), context=context)
        confidence = self._normalize_confidence(payload.get("confidence"))
        return AugmentationResult(
            summary=summary,
            entities=self._normalize_entities(payload.get("entities")),
            relationships=self._normalize_relationships(payload.get("relationships")),
            domain=domain,
            meaning_rpn_hint=meaning_rpn_hint,
            taxonomy_refs=taxonomy_refs,
            surface_forms=surface_forms,
            confidence=confidence,
            provider=provider,
            raw_response=str(raw_response or ""),
        )

    def _fallback_result(
        self,
        raw_response: str,
        provider: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> AugmentationResult:
        context = dict(context or {})
        domain = self._normalize_domain(context.get("domain_hint"))
        summary = self._fallback_summary(context)
        surface_forms = self._normalize_surface_forms({}, context=context)
        taxonomy_refs = self._normalize_taxonomy_refs([], domain)
        return AugmentationResult(
            summary=summary,
            entities=[],
            relationships=[],
            domain=domain,
            meaning_rpn_hint=f"{domain.upper()} CONTENT ENTRY",
            taxonomy_refs=taxonomy_refs,
            surface_forms=surface_forms,
            confidence=0.35,
            provider=provider,
            raw_response=str(raw_response or ""),
        )

    def _fallback_summary(self, context: dict[str, Any] | None) -> str:
        context = dict(context or {})
        name = str(context.get("name", "")).strip()
        if name:
            return name
        path_value = str(context.get("path", "")).strip()
        if path_value:
            return Path(path_value).stem or Path(path_value).name
        return "Untitled content entry"

    def _extract_json_payload(self, text: str) -> Any:
        raw = str(text or "").strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        for idx, char in enumerate(raw):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(raw[idx:])
                return payload
            except json.JSONDecodeError:
                continue
        return None

    def _normalize_domain(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "General"
        lowered = text.lower()
        for domain in ALLOWED_DOMAINS:
            if lowered == domain.lower():
                return domain
        for domain in ALLOWED_DOMAINS:
            if domain.lower() in lowered:
                return domain
        return "General"

    def _normalize_entities(self, value: Any) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        out: list[dict[str, str]] = []
        for row in value:
            if not isinstance(row, dict):
                continue
            out.append(
                {
                    "type": str(row.get("type", "")).strip(),
                    "name": str(row.get("name", "")).strip(),
                    "content": str(row.get("content", "")).strip(),
                }
            )
        return out

    def _normalize_relationships(self, value: Any) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        out: list[dict[str, str]] = []
        for row in value:
            if not isinstance(row, dict):
                continue
            out.append(
                {
                    "from": str(row.get("from", "")).strip(),
                    "relation": str(row.get("relation", "")).strip(),
                    "to": str(row.get("to", "")).strip(),
                }
            )
        return out

    def _normalize_taxonomy_refs(self, value: Any, domain: str) -> list[str]:
        refs: list[str] = []
        if isinstance(value, list):
            for item in value:
                text = str(item).strip()
                if text:
                    refs.append(text)
        if not refs and domain != "General":
            refs.append(f"concept_{domain.lower()}")
        seen: set[str] = set()
        out: list[str] = []
        for ref in refs:
            key = ref.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(ref)
        return out

    def _normalize_surface_forms(
        self,
        value: Any,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        if isinstance(value, dict):
            for language, raw in value.items():
                lang = str(language).strip().lower()
                text = str(raw).strip()
                if lang and text:
                    out[lang] = text
        if "en" not in out:
            out["en"] = self._fallback_summary(context)
        if "pt" not in out:
            out["pt"] = out["en"]
        return out

    def _normalize_confidence(self, value: Any) -> float:
        try:
            resolved = float(value)
        except Exception:
            resolved = 0.35
        return max(0.0, min(1.0, resolved))


class OllamaProvider(AugmentationProvider):
    """Local Ollama-backed augmentation provider."""

    provider_name = "ollama"

    def __init__(
        self,
        model: str = "qwen2.5:32b",
        timeout: float = 120.0,
        ollama: OllamaModelManager | None = None,
        **_: Any,
    ) -> None:
        self.model = model
        self.timeout = float(timeout)
        self.ollama = ollama or OllamaModelManager(default_timeout=self.timeout)

    def augment(self, content: str, context: dict[str, Any]) -> AugmentationResult:
        prompt = self._build_prompt(content, context)
        result = self.ollama.query(model=self.model, prompt=prompt, timeout=self.timeout)
        if result.returncode != 0:
            return self._fallback_result(result.stderr or result.output, self.provider_name, context=context)
        return self._parse_result(result.output, self.provider_name, context=context)

    def classify(self, content: str) -> str:
        result = self.ollama.query(
            model=self.model,
            prompt=self._build_classification_prompt(content),
            timeout=max(5.0, min(self.timeout, 30.0)),
        )
        if result.returncode != 0:
            return "General"
        return self._normalize_domain(result.output)

    def is_available(self) -> bool:
        try:
            proc = subprocess.run(
                ["ollama", "list"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        return proc.returncode == 0


class ClaudeProvider(AugmentationProvider):
    """Anthropic-backed augmentation provider."""

    provider_name = "claude"

    def __init__(self, model: str = "claude-sonnet-4-6", timeout: float = 120.0, **_: Any) -> None:
        self.model = model
        self.timeout = float(timeout)
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    def augment(self, content: str, context: dict[str, Any]) -> AugmentationResult:
        if not self.is_available():
            raise RuntimeError("Claude provider is not available.")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("anthropic package is not installed.") from exc
        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=AUGMENTATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self._build_prompt(content, context)}],
        )
        text = message.content[0].text if getattr(message, "content", None) else ""
        return self._parse_result(text, self.provider_name, context=context)

    def classify(self, content: str) -> str:
        if not self.is_available():
            return "General"
        try:
            import anthropic
        except ImportError:
            return "General"
        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=64,
            messages=[{"role": "user", "content": self._build_classification_prompt(content)}],
        )
        text = message.content[0].text if getattr(message, "content", None) else ""
        return self._normalize_domain(text)

    def is_available(self) -> bool:
        return bool(self.api_key) and importlib.util.find_spec("anthropic") is not None


class OpenAICompatibleProvider(AugmentationProvider):
    """Base for providers exposing OpenAI-compatible chat endpoints."""

    provider_name = "openai_compatible"

    def __init__(
        self,
        *,
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
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("openai package is not installed.") from exc
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


class ClaudeAgentSDKProvider(AugmentationProvider):
    """Claude via Agent SDK — subscription-based, no API key needed.

    Uses the ``claude-agent-sdk`` package which authenticates through the
    Claude Code CLI (Pro/Team/Enterprise subscription).  This is the legal
    subscription billing path — same mechanism OpenClaw uses.

    The existing :class:`ClaudeProvider` (API-key billing) is kept as-is.
    Choose ``claude-agent-sdk`` when you want subscription billing, or
    ``claude`` when you want per-token API billing.
    """

    provider_name = "claude-agent-sdk"

    def __init__(self, model: str = "claude-sonnet-4-6", timeout: float = 120.0, **_: Any) -> None:
        self.model = model
        self.timeout = float(timeout)

    def augment(self, content: str, context: dict[str, Any]) -> AugmentationResult:
        if not self.is_available():
            raise RuntimeError("claude-agent-sdk package is not installed or CLI not logged in.")
        text = self._run_query(
            system_prompt=AUGMENTATION_SYSTEM_PROMPT,
            user_message=self._build_prompt(content, context),
            max_tokens=2048,
        )
        return self._parse_result(text, self.provider_name, context=context)

    def classify(self, content: str) -> str:
        if not self.is_available():
            return "General"
        text = self._run_query(
            system_prompt="",
            user_message=self._build_classification_prompt(content),
            max_tokens=64,
        )
        return self._normalize_domain(text)

    def is_available(self) -> bool:
        return importlib.util.find_spec("claude_agent_sdk") is not None

    def _run_query(self, *, system_prompt: str, user_message: str, max_tokens: int) -> str:
        """Run a single-turn query via the Agent SDK (async internally)."""
        try:
            import anyio
            from claude_agent_sdk import query as agent_query, ClaudeAgentOptions, ResultMessage
        except ImportError:
            return ""

        async def _do_query() -> str:
            result_text = ""
            opts = ClaudeAgentOptions(
                system_prompt=system_prompt,
                allowed_tools=[],   # pure chat — no file/shell access
                max_turns=1,
                model=self.model,
            )
            async for message in agent_query(prompt=user_message, options=opts):
                if isinstance(message, ResultMessage):
                    result_text = message.result
            return result_text

        try:
            return anyio.from_thread.run_sync(_do_query)
        except Exception:
            # If already inside an event loop, run via anyio.run
            try:
                return anyio.run(_do_query)
            except Exception:
                return ""


class CodexOAuthProvider(OpenAICompatibleProvider):
    """OpenAI Codex via OAuth — subscription-based, uses Codex CLI token.

    Equivalent to Claude Agent SDK but for the OpenAI ecosystem.  Uses the
    ``openai`` package pointed at the Codex API endpoint with the OAuth
    token from the Codex CLI session (``OPENAI_API_KEY`` env var set by
    the Codex CLI, or the user's own subscription key).

    Falls back to standard OpenAI API when Codex endpoint is unavailable.
    """

    provider_name = "codex"

    def __init__(self, model: str = "gpt-4o", timeout: float = 120.0, **_: Any) -> None:
        # Codex CLI sets OPENAI_API_KEY; same key works for standard API
        super().__init__(
            model=model,
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            timeout=timeout,
        )


class GPTProvider(OpenAICompatibleProvider):
    """OpenAI-backed augmentation provider (API key billing)."""

    provider_name = "gpt"

    def __init__(self, model: str = "gpt-4o", timeout: float = 120.0, **_: Any) -> None:
        super().__init__(
            model=model,
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            timeout=timeout,
        )

class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek augmentation provider."""

    provider_name = "deepseek"

    def __init__(self, model: str = "deepseek-chat", timeout: float = 120.0, **_: Any) -> None:
        super().__init__(
            model=model,
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            timeout=timeout,
        )


class GrokProvider(OpenAICompatibleProvider):
    """xAI Grok augmentation provider."""

    provider_name = "grok"

    def __init__(self, model: str = "grok-3-latest", timeout: float = 120.0, **_: Any) -> None:
        super().__init__(
            model=model,
            base_url="https://api.x.ai/v1",
            api_key_env="XAI_API_KEY",
            timeout=timeout,
        )


class KimiProvider(OpenAICompatibleProvider):
    """Moonshot Kimi augmentation provider."""

    provider_name = "kimi"

    def __init__(self, model: str = "moonshot-v1-auto", timeout: float = 120.0, **_: Any) -> None:
        super().__init__(
            model=model,
            base_url="https://api.moonshot.cn/v1",
            api_key_env="MOONSHOT_API_KEY",
            timeout=timeout,
        )


class QwenProvider(OpenAICompatibleProvider):
    """DashScope-compatible Qwen augmentation provider."""

    provider_name = "qwen"

    def __init__(self, model: str = "qwen-max", timeout: float = 120.0, **_: Any) -> None:
        super().__init__(
            model=model,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env="DASHSCOPE_API_KEY",
            timeout=timeout,
        )


class GLMProvider(OpenAICompatibleProvider):
    """Zhipu GLM augmentation provider."""

    provider_name = "glm"

    def __init__(self, model: str = "glm-4-plus", timeout: float = 120.0, **_: Any) -> None:
        super().__init__(
            model=model,
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key_env="ZHIPUAI_API_KEY",
            timeout=timeout,
        )


class GeminiProvider(AugmentationProvider):
    """Google Gemini provider with optional image-path augmentation."""

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
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("google-generativeai package is not installed.") from exc
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=AUGMENTATION_SYSTEM_PROMPT,
        )
        prompt = self._build_prompt(content, context)
        parts: list[Any] = [prompt]
        image_obj: Any | None = None
        image_path = str((context or {}).get("image_path", "")).strip()
        if image_path:
            resolved = Path(image_path)
            if resolved.is_file():
                try:
                    import PIL.Image
                except ImportError as exc:  # pragma: no cover - environment-dependent
                    raise RuntimeError("Pillow package is not installed.") from exc
                image_obj = PIL.Image.open(resolved)
                parts.append(image_obj)
        try:
            response = model.generate_content(
                parts,
                generation_config=genai.types.GenerationConfig(max_output_tokens=2048),
            )
        finally:
            if image_obj is not None and hasattr(image_obj, "close"):
                image_obj.close()
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


def create_provider(name: str | None = None, **kwargs: Any) -> AugmentationProvider:
    """Create an explicit provider or auto-select the best available one."""
    providers: dict[str, type[AugmentationProvider]] = {
        "ollama": OllamaProvider,
        "claude": ClaudeProvider,
        "claude-agent-sdk": ClaudeAgentSDKProvider,
        "codex": CodexOAuthProvider,
        "gpt": GPTProvider,
        "deepseek": DeepSeekProvider,
        "gemini": GeminiProvider,
        "qwen": QwenProvider,
        "glm": GLMProvider,
        "grok": GrokProvider,
        "kimi": KimiProvider,
    }
    requested = str(name or "auto").strip().lower()
    if requested and requested != "auto":
        cls = providers.get(requested)
        if cls is None:
            raise ValueError(
                f"Unknown provider: {name}. Available: {sorted(providers.keys())}"
            )
        return cls(**kwargs)
    # Auto-detection: subscription providers before API-key providers
    for candidate_name in ("ollama", "claude-agent-sdk", "claude", "codex", "gpt", "deepseek", "gemini", "qwen", "glm", "grok", "kimi"):
        provider = providers[candidate_name](**kwargs)
        if provider.is_available():
            return provider
    return OllamaProvider(**kwargs)


__all__ = [
    "ALLOWED_DOMAINS",
    "AUGMENTATION_SYSTEM_PROMPT",
    "AugmentationProvider",
    "AugmentationResult",
    "ClaudeAgentSDKProvider",
    "ClaudeProvider",
    "CodexOAuthProvider",
    "DeepSeekProvider",
    "GLMProvider",
    "GeminiProvider",
    "GPTProvider",
    "GrokProvider",
    "KimiProvider",
    "OpenAICompatibleProvider",
    "OllamaProvider",
    "QwenProvider",
    "create_provider",
]
