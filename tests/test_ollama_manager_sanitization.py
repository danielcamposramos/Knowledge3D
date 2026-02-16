from __future__ import annotations

from knowledge3d.ingestion.ollama_manager import OllamaModelManager


def test_sanitize_prompt_removes_null_and_control_bytes() -> None:
    manager = OllamaModelManager(default_timeout=1.0)
    raw = "alpha\x00beta\x01gamma\n\tzeta"
    clean = manager._sanitize_prompt(raw)  # type: ignore[attr-defined]

    assert "\x00" not in clean
    assert "\x01" not in clean
    assert "alpha" in clean
    assert "beta" in clean
    assert "gamma" in clean
    assert "\n" in clean
    assert "\t" in clean or " zeta" in clean


def test_query_uses_sanitized_prompt(monkeypatch) -> None:
    seen: dict[str, str] = {}

    class _Proc:
        def __init__(self) -> None:
            self.stdout = "{}"
            self.stderr = ""
            self.returncode = 0

    def _fake_run(cmd, check, capture_output, text, timeout):  # noqa: ANN001
        seen["prompt"] = cmd[3]
        return _Proc()

    monkeypatch.setattr("subprocess.run", _fake_run)
    manager = OllamaModelManager(default_timeout=1.0)
    result = manager.query(model="qwen2.5:14b", prompt="a\x00b\x02c", timeout=1.0)

    assert result.returncode == 0
    assert "prompt" in seen
    assert "\x00" not in seen["prompt"]
    assert "\x02" not in seen["prompt"]
