from __future__ import annotations

from pathlib import Path

from knowledge3d.ingestion.enrichment_pipeline import EnrichmentPipeline
from knowledge3d.ingestion.k3d_transformer import K3DTransformer
from knowledge3d.ingestion.numbered_context import NumberedContextProvider
from knowledge3d.ingestion.ollama_manager import OllamaModelManager, OllamaQueryResult
from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager


def test_numbered_context_provider_supports_chunked_and_line_access():
    content = "\n".join(f"Line {idx}" for idx in range(1, 81))
    provider = NumberedContextProvider(content=content, chunk_size=128)

    initial = provider.get_initial_context(num_chunks=2)
    assert initial["total_chunks"] >= 2
    assert len(initial["provided_chunks"]) == 2
    assert "request_more" in initial["instructions"]

    snippet = provider.get_lines(line_start=10, line_end=12)
    assert "Line 10" in snippet
    assert "Line 12" in snippet


def test_ollama_model_manager_unloads_after_context(monkeypatch):
    calls: list[list[str]] = []

    class _Proc:
        def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    def _fake_run(cmd, check, capture_output, text, timeout):  # noqa: ANN001
        calls.append(list(cmd))
        if len(cmd) >= 3 and cmd[1] == "run":
            return _Proc(stdout='{"patterns":[]}')
        if len(cmd) >= 3 and cmd[1] == "stop":
            return _Proc(stdout="")
        return _Proc(stdout="")

    monkeypatch.setattr("subprocess.run", _fake_run)

    with OllamaModelManager(default_timeout=2.0) as manager:
        manager.load_model("qwen2.5:14b")
        result = manager.query(model="qwen2.5:14b", prompt="hello", timeout=2.0)
        assert isinstance(result, OllamaQueryResult)
        assert result.returncode == 0

    run_calls = [cmd for cmd in calls if len(cmd) >= 3 and cmd[1] == "run"]
    stop_calls = [cmd for cmd in calls if len(cmd) >= 3 and cmd[1] == "stop"]
    assert run_calls
    assert stop_calls
    assert stop_calls[-1][2] == "qwen2.5:14b"


def test_enrichment_pipeline_handles_structured_rag_request_more(monkeypatch):
    class _FakeManager:
        def __init__(self, default_timeout=120.0):  # noqa: ANN001
            self.call_count = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return None

        def load_model(self, model_name):  # noqa: ANN001
            return None

        def query(self, model, prompt, timeout=None):  # noqa: ANN001
            self.call_count += 1
            if "READY" in prompt:
                return OllamaQueryResult(model=model, output="{}", returncode=0, stderr="")
            if "Additional Context" not in prompt:
                return OllamaQueryResult(
                    model=model,
                    output='{"request_more": true, "chunk_ids": [2]}',
                    returncode=0,
                    stderr="",
                )
            return OllamaQueryResult(
                model=model,
                output='{"patterns":[{"name":"multi_step","rpn_template":"a b + c +","transformation_steps":["add","add"]}]}',
                returncode=0,
                stderr="",
            )

    monkeypatch.setattr("knowledge3d.ingestion.enrichment_pipeline.OllamaModelManager", _FakeManager)

    pipeline = EnrichmentPipeline(use_local_models=True)
    text = "\n".join(["algebra transformation"] * 300)
    patterns = pipeline.extract_procedural_patterns(content=text, domain="math")
    assert patterns
    assert patterns[0]["name"] == "multi_step"


def test_k3d_transformer_crystallizes_patterns_and_concepts(tmp_path: Path):
    galaxy_manager = GalaxyManager(storage_root=tmp_path / "galaxies")
    transformer = K3DTransformer(galaxy_manager=galaxy_manager)

    enrichment_result = {
        "metadata": {"domain": "math"},
        "patterns": [
            {"name": "power_rule", "rpn_template": "x n pow n *"},
            {"name": "chain_rule", "rpn_template": "f g compose deriv"},
        ],
        "related_concepts": ["derivative", "composition"],
        "embeddings": {64: [0.1] * 64, 128: [0.2] * 128},
    }
    result = transformer.crystallize_enrichment_to_galaxies(
        enrichment_result=enrichment_result,
        target_galaxies=["Math", "Grammar"],
    )

    assert result["galaxy_entries_created"] >= 4
    assert result["rpn_programs_created"] == 2
    assert result["embeddings_stored"] == 192
    assert len(galaxy_manager.get_galaxy("Math").entries) >= 4
