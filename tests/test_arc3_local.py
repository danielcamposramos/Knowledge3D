from __future__ import annotations

from pathlib import Path

import pytest

import benchmarks.arc3_local as arc3_local


def test_arc3_local_module_is_archived() -> None:
    assert arc3_local.ARCHIVED_TRANSITIONAL_SURFACE is True
    assert arc3_local.ARCHIVED_PATH == "Old_Attempts/benchmarks/arc3_local.py"
    assert "archived" in arc3_local.ARCHIVE_REASON


def test_arc3_local_runtime_entrypoints_raise_archived_error() -> None:
    with pytest.raises(RuntimeError, match="arc3_local_archived"):
        arc3_local.run_local_arc3()
    with pytest.raises(RuntimeError, match="arc3_local_archived"):
        arc3_local.make_task(0)


def test_run_arc3_local_script_is_archived() -> None:
    source = Path("scripts/run_arc3_local.py").read_text(encoding="utf-8")
    assert "arc3_local_archived_use_arc3_sdk_agent_or_headless_tablet_runner" in source
