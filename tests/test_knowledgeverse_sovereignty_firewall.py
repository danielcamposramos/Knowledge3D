from __future__ import annotations

import sys

import pytest

from knowledge3d.knowledgeverse.sovereignty_firewall import SovereigntyFirewall


def test_validate_feeder_imports_accepts_allowed_and_stdlib(tmp_path):
    feeder = tmp_path / "good_feeder.py"
    feeder.write_text(
        "import json\n"
        "import re\n"
        "import numpy as np\n"
        "from pathlib import Path\n"
        "from knowledge3d.tools.phase10 import thinking_tag_trainer\n",
        encoding="utf-8",
    )

    is_valid, violations = SovereigntyFirewall.validate_feeder_imports(feeder)

    assert is_valid
    assert violations == []


def test_validate_feeder_imports_rejects_forbidden_feeder_import(tmp_path):
    feeder = tmp_path / "bad_feeder.py"
    feeder.write_text("import cupy as cp\n", encoding="utf-8")

    is_valid, violations = SovereigntyFirewall.validate_feeder_imports(feeder)

    assert not is_valid
    assert any("cupy" in violation for violation in violations)


def test_validate_feeder_imports_rejects_disallowed_import(tmp_path):
    feeder = tmp_path / "bad_import.py"
    feeder.write_text("import requests\n", encoding="utf-8")

    is_valid, violations = SovereigntyFirewall.validate_feeder_imports(feeder)

    assert not is_valid
    assert any("requests" in violation for violation in violations)


def test_runtime_assert_hot_path_detects_forbidden_module(monkeypatch):
    fake_mod = "__sovereign_violation_module__"
    monkeypatch.setitem(sys.modules, fake_mod, object())

    with pytest.raises(RuntimeError, match="Sovereignty violation detected"):
        SovereigntyFirewall.runtime_assert_hot_path(forbidden_modules={fake_mod})


def test_validate_rpn_output_contract():
    valid = {
        "id": "entry_1",
        "program": "1 2 +",
        "entry_point": "main",
        "metadata": {"source": "test"},
    }
    is_valid, reason = SovereigntyFirewall.validate_rpn_output(valid)
    assert is_valid
    assert reason is None

    missing = {
        "id": "entry_2",
        "program": "1 2 +",
        "metadata": {},
    }
    is_valid, reason = SovereigntyFirewall.validate_rpn_output(missing)
    assert not is_valid
    assert reason == "Missing required field: entry_point"

