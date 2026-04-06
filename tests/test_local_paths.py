from __future__ import annotations

from pathlib import Path

from knowledge3d import local_paths


def test_resolve_storage_root_prefers_explicit_value(tmp_path: Path) -> None:
    explicit = tmp_path / "custom-root"
    assert local_paths.resolve_storage_root(explicit) == explicit


def test_default_storage_root_prefers_existing_local_path(monkeypatch, tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    sibling_root = tmp_path / "sibling"
    local_root.mkdir()
    sibling_root.mkdir()

    monkeypatch.setattr(local_paths, "PRIMARY_LOCAL_STORAGE_ROOT", local_root)
    monkeypatch.setattr(local_paths, "repo_sibling_storage_root", lambda: sibling_root)
    monkeypatch.delenv("K3D_STORAGE_ROOT", raising=False)
    monkeypatch.delenv("K3D_LOCAL_DIR", raising=False)

    assert local_paths.default_storage_root() == local_root


def test_default_storage_root_honors_env_override(monkeypatch, tmp_path: Path) -> None:
    env_root = tmp_path / "env-root"
    local_root = tmp_path / "local"
    sibling_root = tmp_path / "sibling"
    env_root.mkdir()
    local_root.mkdir()
    sibling_root.mkdir()

    monkeypatch.setenv("K3D_STORAGE_ROOT", str(env_root))
    monkeypatch.setattr(local_paths, "PRIMARY_LOCAL_STORAGE_ROOT", local_root)
    monkeypatch.setattr(local_paths, "repo_sibling_storage_root", lambda: sibling_root)

    assert local_paths.default_storage_root() == env_root
