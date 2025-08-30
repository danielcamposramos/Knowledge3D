"""
Publish local run artifacts (logs, small models) into the repo for documentation.

Copies:
- ../Knowledge3D.local/logs/session-*.jsonl -> docs/reports/logs/
- ../Knowledge3D.local/models/intent.pkl -> docs/reports/models/intent.pkl (sklearn tiny model)

Also writes simple indexes:
- docs/reports/logs/INDEX.md
- docs/reports/models/INDEX.md
"""
from __future__ import annotations

from pathlib import Path
import shutil


def copy_logs(repo_root: Path) -> int:
    local_root = repo_root.parent / f"{repo_root.name}.local"
    src = local_root / "logs"
    dst = repo_root / "docs" / "reports" / "logs"
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in sorted(src.glob("session-*.jsonl")):
        shutil.copy2(p, dst / p.name)
        count += 1
    # Index
    (dst / "INDEX.md").write_text(
        "# Logs Index\n\n" + "\n".join(f"- {p.name}" for p in sorted(dst.glob("session-*.jsonl"))),
        encoding="utf-8",
    )
    return count


def copy_models(repo_root: Path) -> int:
    local_root = repo_root.parent / f"{repo_root.name}.local"
    src = local_root / "models"
    dst = repo_root / "docs" / "reports" / "models"
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    # Only copy small sklearn model
    intent = src / "intent.pkl"
    if intent.exists():
        shutil.copy2(intent, dst / "intent.pkl")
        count += 1
    (dst / "INDEX.md").write_text(
        "# Models Index\n\n" + "\n".join(f"- {p.name} ({p.stat().st_size} bytes)" for p in sorted(dst.iterdir()) if p.is_file()),
        encoding="utf-8",
    )
    return count


def main() -> None:  # pragma: no cover
    repo_root = Path(__file__).resolve().parents[2]
    n_logs = copy_logs(repo_root)
    n_models = copy_models(repo_root)
    print(f"Published logs: {n_logs}, models: {n_models}")


if __name__ == "__main__":
    main()

