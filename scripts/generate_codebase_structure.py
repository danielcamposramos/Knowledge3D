#!/usr/bin/env python3
"""Generate a complete structural metadata dump of the K3D codebase.

This script procedurally derives architecture metadata for every directory
and file in the repository.  It produces K3D-native JSON data structures
that can be queried from a static snapshot (no running instance needed).

Re-run after any code change to refresh the metadata:

    python scripts/generate_codebase_structure.py

Output (all under data/codebase_structure/):
    codebase_structure.json    -- single queryable JSON document
    directories.jsonl          -- one entry per directory (JSONL)
    files.jsonl                -- one entry per source file (JSONL)
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories to skip entirely
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".egg-info", ".venv",
    ".venv_k3d", ".pyenv", ".pytest_cache", ".benchmarks",
    "knowledge3d.egg-info", ".tools", ".claude",
}

# File extensions to include
SOURCE_EXTENSIONS = {
    ".py", ".cu", ".ptx", ".ts", ".js", ".md", ".json", ".jsonl",
    ".css", ".html", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".sh", ".bat", ".glsl", ".wgsl",
}

# ── Architecture component mapping ──────────────────────────────────────

COMPONENT_MAP: dict[str, dict[str, str]] = {
    # Core K3D modules
    "knowledge3d/cranium/kernels":       {"component": "cranium_gpu_kernels",    "layer": "execution",    "runs": "gpu_inference"},
    "knowledge3d/cranium/ptx":           {"component": "cranium_ptx_binaries",   "layer": "execution",    "runs": "gpu_inference"},
    "knowledge3d/cranium/ptx_runtime":   {"component": "cranium_ptx_runtime",    "layer": "execution",    "runs": "gpu_inference"},
    "knowledge3d/cranium/bridges":       {"component": "cranium_bridges",        "layer": "composition",  "runs": "inference"},
    "knowledge3d/cranium/codecs":        {"component": "cranium_codecs",         "layer": "execution",    "runs": "inference"},
    "knowledge3d/cranium/ternary":       {"component": "cranium_ternary",        "layer": "execution",    "runs": "inference"},
    "knowledge3d/cranium/sovereign":     {"component": "cranium_sovereign",      "layer": "execution",    "runs": "system_init"},
    "knowledge3d/cranium/specialists":   {"component": "cranium_specialists",    "layer": "reasoning",    "runs": "inference"},
    "knowledge3d/cranium/tests":         {"component": "cranium_tests",          "layer": "testing",      "runs": "test"},
    "knowledge3d/cranium/ocr":           {"component": "cranium_ocr",            "layer": "ingestion",    "runs": "ingestion"},
    "knowledge3d/cranium/sleep":         {"component": "cranium_sleep",          "layer": "learning",     "runs": "sleep_time_compute"},
    "knowledge3d/cranium/actions":       {"component": "cranium_actions",        "layer": "execution",    "runs": "inference"},
    "knowledge3d/cranium/tools":         {"component": "cranium_tools",          "layer": "execution",    "runs": "inference"},
    "knowledge3d/cranium/utils":         {"component": "cranium_utils",          "layer": "utility",      "runs": "always"},
    "knowledge3d/knowledgeverse/specialists": {"component": "kv_specialists",    "layer": "reasoning",    "runs": "inference"},
    "knowledge3d/knowledgeverse":        {"component": "knowledgeverse",         "layer": "memory",       "runs": "always"},
    "knowledge3d/daemon":                {"component": "daemon",                 "layer": "orchestration", "runs": "always"},
    "knowledge3d/augmentation":          {"component": "augmentation",           "layer": "learning",     "runs": "augmentation"},
    "knowledge3d/ingestion":             {"component": "ingestion",              "layer": "ingestion",    "runs": "ingestion"},
    "knowledge3d/training":              {"component": "training",               "layer": "learning",     "runs": "training"},
    "knowledge3d/sleep":                 {"component": "sleep_compute",          "layer": "learning",     "runs": "sleep_time_compute"},
    "knowledge3d/spatial":               {"component": "spatial",                "layer": "memory",       "runs": "inference"},
    "knowledge3d/gpu":                   {"component": "gpu_management",         "layer": "execution",    "runs": "system_init"},
    "knowledge3d/fog":                   {"component": "fog_compute",            "layer": "execution",    "runs": "inference"},
    "knowledge3d/models":                {"component": "models",                 "layer": "reasoning",    "runs": "inference"},
    "knowledge3d/rl":                    {"component": "reinforcement_learning", "layer": "learning",     "runs": "training"},
    "knowledge3d/skills":                {"component": "skills",                 "layer": "reasoning",    "runs": "inference"},
    "knowledge3d/core":                  {"component": "core",                   "layer": "utility",      "runs": "always"},
    "knowledge3d/utils":                 {"component": "utils",                  "layer": "utility",      "runs": "always"},
    "knowledge3d/bridge":                {"component": "legacy_bridge",          "layer": "composition",  "runs": "inference"},
    "knowledge3d/visualization":         {"component": "visualization",          "layer": "output",       "runs": "on_demand"},
    "knowledge3d/viewer":                {"component": "viewer_backend",         "layer": "output",       "runs": "on_demand"},
    # Non-knowledge3d top-level dirs
    "benchmarks":                        {"component": "benchmarks",             "layer": "testing",      "runs": "benchmark"},
    "tests":                             {"component": "tests",                  "layer": "testing",      "runs": "test"},
    "scripts":                           {"component": "scripts",                "layer": "tooling",      "runs": "on_demand"},
    "viewer":                            {"component": "viewer_frontend",        "layer": "output",       "runs": "on_demand"},
    "docs":                              {"component": "documentation",          "layer": "documentation","runs": "reference"},
    "docs/vocabulary":                   {"component": "specifications",         "layer": "documentation","runs": "reference"},
    "docs/briefings":                    {"component": "briefings",              "layer": "documentation","runs": "reference"},
    "data":                              {"component": "data",                   "layer": "data",         "runs": "reference"},
    "TEMP":                              {"component": "temp_specs",             "layer": "documentation","runs": "reference"},
    "docker":                            {"component": "docker",                 "layer": "infrastructure","runs": "deployment"},
    "tools":                             {"component": "dev_tools",              "layer": "tooling",      "runs": "on_demand"},
}

# ── Three Brain mapping ─────────────────────────────────────────────────

THREE_BRAIN_MAP = {
    "cranium":       "Cranium (Execution Brain)",
    "knowledgeverse":"Galaxy Universe (Memory Brain)",
    "daemon":        "Orchestration (House Brain)",
    "spatial":       "Galaxy Universe (Spatial)",
    "augmentation":  "Galaxy Universe (Ingestion)",
    "ingestion":     "Galaxy Universe (Ingestion)",
    "training":      "Cranium (Learning)",
    "sleep":         "Cranium (Sleep-Time Compute)",
    "models":        "Cranium (TRM)",
    "rl":            "Cranium (Learning)",
    "gpu":           "Cranium (GPU Management)",
    "fog":           "House (Fog Compute)",
    "skills":        "Cranium (Skills)",
    "visualization": "House (Visualization)",
    "viewer":        "House (Viewer)",
}

# ── Galaxy mapping for data files ───────────────────────────────────────

GALAXY_MAP = {
    "Drawing":    "Drawing Galaxy (visual primitives)",
    "Character":  "Character Galaxy (procedural glyphs)",
    "Word":       "Word Galaxy (lexemes)",
    "Grammar":    "Grammar Galaxy (transformation rules)",
    "Math":       "Math Galaxy (symbols, templates)",
    "Reality":    "Reality Galaxy (physics, chemistry, biology)",
    "Audio":      "Audio Galaxy (temporal patterns)",
    "3DObjects":  "3DObjects Galaxy (mesh/geometry)",
    "Tool":       "Tool Galaxy (execution nodes)",
}


def _git_last_modified(path: Path) -> str:
    """Return ISO date of last git commit touching this path."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%aI", "--", str(path)],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=5,
        )
        return result.stdout.strip() or ""
    except Exception:
        return ""


def _file_hash(path: Path) -> str:
    """Return short SHA-256 of file contents."""
    try:
        h = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        return h
    except Exception:
        return ""


def _resolve_component(rel_path: str) -> dict[str, str]:
    """Find the best matching component for a relative path."""
    parts = rel_path.replace("\\", "/")
    best_match = ""
    best_info: dict[str, str] = {"component": "other", "layer": "other", "runs": "unknown"}
    for prefix, info in COMPONENT_MAP.items():
        if parts.startswith(prefix) and len(prefix) > len(best_match):
            best_match = prefix
            best_info = info
    return best_info


def _resolve_three_brain(rel_path: str) -> str:
    """Map path to Three Brain System component."""
    parts = rel_path.split("/")
    if len(parts) >= 2 and parts[0] == "knowledge3d":
        module = parts[1]
        return THREE_BRAIN_MAP.get(module, "")
    if parts[0] == "benchmarks":
        return "House (Benchmarks)"
    if parts[0] == "viewer":
        return "House (Viewer)"
    if parts[0] == "docs":
        return "Documentation"
    return ""


def _analyze_python_file(path: Path) -> dict[str, Any]:
    """Extract structural metadata from a Python file."""
    info: dict[str, Any] = {
        "classes": [],
        "functions": [],
        "imports": [],
        "galaxy_refs": [],
        "sovereignty_flags": [],
    }
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return info

    # AST analysis
    try:
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(ast.dump(base))
                info["classes"].append({
                    "name": node.name,
                    "line": node.lineno,
                    "bases": bases,
                    "docstring": ast.get_docstring(node) or "",
                })
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Only top-level and class-level functions
                    info["functions"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "docstring": ast.get_docstring(node) or "",
                    })
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    info["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    info["imports"].append(node.module)
    except SyntaxError:
        pass

    # Galaxy references
    for galaxy in GALAXY_MAP:
        if galaxy in source:
            info["galaxy_refs"].append(galaxy)

    # Sovereignty flags
    for lib in ("numpy", "cupy", "scipy", "sympy", "torch", "tensorflow"):
        if re.search(rf"\bimport\s+{lib}\b", source) or re.search(rf"\bfrom\s+{lib}\b", source):
            info["sovereignty_flags"].append(f"imports_{lib}")

    # Detect PTX/CUDA references
    if "ctypes" in source and (".ptx" in source or ".cu" in source):
        info["sovereignty_flags"].append("sovereign_ptx_loader")
    if "rpn_program" in source or "RPN" in source:
        info["galaxy_refs"].append("RPN_execution")

    return info


def _analyze_cuda_file(path: Path) -> dict[str, Any]:
    """Extract metadata from .cu or .ptx files."""
    info: dict[str, Any] = {"kernels": [], "entry_points": []}
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return info

    # Find __global__ kernel declarations
    for match in re.finditer(r"__global__\s+void\s+(\w+)", source):
        info["kernels"].append(match.group(1))

    # Find .entry points in PTX
    for match in re.finditer(r"\.entry\s+(\w+)", source):
        info["entry_points"].append(match.group(1))

    return info


def _analyze_markdown_file(path: Path) -> dict[str, Any]:
    """Extract metadata from markdown files."""
    info: dict[str, Any] = {"title": "", "sections": []}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    except Exception:
        return info

    for line in lines[:100]:
        if line.startswith("# ") and not info["title"]:
            info["title"] = line[2:].strip()
        elif line.startswith("## "):
            info["sections"].append(line[3:].strip())

    return info


def _analyze_jsonl_file(path: Path) -> dict[str, Any]:
    """Extract metadata from JSONL galaxy files."""
    info: dict[str, Any] = {"entry_count": 0, "sample_fields": [], "galaxy_name": ""}
    try:
        count = 0
        sample: dict[str, Any] = {}
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                count += 1
                if count == 1:
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        pass
        info["entry_count"] = count
        if sample:
            info["sample_fields"] = sorted(sample.keys())
            if "domain" in sample:
                info["galaxy_name"] = str(sample.get("domain", ""))
            elif "category" in sample:
                info["galaxy_name"] = str(sample.get("category", ""))
    except Exception:
        pass
    return info


def _describe_file(rel_path: str, ext: str) -> str:
    """Generate a short description based on path and extension."""
    parts = rel_path.split("/")
    name = parts[-1] if parts else ""

    if ext == ".cu":
        return f"CUDA kernel source: {name}"
    if ext == ".ptx":
        return f"Compiled PTX assembly: {name}"
    if ext == ".py" and "test" in name.lower():
        return f"Test suite: {name}"
    if ext == ".py" and parts[0] == "scripts":
        return f"Utility script: {name}"
    if ext == ".md" and parts[0] == "docs":
        return f"Documentation: {name}"
    if ext == ".jsonl" and "galaxies" in rel_path.lower():
        galaxy = name.replace(".jsonl", "").replace("_enriched", "")
        return f"Galaxy data: {GALAXY_MAP.get(galaxy, galaxy)}"

    return ""


def generate_structure() -> dict[str, Any]:
    """Walk the repo and generate the full structural metadata."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    directories: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    stats = {
        "total_directories": 0,
        "total_files": 0,
        "by_extension": {},
        "by_component": {},
        "by_layer": {},
        "by_three_brain": {},
    }

    for dirpath_str, dirnames, filenames in os.walk(REPO_ROOT):
        dirpath = Path(dirpath_str)
        rel_dir = str(dirpath.relative_to(REPO_ROOT)).replace("\\", "/")

        # Skip excluded directories
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        if rel_dir == ".":
            rel_dir = ""

        # Skip some top-level dirs that aren't part of K3D source
        skip_prefixes = ("Old_Attempts", "ext", "Large_Assets", "envs", ".vscode",
                         "logs", "metrics", "output", "results", "telemetry_output",
                         "validation_cache", "validation_results", "codeopt",
                         "jules-scratch", "reports", "checkpoints", "fonts",
                         "build", "demo", "examples", "spec")
        if rel_dir and any(rel_dir.startswith(p) for p in skip_prefixes):
            dirnames.clear()
            continue

        comp = _resolve_component(rel_dir) if rel_dir else {"component": "root", "layer": "root", "runs": "always"}
        brain = _resolve_three_brain(rel_dir) if rel_dir else ""

        dir_entry: dict[str, Any] = {
            "path": rel_dir or ".",
            "type": "directory",
            "component": comp["component"],
            "layer": comp["layer"],
            "runs": comp["runs"],
            "three_brain": brain,
            "file_count": 0,
            "subdirectories": sorted(dirnames),
        }

        file_count = 0
        for filename in sorted(filenames):
            filepath = dirpath / filename
            ext = filepath.suffix.lower()

            if ext not in SOURCE_EXTENSIONS:
                continue

            rel_file = str(filepath.relative_to(REPO_ROOT)).replace("\\", "/")
            file_comp = _resolve_component(rel_file)

            file_entry: dict[str, Any] = {
                "path": rel_file,
                "type": "file",
                "extension": ext,
                "size_bytes": filepath.stat().st_size if filepath.exists() else 0,
                "content_hash": _file_hash(filepath),
                "component": file_comp["component"],
                "layer": file_comp["layer"],
                "runs": file_comp["runs"],
                "three_brain": _resolve_three_brain(rel_file),
                "description": _describe_file(rel_file, ext),
            }

            # Extension-specific analysis
            if ext == ".py":
                py_info = _analyze_python_file(filepath)
                file_entry["classes"] = [c["name"] for c in py_info["classes"]]
                file_entry["class_details"] = py_info["classes"]
                file_entry["functions"] = [f["name"] for f in py_info["functions"]]
                file_entry["function_details"] = py_info["functions"]
                file_entry["imports"] = py_info["imports"]
                file_entry["galaxy_refs"] = py_info["galaxy_refs"]
                file_entry["sovereignty_flags"] = py_info["sovereignty_flags"]

            elif ext in (".cu", ".ptx"):
                cuda_info = _analyze_cuda_file(filepath)
                file_entry["kernels"] = cuda_info.get("kernels", [])
                file_entry["entry_points"] = cuda_info.get("entry_points", [])

            elif ext == ".md":
                md_info = _analyze_markdown_file(filepath)
                file_entry["title"] = md_info.get("title", "")
                file_entry["sections"] = md_info.get("sections", [])

            elif ext == ".jsonl":
                jsonl_info = _analyze_jsonl_file(filepath)
                file_entry["entry_count"] = jsonl_info.get("entry_count", 0)
                file_entry["sample_fields"] = jsonl_info.get("sample_fields", [])
                file_entry["galaxy_name"] = jsonl_info.get("galaxy_name", "")

            files.append(file_entry)
            file_count += 1

            # Stats
            stats["by_extension"][ext] = stats["by_extension"].get(ext, 0) + 1
            stats["by_component"][file_comp["component"]] = stats["by_component"].get(file_comp["component"], 0) + 1
            stats["by_layer"][file_comp["layer"]] = stats["by_layer"].get(file_comp["layer"], 0) + 1
            if brain:
                stats["by_three_brain"][brain] = stats["by_three_brain"].get(brain, 0) + 1

        dir_entry["file_count"] = file_count
        if rel_dir:  # Don't include root as a directory entry
            directories.append(dir_entry)
            stats["total_directories"] += 1

    stats["total_files"] = len(files)

    # Build the full structure document
    structure: dict[str, Any] = {
        "metadata": {
            "generated_at": timestamp,
            "generator": "scripts/generate_codebase_structure.py",
            "repo_root": str(REPO_ROOT),
            "description": (
                "Complete structural metadata for the Knowledge3D codebase. "
                "Procedurally derived from source code -- re-run on change to refresh. "
                "All data in K3D-native JSON, queryable from a static snapshot."
            ),
            "schema_version": "1.0",
            "re_generate_command": "python scripts/generate_codebase_structure.py",
        },
        "architecture": {
            "three_brain_system": {
                "cranium": "Execution Brain -- GPU kernels, PTX runtime, bridges, codecs, ternary logic",
                "galaxy_universe": "Memory Brain -- Knowledgeverse, Galaxy entries, spatial layout, augmentation",
                "house": "Orchestration Brain -- Daemon, viewer, visualization, fog compute",
            },
            "layers": {
                "execution":      "GPU kernels, PTX binaries, sovereign compute",
                "composition":    "Bridges that compose primitives into higher-level operations",
                "reasoning":      "TRM navigation, specialists, routing logic",
                "memory":         "Galaxy Universe, Knowledgeverse state, spatial layout",
                "learning":       "Training, augmentation, sleep-time compute, RL",
                "ingestion":      "Data ingest pipelines (fonts, lexicons, documents)",
                "orchestration":  "Daemon, command dispatch, lifecycle management",
                "output":         "Viewer, visualization, rendering",
                "testing":        "Test suites, benchmarks",
                "tooling":        "Scripts, dev tools",
                "documentation":  "Specs, briefings, papers",
                "data":           "Galaxy snapshots, execution journals, training data",
                "infrastructure": "Docker, CI/CD",
            },
            "sovereignty_rule": (
                "Hot path (inference) must be sovereign: PTX + Galaxy only. "
                "No numpy, cupy, scipy, sympy in inference loops. "
                "Ingestion path may use any libraries."
            ),
            "galaxy_composition": {
                "order": ["Drawing", "Character", "Word", "Grammar", "Math", "Reality", "Audio", "3DObjects", "Tool"],
                "principle": (
                    "Each layer references the one below. Words reference Characters, "
                    "Characters reference Drawing primitives. Save Information Principle: "
                    "no duplication, only symlinks."
                ),
            },
        },
        "statistics": stats,
        "directories": directories,
        "files": files,
    }

    return structure


def main() -> int:
    print("Generating codebase structural metadata...")
    structure = generate_structure()

    out_dir = REPO_ROOT / "data" / "codebase_structure"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full structure document
    full_path = out_dir / "codebase_structure.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(structure, f, indent=2, ensure_ascii=False, sort_keys=False)
    print(f"  Written: {full_path.relative_to(REPO_ROOT)}  ({full_path.stat().st_size / 1024:.0f}KB)")

    # Directory JSONL
    dir_path = out_dir / "directories.jsonl"
    with open(dir_path, "w", encoding="utf-8") as f:
        for entry in structure["directories"]:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=False) + "\n")
    print(f"  Written: {dir_path.relative_to(REPO_ROOT)}  ({len(structure['directories'])} entries)")

    # File JSONL
    file_path = out_dir / "files.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in structure["files"]:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=False) + "\n")
    print(f"  Written: {file_path.relative_to(REPO_ROOT)}  ({len(structure['files'])} entries)")

    print(f"\nStats:")
    print(f"  Directories: {structure['statistics']['total_directories']}")
    print(f"  Files:       {structure['statistics']['total_files']}")
    print(f"  By layer:    {json.dumps(structure['statistics']['by_layer'], indent=4)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
