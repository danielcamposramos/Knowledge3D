"""Sovereignty boundary checks for Knowledgeverse ingestion and runtime.

The firewall enforces two contracts:
1. Ingestion feeders are statically validated before execution.
2. Runtime hot-path modules can assert forbidden dependencies are not loaded.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable, Mapping


class SovereigntyFirewall:
    """Boundary security for Ingestion Stargate and hot path checks."""

    # Ingestion-path libraries are allowed to be imported by feeder scripts.
    # These imports are still outside the sovereign hot path.
    ALLOWED_INGESTION_LIBS = {
        "argparse",
        "collections",
        "csv",
        "cv2",
        "json",
        "math",
        "numpy",
        "os",
        "pandas",
        "pathlib",
        "pdfplumber",
        "PIL",
        "re",
        "subprocess",
        "sys",
        "torch",
        "transformers",
        "typing",
    }

    # These modules are known to create sovereignty risk in feeder scripts.
    # They are disallowed here to prevent accidental leakage into runtime paths.
    FORBIDDEN_FEEDER_LIBS = {
        "cupy",
        "scipy",
        "sympy",
    }

    # If any of these are loaded during hot-path execution, fail fast.
    FORBIDDEN_HOT_PATH_LIBS = {
        "cupy",
        "jax",
        "numpy",
        "scipy",
        "sympy",
        "tensorflow",
        "torch",
    }

    @staticmethod
    def _stdlib_modules() -> set[str]:
        names = getattr(sys, "stdlib_module_names", None)
        if not names:
            return set()
        return set(names)

    @staticmethod
    def _module_root(module_name: str | None) -> str:
        if not module_name:
            return ""
        return module_name.split(".", 1)[0]

    @classmethod
    def validate_feeder_imports(cls, feeder_path: str | Path) -> tuple[bool, list[str]]:
        """Validate feeder imports using static AST checks.

        Returns:
            (is_valid, violations)
        """

        feeder_path = Path(feeder_path)
        tree = ast.parse(feeder_path.read_text(encoding="utf-8"), filename=str(feeder_path))

        violations: list[str] = []
        stdlib = cls._stdlib_modules()

        def _check_module(module_name: str | None) -> None:
            root = cls._module_root(module_name)
            if not root:
                return
            if root in cls.FORBIDDEN_FEEDER_LIBS:
                violations.append(f"Forbidden feeder import: {root}")
                return
            if root.startswith("knowledge3d"):
                return
            if root in cls.ALLOWED_INGESTION_LIBS:
                return
            if root in stdlib:
                return
            violations.append(f"Disallowed import: {root}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    _check_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                _check_module(node.module)

        return len(violations) == 0, violations

    @classmethod
    def validate_rpn_output(
        cls,
        candidate_rpn: Mapping[str, object],
    ) -> tuple[bool, str | None]:
        """Validate Stargate feeder output conforms to minimal RPN schema."""

        required_fields = ("id", "program", "entry_point", "metadata")
        for field in required_fields:
            if field not in candidate_rpn:
                return False, f"Missing required field: {field}"

        program = candidate_rpn.get("program")
        if not isinstance(program, str):
            return False, "RPN program must be string, not executable code"
        if not program.strip():
            return False, "RPN program must be non-empty"

        metadata = candidate_rpn.get("metadata")
        if not isinstance(metadata, dict):
            return False, "metadata must be a dict"

        return True, None

    @classmethod
    def runtime_assert_hot_path(cls, forbidden_modules: Iterable[str] | None = None) -> None:
        """Fail fast if forbidden modules are loaded in runtime process."""

        forbidden = set(forbidden_modules or cls.FORBIDDEN_HOT_PATH_LIBS)
        loaded_forbidden: list[str] = []

        for module in forbidden:
            if module in sys.modules:
                loaded_forbidden.append(module)
                continue
            prefix = f"{module}."
            if any(name.startswith(prefix) for name in sys.modules):
                loaded_forbidden.append(module)

        if loaded_forbidden:
            loaded_forbidden = sorted(set(loaded_forbidden))
            raise RuntimeError(
                "Sovereignty violation detected: "
                f"{loaded_forbidden}. Hot path must be PTX-only."
            )

