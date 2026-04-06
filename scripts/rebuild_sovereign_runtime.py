#!/usr/bin/env python3
"""Compatibility shim for the canonical sovereign artifact rebuild tool."""

from __future__ import annotations

from rebuild_sovereign_artifact import main


if __name__ == "__main__":
    raise SystemExit(main())
