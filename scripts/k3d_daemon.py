#!/usr/bin/env python3
"""CLI wrapper to launch persistent K3D daemon."""

from __future__ import annotations

from knowledge3d.daemon.main import main


if __name__ == "__main__":
    raise SystemExit(main())
