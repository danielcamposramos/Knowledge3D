#!/usr/bin/env python3
"""Archived transitional ARC validator.

This script used a CPU-side `ARCRPNExecutor` path and is no longer part of the
live sovereign benchmark runtime. The historical version lives at
`Old_Attempts/scripts/evaluate_arc_with_validation.py`.
"""

from __future__ import annotations

import sys


ARCHIVE_REASON = "evaluate_arc_with_validation_archived_use_arc2_local_runner_or_tablet_boundary"


def main() -> int:
    print(ARCHIVE_REASON, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
