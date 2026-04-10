from __future__ import annotations

import sys


ARCHIVE_REASON = "arc3_local_archived_use_arc3_sdk_agent_or_headless_tablet_runner"


def main() -> int:
    print(ARCHIVE_REASON, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
