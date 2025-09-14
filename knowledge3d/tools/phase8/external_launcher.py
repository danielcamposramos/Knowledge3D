from __future__ import annotations

"""
External launch URL helpers for the open-plan House.

Examples
  url = ExternalLauncher().generate_launch_url('garden')
  info = ExternalLauncher().handle_launch_url('knowledge3d://house/library')
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ExternalLauncher:
    def generate_launch_url(self, zone: str) -> str:
        z = (zone or '').strip().lower()
        return f"knowledge3d://house/{z}"

    def handle_launch_url(self, url: str) -> Dict[str, str]:
        if not isinstance(url, str):
            return {"status": "error", "message": "Invalid URL"}
        if not url.startswith("knowledge3d://house/"):
            return {"status": "error", "message": "Invalid URL"}
        zone = url.rsplit('/', 1)[-1]
        # Here we would load house_master and spawn avatar at zone
        return {"status": "success", "zone": zone}

