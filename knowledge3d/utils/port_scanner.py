from __future__ import annotations

import socket
from typing import Optional


class PortScanner:
    @staticmethod
    def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
        """Return True if TCP port is in use on host.

        Uses a non-blocking connect_ex check.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.15)
                return s.connect_ex((host, int(port))) == 0
        except Exception:
            return False

    @staticmethod
    def find_free_port(start_port: int = 8765, max_attempts: int = 100, host: str = "127.0.0.1") -> Optional[int]:
        for port in range(int(start_port), int(start_port) + int(max_attempts)):
            if not PortScanner.is_port_in_use(port, host=host):
                return int(port)
        return None

    @staticmethod
    def get_dynamic_port(config_port: Optional[int] = None, host: str = "127.0.0.1") -> int:
        if config_port is not None and not PortScanner.is_port_in_use(int(config_port), host=host):
            return int(config_port)
        port = PortScanner.find_free_port(8765, host=host)
        if port is not None:
            print(f"[PortScanner] dynamic port assigned: {port}")
            return port
        port = PortScanner.find_free_port(10000, host=host)
        if port is not None:
            print(f"[PortScanner] fallback port assigned: {port}")
            return port
        raise RuntimeError("No free port found")

