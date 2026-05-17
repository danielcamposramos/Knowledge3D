"""
Serve ../Knowledge3D.local/datasets over HTTP with permissive CORS.

Usage:
  python3 -m knowledge3d.tools.serve_datasets --port 8766

Then access from the viewer (Exams app) at:
  http://127.0.0.1:8766/exams_index.json
"""
from __future__ import annotations

import http.server
import os
import socketserver
from pathlib import Path
import argparse


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        return super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Serve ../Knowledge3D.local/datasets with CORS")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = repo.parent / f"{repo.name}.local" / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    with socketserver.TCPServer(("127.0.0.1", args.port), CORSRequestHandler) as httpd:
        print(f"Serving {root} at http://127.0.0.1:{args.port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

