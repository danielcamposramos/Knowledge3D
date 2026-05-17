from __future__ import annotations

import os
import urllib.error
import urllib.request

import pytest


def test_ptx_mcp_endpoint_responds() -> None:
    if os.environ.get("K3D_SKIP_MCP_TESTS") == "1":
        pytest.skip("developer requested MCP smoke skip")
    if os.environ.get("CI"):
        pytest.skip("developer-local MCP smoke only")

    request = urllib.request.Request("http://localhost:8503/mcp/", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=3.0) as response:
            assert int(getattr(response, "status", 200)) < 500
    except urllib.error.HTTPError as exc:
        assert int(exc.code) in {200, 400, 404, 405}
