#!/usr/bin/env python3
"""Run the PRISM Web Dashboard on port $PORT (default 8000).

Directly serves the standalone PRISM Web SPA (prism/dashboard/web/index.html)
without Streamlit, websockets, or external dependencies.
"""

import os
import sys
import http.server
import socketserver
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "prism" / "dashboard" / "web"


class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = http.server.SimpleHTTPRequestHandler.extensions_map.copy()
    extensions_map.update({
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
        ".png": "image/png",
        ".html": "text/html; charset=utf-8",
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "public, max-age=3600")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def main():
    port = int(os.environ.get("PORT", 8000))
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])

    socketserver.TCPServer.allow_reuse_address = True

    print(f"🚀 Serving PRISM Web Dashboard from {WEB_DIR} on http://0.0.0.0:{port}...")
    with socketserver.TCPServer(("0.0.0.0", port), CustomHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped.")


if __name__ == "__main__":
    main()
