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

    def _is_cron_or_health(self) -> bool:
        clean_path = self.path.split("?")[0].rstrip("/")
        if clean_path in ("/health", "/ping", "/healthz", "/_health", "/status"):
            return True
        user_agent = self.headers.get("User-Agent", "").lower()
        if any(bot in user_agent for bot in ("cron-job", "uptimerobot", "betteruptime", "pingdom", "freshping", "statuscake")):
            return True
        return False

    def do_GET(self):
        if self._is_cron_or_health():
            msg = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return

        super().do_GET()

    def do_HEAD(self):
        if self._is_cron_or_health():
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "15")
            self.end_headers()
            return

        super().do_HEAD()

    def end_headers(self):
        if "favicon" in self.path or self.path.endswith(".html") or self.path in ("/", "") or self._is_cron_or_health():
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        else:
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
