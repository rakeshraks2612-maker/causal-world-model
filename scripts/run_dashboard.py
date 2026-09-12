#!/usr/bin/env python3
"""Run the PRISM Decision Intelligence Dashboard (Task 7.2).

Launches the Streamlit application for interactive decision exploration.
Usage:
    python3 scripts/run_dashboard.py [--port 8501]
"""

import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = REPO_ROOT / "prism" / "dashboard" / "app.py"


def main():
    port = "8501"
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = sys.argv[idx + 1]

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        port,
        "--server.headless",
        "true",
        "--theme.base",
        "dark",
    ]
    print(f"🚀 Launching PRISM Decision Intelligence Dashboard on http://localhost:{port}...")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")


if __name__ == "__main__":
    main()
