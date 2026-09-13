#!/usr/bin/env python3
"""Run the PRISM Decision Intelligence Dashboard (Task 7.2).

Launches the Streamlit application for interactive decision exploration.
Usage:
    python3 scripts/run_dashboard.py [--port 8501]
"""

import os
import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = REPO_ROOT / "prism" / "dashboard" / "app.py"


def patch_streamlit_static():
    """Overwrite Streamlit's default static favicon and title with official PRISM assets."""
    try:
        import streamlit
        import shutil
        st_static = Path(streamlit.__file__).parent / "static"
        src_favicon = REPO_ROOT / "prism" / "dashboard" / "web" / "favicon.png"
        src_svg = REPO_ROOT / "prism" / "dashboard" / "web" / "favicon.svg"
        src_ico = REPO_ROOT / "prism" / "dashboard" / "web" / "favicon.ico"
        if st_static.exists():
            if src_favicon.exists():
                shutil.copyfile(src_favicon, st_static / "favicon.png")
            if src_svg.exists():
                shutil.copyfile(src_svg, st_static / "favicon.svg")
            if src_ico.exists():
                shutil.copyfile(src_ico, st_static / "favicon.ico")
            index_path = st_static / "index.html"
            if index_path.exists():
                html = index_path.read_text(encoding="utf-8")
                html = html.replace("<title>Streamlit</title>", "<title>PRISM · Superintelligence for physical judgment</title>")
                html = html.replace('href="./favicon.png"', 'href="./favicon.png?v=prism"')
                index_path.write_text(html, encoding="utf-8")
    except Exception as e:
        print(f"Notice: could not patch streamlit static assets: {e}")


def main():
    patch_streamlit_static()
    port = os.environ.get("PORT", "8501")
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
        str(port),
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
        "--browser.gatherUsageStats",
        "false",
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
