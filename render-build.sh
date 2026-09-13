#!/usr/bin/env bash
# Render Build Script for PRISM
set -o errexit

echo "📦 Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "📦 Installing PRISM requirements..."
pip install -r requirements.txt

echo "📦 Installing PRISM package in editable mode..."
pip install -e .

echo "🎨 Patching Streamlit static assets with official PRISM logo..."
python -c "
import streamlit, shutil
from pathlib import Path
st_static = Path(streamlit.__file__).parent / 'static'
src_favicon = Path('prism/dashboard/web/favicon.png')
src_svg = Path('prism/dashboard/web/favicon.svg')
src_ico = Path('prism/dashboard/web/favicon.ico')

if st_static.exists():
    if src_favicon.exists():
        shutil.copyfile(src_favicon, st_static / 'favicon.png')
    if src_svg.exists():
        shutil.copyfile(src_svg, st_static / 'favicon.svg')
    if src_ico.exists():
        shutil.copyfile(src_ico, st_static / 'favicon.ico')
    index_path = st_static / 'index.html'
    if index_path.exists():
        html = index_path.read_text(encoding='utf-8')
        html = html.replace('<title>Streamlit</title>', '<title>PRISM · Superintelligence for physical judgment</title>')
        html = html.replace('href=\"./favicon.png\"', 'href=\"./favicon.png?v=prism\"')
        index_path.write_text(html, encoding='utf-8')
        print('✅ Patched Streamlit static favicon and title!')
"

echo "✅ PRISM build completed successfully!"

