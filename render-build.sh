#!/usr/bin/env bash
# Render Build Script for PRISM
set -o errexit

echo "📦 Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "📦 Installing PRISM requirements..."
pip install -r requirements.txt

echo "📦 Installing PRISM package in editable mode..."
pip install -e .

echo "✅ PRISM build completed successfully!"
