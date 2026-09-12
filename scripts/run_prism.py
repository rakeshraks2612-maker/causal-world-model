#!/usr/bin/env python3
"""PRISM Executable CLI Wrapper (Task 7.1)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prism.cli import main

if __name__ == "__main__":
    main()
