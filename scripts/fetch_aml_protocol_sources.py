#!/usr/bin/env python3
"""Download hash-pinned AML sources for optional local QA; review upstream terms."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mandol_competition.textual.core.aml_protocol_sources import main

if __name__ == "__main__":
    main(checkout=ROOT)
