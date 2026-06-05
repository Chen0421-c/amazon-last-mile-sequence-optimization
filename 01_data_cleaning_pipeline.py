#!/usr/bin/env python3
"""Entry point for the dissertation data cleaning pipeline.

Run this script in Google Colab after mounting Drive. It reads the Amazon Last
Mile JSON files from Google Drive route-by-route and writes cleaned CSV outputs
under ``processed_outputs``. The original JSON files are never modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from last_mile_cleaning.clean_pipeline import main


if __name__ == "__main__":
    main()
