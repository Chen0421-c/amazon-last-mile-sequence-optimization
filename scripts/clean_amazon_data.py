#!/usr/bin/env python3
"""Run the Amazon Last Mile cleaning pipeline from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from last_mile_cleaning.clean_pipeline import main


if __name__ == "__main__":
    main()
