#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.mobile_api.benchmark import write_report


if __name__ == "__main__":
    print(json.dumps(write_report(), ensure_ascii=False, indent=2))
