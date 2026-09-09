from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
