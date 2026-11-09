"""Ensure the repository root is importable when pytest is invoked directly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
