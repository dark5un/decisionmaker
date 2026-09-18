"""Ensure `server.src.*` resolves to server/src via the repo ROOT (namespace layout)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))