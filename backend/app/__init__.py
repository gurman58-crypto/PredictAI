"""PredictAI API.

Importing ``app`` makes the sibling ``ml_models`` package importable (the API runs the trained
models in-process) and loads ``backend/.env`` if present.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(_BACKEND_DIR / ".env")

# Directory that CONTAINS the ``ml_models`` package (project root by default).
_ML_ROOT = Path(os.environ.get("PREDICTAI_ML_ROOT", _BACKEND_DIR.parent)).resolve()
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

# keep TensorFlow start-up logs quiet unless the user asks for them
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
