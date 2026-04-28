"""
Subprocess prediction worker.

app.py calls this via subprocess.run() and passes a JSON payload on stdin.
Running ML inference in a child process means TF, Keras, XGBoost, and CLIP
are completely isolated from Streamlit's server thread — any warnings, slow
imports, or device-mismatch messages are contained here and never touch the
WebSocket that keeps the browser connected.

Usage (called internally by app.py):
    echo '{"input_dict": {...}, "image_paths": []}' | python -m src.inference.predict_worker

Legacy single-image key "image_path" is still accepted for backwards compat.
"""

import sys
import json
import os
import warnings
import logging

# ── Ensure project root is on sys.path ───────────────────────────────────────
# __file__ is  .../housing_ml/src/inference/predict_worker.py
# project root is two levels up
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── Silence everything before any ML import ──────────────────────────────────
os.environ["TRANSFORMERS_NO_TF"]    = "1"
os.environ["USE_TF"]                = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

# ── Load project environment (.env for RENTCAST_API_KEY etc.) ─────────────────
from dotenv import load_dotenv      # noqa: E402
load_dotenv()

# ── Silence XGBoost C++ logger ────────────────────────────────────────────────
import xgboost as xgb               # noqa: E402
xgb.set_config(verbosity=0)

# ── Main prediction import ────────────────────────────────────────────────────
from src.inference.predict import predict_price  # noqa: E402


def main():
    try:
        payload    = json.load(sys.stdin)
        input_dict = payload["input_dict"]

        # Accept list of paths (new) or single path (legacy) — normalise to list
        image_paths = payload.get("image_paths") or []
        legacy      = payload.get("image_path")
        if not image_paths and legacy:
            image_paths = [legacy]

        price, log_lines = predict_price(input_dict, image_paths)

        json.dump({"ok": True, "price": price, "log_lines": log_lines}, sys.stdout)
        sys.stdout.flush()

    except Exception as exc:
        import traceback
        json.dump({"ok": False, "error": str(exc), "traceback": traceback.format_exc()},
                  sys.stdout)
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
