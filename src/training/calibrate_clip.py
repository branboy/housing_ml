"""
Calibrate CLIP condition scores against training-data residuals.

Run once after training the structured model:
    python -m src.training.calibrate_clip

What it does
------------
1. Loads ALL available training homes:
     - 15 k CA homes from fusion_dataset.csv + images
     - 89 multi-state homes from realty_manifest.csv (if present)
   Both datasets are combined so the regression sees the full image pool.
2. Computes a CLIP condition score for every image.
3. Computes the structured-model residual for every home:
       residual = log1p(actual_price) − structured_pred
4. Fits a linear regression:  residual = slope × clip_score + intercept
   The slope tells us how many log-price units a unit change in CLIP score is worth.
5. Saves slope + intercept to outputs/models/clip_calibration.json.

Why calibration matters
-----------------------
Raw CLIP cosine-similarity differences are in roughly [−0.05, +0.05].
Without calibration the price adjustment would be tiny.  The calibration
scales the signal so that a perfectly modern home gets ~+8–12% and a
perfectly outdated home gets ~−8–12%.

The fit is intentionally simple (1-parameter linear regression through the
mean) to avoid overfitting the CA training distribution.
"""

from __future__ import annotations

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import joblib
from src.models.clip_condition import compute_clip_score, CALIBRATION_PATH
from src.utils.structured_predict import predict_structured_from_row

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

FUSION_PATH    = Path("data/processed/fusion_dataset.csv")
REALTY_PATH    = Path("data/processed/realty_manifest.csv")
PIPELINE_PATH  = Path("outputs/models/structured_pipeline.pkl")
# How many images to sample for calibration (None = all)
MAX_IMAGES     = None   # set to e.g. 500 for a quick preview run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fix_image_path(p: str) -> str:
    """Convert stored Windows absolute paths to portable relative paths."""
    p_str = str(p).replace("\\", "/")
    if "data/raw/images" in p_str:
        return str(Path("data/raw/images") / Path(p_str).name)
    return p_str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Load datasets ─────────────────────────────────────────────────────────
    # Always load the CA fusion dataset.
    log.info("Loading CA fusion dataset …")
    df_ca = pd.read_csv(FUSION_PATH)
    df_ca["image_path_fixed"] = df_ca["image_path"].apply(fix_image_path)
    df_ca["dataset_source"]   = "ca_fusion"
    log.info(f"  CA fusion: {len(df_ca):,} homes")

    frames = [df_ca]
    use_multistate = False

    # Also load multi-state (realty) images when available — combine with CA.
    if REALTY_PATH.exists():
        df_ms = pd.read_csv(REALTY_PATH)
        if len(df_ms) > 0:
            df_ms["image_path_fixed"] = df_ms["image_path"].astype(str)
            df_ms["dataset_source"]   = "multi_state"
            log.info(f"  Multi-state manifest: {len(df_ms):,} homes")
            frames.append(df_ms)
            use_multistate = True
        else:
            log.warning("Multi-state manifest is empty — skipping.")
    else:
        log.info("No multi-state manifest found — using CA only.")

    df = pd.concat(frames, ignore_index=True)

    log.info(f"  Combined: {len(df):,} homes  |  "
             f"price range ${df.price.min():,.0f}–${df.price.max():,.0f}")
    if use_multistate:
        log.info(f"  Sources  : CA fusion ({len(df_ca):,})  +  multi-state ({len(df_ms):,})")

    log.info("Loading structured pipeline …")
    pipeline = joblib.load(PIPELINE_PATH)

    # ── Structured predictions + residuals ───────────────────────────────────
    log.info("Computing structured predictions …")
    df["structured_pred"] = predict_structured_from_row(df, pipeline)
    bias = pipeline.get("bias_correction", 0.0)
    df["structured_pred"] += bias
    df["log_price"]  = np.log1p(df["price"])
    df["residual"]   = (df["log_price"] - df["structured_pred"]).clip(-1.0, 1.0)

    log.info(f"  Residual mean={df.residual.mean():.3f}  "
             f"std={df.residual.std():.3f}  "
             f"median={df.residual.median():.3f}")

    # ── Sample if requested ───────────────────────────────────────────────────
    if MAX_IMAGES and len(df) > MAX_IMAGES:
        df = df.sample(MAX_IMAGES, random_state=42).reset_index(drop=True)
        log.info(f"  Sampled {MAX_IMAGES} homes for speed.")

    # ── CLIP scoring ─────────────────────────────────────────────────────────
    import torch
    if torch.cuda.is_available():
        device   = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        eta      = "~2–5 min on GPU"
    elif torch.backends.mps.is_available():
        device   = torch.device("mps")
        gpu_name = "Apple MPS"
        eta      = "~5–10 min on MPS"
    else:
        device   = torch.device("cpu")
        gpu_name = None
        eta      = "~60–90 min on CPU"

    log.info(f"Computing CLIP scores for {len(df):,} images …")
    log.info(f"  Device : {gpu_name or 'CPU'}")
    log.info(f"  ETA    : {eta}")
    log.info("  (First run downloads ~600 MB CLIP weights — subsequent runs use cache)")

    scores, residuals = [], []
    errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), unit="img"):
        img_path = str(row["image_path_fixed"])
        if not Path(img_path).exists():
            errors += 1
            continue
        try:
            score = compute_clip_score(img_path, device=device)
            scores.append(score)
            residuals.append(row["residual"])
        except Exception as exc:
            log.debug(f"  Skipping {img_path}: {exc}")
            errors += 1

    log.info(f"  Scored {len(scores):,} images  |  {errors} skipped")

    if len(scores) < 50:
        log.error(f"Too few valid images ({len(scores)}) — aborting calibration.")
        return

    clip_scores = np.array(scores)
    resids      = np.array(residuals)

    # ── Correlation ───────────────────────────────────────────────────────────
    corr = float(np.corrcoef(clip_scores, resids)[0, 1])
    log.info(f"  CLIP score range : {clip_scores.min():.4f}  to  {clip_scores.max():.4f}")
    log.info(f"  CLIP score mean  : {clip_scores.mean():.4f}  std: {clip_scores.std():.4f}")
    log.info(f"  Pearson corr(CLIP, residual) : {corr:.4f}")

    if abs(corr) < 0.05:
        log.warning("Very low correlation — prompts may not be well-suited to this image set.")

    # ── Linear calibration ───────────────────────────────────────────────────
    # Fit:  residual ≈ slope × clip_score + intercept
    # We force intercept toward zero: average homes should get zero adjustment.
    X = np.column_stack([clip_scores, np.ones(len(clip_scores))])
    beta, _, _, _ = np.linalg.lstsq(X, resids, rcond=None)
    slope, intercept = float(beta[0]), float(beta[1])

    # Clamp intercept to ±0.02 to avoid systematic shift
    intercept = float(np.clip(intercept, -0.02, 0.02))

    # ── R² ───────────────────────────────────────────────────────────────────
    y_pred  = slope * clip_scores + intercept
    ss_res  = np.sum((resids - y_pred) ** 2)
    ss_tot  = np.sum((resids - resids.mean()) ** 2)
    r2      = float(1.0 - ss_res / ss_tot)

    log.info(f"  slope     = {slope:.5f}")
    log.info(f"  intercept = {intercept:.5f}  (clamped to ±0.02)")
    log.info(f"  R²        = {r2:.4f}")
    log.info(f"  Interpretation:")
    log.info(f"    CLIP +0.03 (very modern)  → {slope * 0.03:+.4f} log units "
             f"({(np.exp(slope * 0.03) - 1) * 100:+.1f}%)")
    log.info(f"    CLIP −0.03 (very outdated) → {slope * -0.03:+.4f} log units "
             f"({(np.exp(slope * -0.03) - 1) * 100:+.1f}%)")

    # ── Quartile sanity check ─────────────────────────────────────────────────
    quartiles = pd.qcut(clip_scores, 4, labels=["Q1_outdated","Q2","Q3","Q4_modern"])
    log.info("  Residual by CLIP quartile:")
    for q in ["Q1_outdated","Q2","Q3","Q4_modern"]:
        mask = quartiles == q
        score_range = f"{clip_scores[mask].min():.4f}–{clip_scores[mask].max():.4f}"
        log.info(f"    {q:12s}  score {score_range}  "
                 f"residual median {np.median(resids[mask]):+.3f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    cal = {
        "slope":           slope,
        "intercept":       intercept,
        "r2":              r2,
        "corr":            corr,
        "n_samples":       len(scores),
        "clip_score_mean": float(clip_scores.mean()),
        "clip_score_std":  float(clip_scores.std()),
        "dataset":         f"ca_fusion+multi_state({len(df_ms)})" if use_multistate else "ca_fusion",
    }
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(cal, f, indent=2)

    log.info(f"\nCalibration saved → {CALIBRATION_PATH}")
    log.info("Run  python -m src.inference.predict  to use CLIP condition scoring.")


if __name__ == "__main__":
    main()
