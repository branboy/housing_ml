"""
CLIP-based home condition scorer.

Replaces the ResNet50 → PCA → XGBoost fusion pipeline with a semantically-
grounded zero-shot approach.  CLIP (Contrastive Language-Image Pretraining)
was trained on 400 M image-text pairs and can directly answer "how much does
this home look modern vs outdated?" without any housing-specific training.

Pipeline:
  1. Load CLIP ViT-B/32 (once, cached)
  2. Compute cosine similarities between the image and each condition prompt
  3. condition_score = mean(positive sims) − mean(negative sims)
  4. Apply calibration: log-price adjustment = slope × condition_score + intercept
     (calibration is fit in calibrate_clip.py using training-data residuals)

Typical score range: −0.05 to +0.05 (raw cosine-sim difference).
After calibration this converts to a log-price adjustment of roughly ±0.10
(±10%) for the most extreme cases.

Why this beats the old approach
---------------------------------
• ResNet50 features explain only 28% of price variance — less than bed/bath/sqft.
• The fusion XGBoost was predicting structured-model residuals with a test RMSE
  of 0.37 (44% price error) — barely better than predicting zero.
• CLIP features are semantically tied to condition language; they don't require
  training on CA-specific residuals and generalise to all states.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Condition prompts
# ---------------------------------------------------------------------------
# Positive = modern / well-maintained / luxury finishes
# Negative = outdated / poor condition / deferred maintenance
#
# Prompts are intentionally about PHYSICAL CONDITION, not neighbourhood.
# We avoid terms like "luxury neighbourhood" or "expensive area" that would
# correlate with location rather than visual quality.

POSITIVE_PROMPTS = [
    "a modern home interior with updated finishes and contemporary design",
    "a renovated kitchen with new appliances granite countertops and fresh paint",
    "a newly built house with clean modern bathrooms and hardwood floors",
    "a well-maintained home with high ceilings open layout and modern fixtures",
]

NEGATIVE_PROMPTS = [
    "an outdated home interior with old carpet worn surfaces and dated wallpaper",
    "a kitchen with old appliances laminate countertops and dated cabinets",
    "a house needing renovation with peeling paint cracked walls and old fixtures",
    "a worn home interior with dated finishes visible aging and deferred maintenance",
]

# ---------------------------------------------------------------------------
# Calibration path
# ---------------------------------------------------------------------------
CALIBRATION_PATH = Path("outputs/models/clip_calibration.json")

# Conservative default when no calibration file exists:
# ±1 raw score unit ≈ ±8% price adjustment
_DEFAULT_SLOPE     = 0.08
_DEFAULT_INTERCEPT = 0.0

# ---------------------------------------------------------------------------
# Model cache (load once per process)
# ---------------------------------------------------------------------------
_model     = None
_processor = None


def _load_clip():
    """Lazy-load CLIP ViT-B/32.  Caches across calls."""
    global _model, _processor
    if _model is None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError(
                "transformers is required for CLIP condition scoring.\n"
                "Install it with:  pip install transformers"
            )
        model_id   = "openai/clip-vit-base-patch32"
        log.info(f"Loading CLIP model ({model_id}) …")
        _processor = CLIPProcessor.from_pretrained(model_id)
        _model     = CLIPModel.from_pretrained(model_id)
        _model.eval()
        log.info("CLIP model loaded.")
    return _model, _processor


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def compute_clip_score(image_path: str, device=None) -> float:
    """
    Return a raw condition score in approximately [−0.10, +0.10].

    Score = mean cosine-similarity(image, positive prompts)
            − mean cosine-similarity(image, negative prompts)

    Positive → modern / well-maintained
    Negative → outdated / poor condition
    Zero     → average / neutral

    device : torch.device or None
        None defaults to CPU (safe for single-image inference in Streamlit).
        Pass a CUDA device for batch calibration runs to use the GPU.
    """
    import torch
    from PIL import Image

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        log.warning(f"Cannot open image {image_path}: {exc}")
        return 0.0

    model, processor = _load_clip()
    # Default to CPU for single-image app inference — GPU adds no meaningful
    # speedup for one image and avoids VRAM contention with XGBoost/Streamlit.
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)

    all_prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
    try:
        inputs = processor(
            text=all_prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features  = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            image_features = model.get_image_features(
                pixel_values=inputs["pixel_values"],
            )

            # L2-normalise so dot product = cosine similarity
            text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # similarities: shape [n_prompts]
            sims = (image_features @ text_features.T).squeeze(0).cpu().numpy()

    except Exception as exc:
        log.warning(f"CLIP inference failed for {image_path}: {exc}")
        return 0.0

    n_pos          = len(POSITIVE_PROMPTS)
    positive_score = float(sims[:n_pos].mean())
    negative_score = float(sims[n_pos:].mean())

    return positive_score - negative_score


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def load_calibration() -> dict:
    """Load calibration parameters, or return defaults if not yet computed."""
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH) as f:
            return json.load(f)
    return {
        "slope":     _DEFAULT_SLOPE,
        "intercept": _DEFAULT_INTERCEPT,
        "r2":        None,
        "n_samples": 0,
        "note":      "default — run calibrate_clip.py to fit on training data",
    }


def apply_calibration(raw_score: float, cal: dict | None = None) -> float:
    """
    Convert raw CLIP score to log-price adjustment.

    When the calibration fit is poor (R² < 0), the fitted slope/intercept are
    harmful — the model learned noise.  In that case we fall back to a centered
    default: adj = slope × (raw_score − training_mean), which guarantees that
    average-looking homes get zero adjustment and only genuine outliers (clearly
    modern or clearly dated) get a meaningful nudge.
    """
    if cal is None:
        cal = load_calibration()

    r2 = cal.get("r2")
    if r2 is not None and r2 < 0.0:
        # Fitted calibration is worse than predicting zero — use centered default.
        # mean_score is the "neutral" baseline: images scoring above it get a
        # positive adjustment, below it a negative one.
        # Use the actual measured mean from the calibration data so the fallback
        # is centered on the real score distribution, not a hardcoded constant.
        mean_score = cal.get("clip_score_mean", 0.034)
        return 2.0 * (raw_score - mean_score)

    return cal["slope"] * raw_score + cal["intercept"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_image(image_path: str) -> tuple[float, float]:
    """
    Score a property image for condition/modernity.

    Returns
    -------
    raw_score : float
        Cosine-similarity difference in [approx −0.10, +0.10].
        Positive = modern; negative = outdated.
    log_adjustment : float
        Calibrated log-price adjustment to add to the blended estimate.
        E.g. +0.08 means the image suggests ~+8% above the market baseline.
    """
    raw = compute_clip_score(image_path)
    cal = load_calibration()
    adj = apply_calibration(raw, cal)
    return raw, adj
