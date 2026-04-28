import os
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
load_dotenv()  # load RENTCAST_API_KEY (and others) from .env before any import that reads them

from ..utils.structured_predict import predict_structured_from_row
from ..utils.rentcast_client import enrich_property, get_avm_estimate, get_market_stats
from ..utils.zillow_client import get_zillow_data


# ------------------------------------
# LOAD MODELS (once at import)
# ------------------------------------
structured_pipeline = joblib.load("outputs/models/structured_pipeline.pkl")

# ------------------------------------
# CONSTANTS
# ------------------------------------

# States where the image adjustment model is valid.
# The fusion/CNN model was trained exclusively on California homes (Dataset C).
# Applying it to other states produces nonsensical adjustments — it must be skipped.
CA_LABELS = {"california", "ca"}

# AVM blend weights — geography-aware.
# Outside CA the structured model has less training richness per zip/city,
# so RentCast AVM should carry substantially more weight.
AVM_BLEND_WEIGHTS_CA = {
    "High":    0.40,
    "Medium":  0.25,
    "Low":     0.10,
    "Unknown": 0.20,
}
AVM_BLEND_WEIGHTS_OTHER = {
    "High":    0.60,   # AVM is more reliable than structured outside CA
    "Medium":  0.45,
    "Low":     0.25,
    "Unknown": 0.40,   # even "Unknown" confidence AVM beats a weak structured pred
}

# Log-divergence threshold for extra AVM boost (separate per geography)
AVM_DIVERGENCE_THRESHOLD_CA    = 0.50
AVM_DIVERGENCE_THRESHOLD_OTHER = 0.30   # lower bar — boost AVM sooner outside CA

# Condition adjustment ceiling.
# CLIP calibration produces smaller, more grounded adjustments than the old
# ResNet→PCA→XGBoost fusion model.  ±0.15 (≈±16%) is the hard ceiling;
# outlier mode further tightens this to ±0.08 in blend_signals context.
CONDITION_CLIP = 0.15

# Note: BIAS_CORRECTION was computed during fusion training on the feature-sparse
# Dataset C (CA homes without zip/state). It corrects for that training-time gap
# and is NOT applied at inference, where users supply full feature inputs.
BIAS_CORRECTION = structured_pipeline.get("bias_correction", 0.0)


def _is_california(input_dict):
    """Return True if the property is in California."""
    state = str(input_dict.get("state", "")).strip().lower()
    return state in CA_LABELS


# ------------------------------------
# STEP 1: STRUCTURED PREDICTION
# ------------------------------------
def predict_structured(input_dict):
    """
    Primary price prediction from structured features.
    This is the main driver — location, size, and market data.
    """
    df = pd.DataFrame([input_dict])
    return float(predict_structured_from_row(df, structured_pipeline)[0])


# ------------------------------------
# STEP 2: MULTI-SIGNAL BLEND
# ------------------------------------

# AVM weights when structured model is identified as an outlier.
# External signals (AVM + market) agree → structured is clearly wrong →
# reduce its influence to near-zero.
AVM_OUTLIER_WEIGHTS = {
    "High":    0.65,
    "Medium":  0.58,
    "Low":     0.48,
    "Unknown": 0.53,
}

def blend_signals(structured_log, input_dict):
    """
    Three-signal blend: structured model + RentCast AVM + zip-level market stats.

    The market price-per-sqft estimate (ppsf × sqft) is the most property-specific
    market signal and is used preferentially over the raw zip median.

    Outlier detection: when AVM and market stats agree with each other but both
    diverge from the structured model by more than the outlier threshold, the
    structured model is identified as an outlier and its weight drops to 5%.
    This handles premium markets where the structured model under-trains
    (e.g. a $930k Maryland home predicted at $307k by the structured model).

    Returns:
        blended_log (float)
        info (dict) with keys: avm_price, avm_confidence, log_divergence,
                               ppsf_estimate, market_price, structured_outlier
    """
    is_ca    = _is_california(input_dict)
    address  = input_dict.get("address")
    city     = input_dict.get("city")
    state    = input_dict.get("state")
    zip_code = input_dict.get("zip_code")
    sqft     = input_dict.get("sqft")

    # ── Signal A: RentCast AVM ────────────────────────────────────────────
    avm_price = avm_price_high = avm_confidence = avm_log = None
    if address:
        avm = get_avm_estimate(
            address=address, city=city, state=state, zip_code=zip_code,
            bedrooms=input_dict.get("bed"), bathrooms=input_dict.get("bath"), sqft=sqft,
        )
        if avm and avm.get("price"):
            avm_price      = float(avm["price"])
            avm_confidence = avm.get("confidence", "Unknown")
            avm_log        = np.log1p(avm_price)
            if avm.get("price_high") and float(avm["price_high"]) > avm_price:
                avm_price_high = float(avm["price_high"])

    # ── Signal B: Zip market stats (ppsf × sqft preferred over raw median) ──
    market_price = ppsf_estimate = market_log = ppsf_log = None
    if zip_code:
        try:
            stats = get_market_stats(zip_code, state=state)
            if stats:
                mp   = stats.get("median_sale_price")
                ppsf = stats.get("price_per_sqft_market")
                if mp and float(mp) > 0:
                    market_price = float(mp)
                    market_log   = np.log1p(market_price)
                if ppsf and float(ppsf) > 0 and sqft and float(sqft) > 0:
                    ppsf_estimate = float(ppsf) * float(sqft)
                    ppsf_log      = np.log1p(ppsf_estimate)
        except Exception:
            pass

    # Property-specific market anchor (ppsf×sqft beats raw median)
    market_anchor_log   = ppsf_log   or market_log
    market_anchor_price = ppsf_estimate or market_price

    # ── Signal C: Zillow Zestimate ────────────────────────────────────────
    # Property-specific estimate from Zillow's full ML pipeline.
    # Substantially more accurate than RentCast in premium markets because
    # Zillow has far denser comparable-sale data, especially for high-value
    # homes.  Only available when both address and zip_code are provided.
    zestimate = zestimate_log = school_rating = None
    if address and zip_code:
        zillow = get_zillow_data(
            address=address, city=city, state=state, zip_code=zip_code
        )
        if zillow and zillow.get("zestimate"):
            zestimate     = float(zillow["zestimate"])
            zestimate_log = np.log1p(zestimate)
            school_rating = zillow.get("school_rating")

    log_divergence = abs(avm_log - structured_log) if avm_log else None

    # ── Outlier detection ─────────────────────────────────────────────────
    # Requires both AVM and market anchor. If they agree with each other but
    # both diverge substantially from structured → structured is the outlier.
    # When Zestimate is also available it is folded into the external average,
    # making the outlier decision more robust.
    two_signals = (avm_log is not None) and (market_anchor_log is not None)
    structured_outlier = False
    struct_vs_external = None

    if two_signals:
        signal_agreement = abs(avm_log - market_anchor_log)

        # External average: include Zestimate when present
        if zestimate_log is not None:
            avg_external = (avm_log + market_anchor_log + zestimate_log) / 3.0
        else:
            avg_external = (avm_log + market_anchor_log) / 2.0

        struct_vs_external = abs(structured_log - avg_external)
        outlier_threshold  = 0.45 if is_ca else 0.42
        structured_outlier = (signal_agreement < 0.30 and
                              struct_vs_external > outlier_threshold)

    # ── Compute blend ─────────────────────────────────────────────────────
    if structured_outlier:
        # AVM and market agree; structured is clearly wrong.
        w_struct = 0.05

        if zestimate_log is not None:
            # ── Zestimate available: it becomes the primary anchor ────────
            # Zillow's Zestimate is a direct property-level estimate from a
            # model with vastly more training data than RentCast, making it
            # far more reliable in premium/thin-comp markets.
            # Weights: struct 5 % / Zestimate 65 % / AVM 15 % / market 15 %
            w_zest   = 0.65
            w_avm    = 0.15
            w_market = 1.0 - w_struct - w_zest - w_avm   # 0.15

            # AVM high-end blend when Unknown confidence (same logic as before)
            effective_avm_log = avm_log
            if (avm_confidence == "Unknown" and avm_price_high is not None
                    and avm_price_high > avm_price * 1.05):
                effective_avm_log = np.log1p(0.25 * avm_price + 0.75 * avm_price_high)

            blended_log = (w_struct  * structured_log
                           + w_zest  * zestimate_log
                           + w_avm   * effective_avm_log
                           + w_market * market_anchor_log)

        else:
            # ── No Zestimate: AVM + market blend (original logic) ─────────
            w_avm    = AVM_OUTLIER_WEIGHTS.get(avm_confidence, 0.53)
            w_market = 1.0 - w_struct - w_avm

            effective_avm_log = avm_log
            if (avm_confidence == "Unknown" and avm_price_high is not None
                    and avm_price_high > avm_price * 1.05):
                effective_avm_log = np.log1p(0.25 * avm_price + 0.75 * avm_price_high)

            blended_log = (w_struct * structured_log
                           + w_avm   * effective_avm_log
                           + w_market * market_anchor_log)

        # ── Premium market uplift ─────────────────────────────────────────
        # When both structured and external signals undervalue a premium
        # non-CA market, apply a graduated upward boost proportional to the
        # divergence.  Threshold matches the outlier detection threshold (0.42)
        # so any confirmed outlier gets at least a small boost.
        # With Zestimate the boost is smaller because Zestimate already captures
        # much of the premium.
        if not is_ca and struct_vs_external is not None and struct_vs_external > 0.42:
            # scale controls how aggressively divergence translates to boost;
            # cap is the absolute ceiling in log units (0.30 ≈ +35% price room).
            # Without Zestimate the scale is higher because the blended base
            # has already pulled down toward the (potentially low) AVM, so the
            # boost needs more punch to compensate.
            scale        = 0.50 if zestimate_log is not None else 2.50
            raw_boost    = (struct_vs_external - 0.42) * scale
            premium_boost = min(0.20 if zestimate_log is not None else 0.30,
                                raw_boost)
            blended_log  += premium_boost

    elif avm_log is not None:
        # Standard geography-aware AVM blend.
        weights       = AVM_BLEND_WEIGHTS_CA    if is_ca else AVM_BLEND_WEIGHTS_OTHER
        div_threshold = AVM_DIVERGENCE_THRESHOLD_CA if is_ca else AVM_DIVERGENCE_THRESHOLD_OTHER

        avm_weight = weights.get(avm_confidence, 0.20)
        if log_divergence and log_divergence > div_threshold:
            extra      = min(0.40, (log_divergence - div_threshold) * 0.50)
            avm_weight = min(0.75, avm_weight + extra)

        # When market corroborates AVM (within 0.30 log units), carve it a
        # small share — it provides an independent local-market cross-check.
        if market_anchor_log and abs(avm_log - market_anchor_log) < 0.30:
            market_w   = min(0.20, avm_weight * 0.25)   # up to 25% of AVM's share
            avm_weight = max(avm_weight - market_w, 0.10)
            blended_log = ((1.0 - avm_weight - market_w) * structured_log
                           + avm_weight    * avm_log
                           + market_w      * market_anchor_log)
        else:
            blended_log = (1.0 - avm_weight) * structured_log + avm_weight * avm_log

    elif market_anchor_log is not None:
        # No address/AVM — use market stats alone as a soft anchor.
        struct_vs_market = abs(structured_log - market_anchor_log)
        market_weight    = min(0.40, 0.15 + struct_vs_market * 0.30)
        blended_log      = (1.0 - market_weight) * structured_log + market_weight * market_anchor_log

    else:
        blended_log = structured_log  # no external signals

    return blended_log, {
        "avm_price":           avm_price,
        "avm_price_high":      avm_price_high,
        "avm_confidence":      avm_confidence,
        "log_divergence":      log_divergence,
        "ppsf_estimate":       ppsf_estimate,
        "market_price":        market_price,
        "market_anchor_price": market_anchor_price,
        "structured_outlier":  structured_outlier,
        "struct_vs_external":  struct_vs_external,
        "zestimate":           zestimate,
        "school_rating":       school_rating,
    }


# ------------------------------------
# STEP 3: IMAGE CONDITION ADJUSTMENT
# ------------------------------------
def predict_condition_adjustment(image_paths: list[str],
                                 input_dict,
                                 base_pred_log) -> tuple[float, list[tuple[str, float, float]]]:
    """
    Use CLIP to score home condition/modernity across one or more photos.

    Each photo is scored independently; the calibrated log-price adjustments
    are averaged (equal weight) to produce the final adjustment.

    Returns
    -------
    avg_log_adj : float
        Mean calibrated log-price adjustment across all photos.
    per_image   : list of (filename, raw_score, log_adj)
        Per-photo details for the breakdown log.
    """
    from ..models.clip_condition import score_image
    from pathlib import Path

    per_image = []
    for path in image_paths:
        raw_score, log_adj = score_image(path)
        filename = Path(path).name
        per_image.append((filename, raw_score, log_adj))

    if not per_image:
        return 0.0, []

    avg_adj = float(np.mean([adj for _, _, adj in per_image]))
    return avg_adj, per_image


# ------------------------------------
# FINAL PREDICTION
# ------------------------------------
def predict_price(input_dict, image_paths=None):
    """
    Full prediction pipeline.

    Parameters
    ----------
    input_dict   : dict   — structured property features
    image_paths  : list[str] | str | None
        One or more local image paths.  A bare string is treated as a single-
        item list for backwards compatibility.

    Returns (final_price: float, breakdown: list[str]).
    The breakdown list contains human-readable lines describing every signal
    that contributed to the estimate — suitable for display in a UI or terminal.

    Callers that only need the price can ignore the second return value:
        price, _ = predict_price(input_dict)
    """
    # Normalise to list
    if image_paths is None:
        image_paths = []
    elif isinstance(image_paths, str):
        image_paths = [image_paths]
    log_lines = []

    def _log(msg=""):
        log_lines.append(msg)

    is_ca = _is_california(input_dict)

    # ── Step 1: Structured prediction ────────────────────────────────────
    structured_log = predict_structured(input_dict)

    # ── Step 2: Multi-signal blend ────────────────────────────────────────
    blended_log, blend_info = blend_signals(structured_log, input_dict)

    avm_price           = blend_info["avm_price"]
    avm_price_high      = blend_info["avm_price_high"]
    avm_confidence      = blend_info["avm_confidence"]
    log_divergence      = blend_info["log_divergence"]
    ppsf_estimate       = blend_info["ppsf_estimate"]
    market_price        = blend_info["market_price"]
    market_anchor_price = blend_info["market_anchor_price"]
    structured_outlier  = blend_info["structured_outlier"]
    struct_vs_external  = blend_info.get("struct_vs_external")
    zestimate           = blend_info.get("zestimate")
    school_rating       = blend_info.get("school_rating")

    # ── Step 3: Image condition adjustment ────────────────────────────────
    condition_adj = 0.0
    per_image_scores = []

    if image_paths:
        base_context = blended_log
        if structured_outlier:
            effective_clip = 0.05   # tighter in outlier mode — blend already compensates
        elif log_divergence is not None and log_divergence > 0.5:
            effective_clip = 0.12
        else:
            effective_clip = CONDITION_CLIP

        raw_adj, per_image_scores = predict_condition_adjustment(
            image_paths, input_dict, base_context
        )
        condition_adj = float(np.clip(raw_adj, -effective_clip, effective_clip))

    final_log   = blended_log + condition_adj
    final_price = float(np.expm1(final_log))

    # ── Sanity check ──────────────────────────────────────────────────────
    MIN_PRICE, MAX_PRICE = 10_000, 50_000_000
    if not (MIN_PRICE <= final_price <= MAX_PRICE):
        _log(f"  WARNING: ${final_price:,.0f} outside plausible range — falling back to structured")
        final_price       = float(np.expm1(structured_log))
        final_log         = structured_log
        condition_adj     = 0.0
        per_image_scores  = []

    # ── Build breakdown ───────────────────────────────────────────────────
    geo_label = "California" if is_ca else "non-CA"
    _log(f"  Geography:          {geo_label}")
    _log(f"  Structured (log):   {structured_log:.4f}  →  ${np.expm1(structured_log):>12,.0f}")

    if structured_outlier:
        _log("  ⚠ Structured model flagged as outlier — AVM + market signals override")

    if avm_price:
        avm_range_str = (
            f", range up to ${avm_price_high:,.0f}"
            if avm_price_high and structured_outlier and avm_confidence == "Unknown"
            else ""
        )
        _log(f"  RentCast AVM:       {np.log1p(avm_price):.4f}  →  ${avm_price:>12,.0f}"
             f"  [{avm_confidence} confidence{avm_range_str}]")

    if zestimate:
        zest_label = "★ Zillow Zestimate"
        _log(f"  {zest_label}: {np.log1p(zestimate):.4f}  →  ${zestimate:>12,.0f}"
             f"  [property-level Zillow estimate]")
        if school_rating is not None:
            _log(f"  School rating:      {school_rating:.1f}/10  [GreatSchools avg nearby]")

    if ppsf_estimate:
        _log(f"  Market ppsf×sqft:   {np.log1p(ppsf_estimate):.4f}  →  ${ppsf_estimate:>12,.0f}"
             f"  [zip price/sqft × home sqft]")
    elif market_price:
        _log(f"  Market median:      {np.log1p(market_price):.4f}  →  ${market_price:>12,.0f}"
             f"  [zip median sale price]")

    if avm_price or market_anchor_price:
        _log(f"  Blended base (log): {blended_log:.4f}  →  ${np.expm1(blended_log):>12,.0f}")

    # Show premium boost if it fired
    if (structured_outlier and not is_ca
            and struct_vs_external is not None and struct_vs_external > 0.42):
        scale         = 0.50 if zestimate else 2.50
        cap           = 0.20 if zestimate else 0.30
        raw_boost     = (struct_vs_external - 0.42) * scale
        premium_boost = min(cap, raw_boost)
        _log(f"  Premium mkt boost:  {premium_boost:+.4f}  ({premium_boost*100:+.1f}%)"
             f"  [divergence {struct_vs_external:.3f}; Zestimate {'used' if zestimate else 'unavailable'}]")

    if image_paths:
        n = len(per_image_scores)
        for fname, raw, adj in per_image_scores:
            _log(f"  CLIP [{fname}]:      raw={raw:+.4f}  adj={adj:+.4f} ({adj*100:+.1f}%)")
        if n > 1:
            _log(f"  CLIP avg ({n} photos): {condition_adj:+.4f}  ({condition_adj*100:+.1f}%)")
        else:
            _log(f"  CLIP condition adj: {condition_adj:+.4f}  ({condition_adj*100:+.1f}%)")

    _log("  " + "─" * 53)
    _log(f"  Final price:        {final_log:.4f}  →  ${final_price:>12,.0f}")

    return final_price, log_lines