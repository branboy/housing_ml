"""
train_structured.py — Train the primary XGBoost price prediction model.

Key improvements over the original:

1. Premium oversampling
   Homes priced above $600k represent ~20% of training data but are the segment
   the model most needs to get right (they contribute the largest absolute errors
   and are the cases where the AVM/market blend must compensate hardest).
   We oversample premium homes 4x before encoding so the TargetEncoder and
   XGBoost both see adequate representation of high-value markets.

2. Tiered sample weights
   Even after oversampling, very expensive homes ($1M+) get an additional
   weight boost during XGBoost fitting, so tree splits preferentially learn
   the premium tail rather than optimising purely for the dense $150–400k bulk.

3. Updated hyperparameters (see structured_model.py)
   min_child_weight raised from 3 → 15 to prevent the model from memorising
   narrow sqft/bed patterns that happen to dominate cheap homes.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.structured_model import (
    evaluate_models,
    prepare_data,
    select_best_model,
    split_data,
    train_models,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ZILLOW_DATA_PATH = "data/processed/structured_clean.csv"
KAGGLE_DATA_PATH = "data/processed/structured_b_clean.csv"
MODEL_PATH       = "outputs/models"
LOG_PATH         = "outputs/logs"

# ── Premium oversampling threshold ────────────────────────────────────────────
# Homes above this price are underrepresented relative to their importance.
PREMIUM_THRESHOLD    = 600_000   # $600k
PREMIUM_MULTIPLIER   = 4         # repeat premium homes 4x
ULTRAPREMIUM_THRESHOLD = 1_000_000  # $1M+ get extra sample weight (not just oversampling)


def ensure_dirs():
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)


def oversample_premium_encoded(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Repeat premium-priced homes in the ALREADY-ENCODED feature matrix.

    Critically, oversampling must happen AFTER target encoding.  If done before,
    the TargetEncoder computes city/zip means on an artificially premium-heavy
    distribution — Charlotte 28202 ($385k true median) ends up encoded at ~$800k
    because the inflated global mean contaminates the Bayesian smoothing step.

    Oversampling encoded rows instead:
    - City/zip/state encodings are computed from the true price distribution ✓
    - XGBoost simply sees more premium examples during tree construction ✓
    - Mid-range markets (Charlotte, Denver) keep their correct encodings ✓
    """
    premium_mask = y.values > np.log1p(PREMIUM_THRESHOLD)
    n_premium    = premium_mask.sum()

    if n_premium == 0:
        print("  No premium homes found — skipping oversample.")
        return X, y

    n_standard  = (~premium_mask).sum()

    # Cap so premium rows stay ≤ 40% of total
    max_extra   = int(n_standard * 0.4 / 0.6) - n_premium
    n_extra     = min(n_premium * (PREMIUM_MULTIPLIER - 1), max(max_extra, 0))

    premium_idx  = np.where(premium_mask)[0]
    extra_idx    = np.random.default_rng(42).choice(premium_idx, size=n_extra, replace=True)
    all_idx      = np.concatenate([np.arange(len(y)), extra_idx])
    np.random.default_rng(42).shuffle(all_idx)

    X_os = X.iloc[all_idx].reset_index(drop=True)
    y_os = y.iloc[all_idx].reset_index(drop=True)

    total       = len(y_os)
    prem_pct    = (n_premium + n_extra) / total * 100
    print(f"  Standard homes      : {n_standard:>10,}  (${PREMIUM_THRESHOLD/1e3:.0f}k and below)")
    print(f"  Premium (+{n_extra:,} extra): {n_premium + n_extra:>10,}  "
          f"({PREMIUM_MULTIPLIER}x of {n_premium:,} unique)")
    print(f"  Total after oversample: {total:>8,}  ({prem_pct:.1f}% premium)")
    return X_os, y_os


def make_sample_weights(y_train: pd.Series, price_threshold_log: float,
                        X_train: pd.DataFrame = None) -> np.ndarray:
    """
    Build per-sample weights for XGBoost fitting.

    Weight schedule:
      price < $600k              →  1.0  (baseline)
      $600k – $1M                →  1.5  (moderate premium boost)
      $1M+                       →  2.5  (strong premium boost)
      scraped Zillow sold_price  →  extra 3× multiplier on top

    The Zillow multiplier compensates for the 546:1 dilution of scraped premium
    homes by the Kaggle bulk — 84 MD scraped homes need outsized influence to
    shift zip-level target encoders away from Kaggle's stale 2020–2022 medians.
    """
    prices = np.expm1(y_train.values)
    weights = np.ones(len(prices))
    weights[prices >  PREMIUM_THRESHOLD]       = 1.5
    weights[prices >  ULTRAPREMIUM_THRESHOLD]  = 2.5

    # Boost scraped Zillow homes (actual sold prices, current market data)
    if X_train is not None and "data_source" in X_train.columns:
        zillow_mask = X_train["data_source"] == "zillow_api"
        # Only boost actual sold prices, not Zestimate-derived prices
        if "price_source" in X_train.columns:
            zillow_mask = zillow_mask & (X_train["price_source"] == "sold_price")
        weights[zillow_mask.values] *= 3.0
        print(f"  Zillow sold-price rows boosted 3×: {zillow_mask.sum():,}")

    return weights


def main():
    ensure_dirs()

    # ── Load data ─────────────────────────────────────────────────────────────
    if os.path.exists(ZILLOW_DATA_PATH):
        DATA_PATH = ZILLOW_DATA_PATH
        print(f"Using merged dataset: {ZILLOW_DATA_PATH}")
    else:
        DATA_PATH = KAGGLE_DATA_PATH
        print(f"Zillow data not found — using Kaggle only: {KAGGLE_DATA_PATH}")

    print("Loading data...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  Loaded {len(df):,} rows  |  price median ${df.price.median():,.0f}")

    # Drop rows with missing or implausible prices — these cause inf MAPE
    # and confuse the target encoder. $10k floor removes data-entry errors;
    # $50M ceiling removes extreme outliers that would dominate the loss.
    before = len(df)
    df = df[(df["price"] >= 10_000) & (df["price"] <= 50_000_000)].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with price outside [$10k, $50M]")

    # ── Feature engineering + encoding ───────────────────────────────────────
    print("\nPreparing features...")
    # Preserve source metadata BEFORE prepare_data() drops it — needed for weights
    source_meta = df[["data_source", "price_source"]].copy() \
        if "data_source" in df.columns else None
    X, y, encoders = prepare_data(df, training=True)

    # ── Train / val / test splits ─────────────────────────────────────────────
    print("Splitting data (80% train / 10% val / 10% test)...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_val, y_train, y_val   = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # Attach source metadata to X_train so make_sample_weights can use it.
    # Use .loc (label-based) not .iloc (positional) — after the price filter
    # df has non-contiguous row labels so .iloc would be out-of-bounds.
    if source_meta is not None:
        meta_train = source_meta.loc[X_train.index].reset_index(drop=True)
        X_train_meta = X_train.reset_index(drop=True).join(meta_train)
    else:
        X_train_meta = X_train

    # ── Sample weights ────────────────────────────────────────────────────────
    print("\nComputing sample weights for premium homes...")
    sample_weights = make_sample_weights(y_train, np.log1p(PREMIUM_THRESHOLD), X_train_meta)
    premium_train  = (np.expm1(y_train.values) > PREMIUM_THRESHOLD).sum()
    print(f"  Premium homes in train set: {premium_train:,} / {len(y_train):,} "
          f"({100*premium_train/len(y_train):.1f}%)")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nTraining XGBoost...")
    models = train_models(X_train, y_train, X_val, y_val,
                          sample_weight=sample_weights)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating on held-out test set...")
    results = evaluate_models(models, X_test, y_test)
    print(json.dumps(results, indent=2))

    # Premium-specific evaluation
    test_prices = np.expm1(y_test.values)
    # Filter out any rows with price <= 0 to avoid divide-by-zero in MAPE
    valid_mask   = test_prices > 0
    premium_mask = (test_prices > PREMIUM_THRESHOLD) & valid_mask
    if premium_mask.sum() > 0:
        best_name_tmp = min(results, key=lambda k: results[k]["rmse"])
        preds_test    = models[best_name_tmp].predict(X_test)
        premium_mape  = np.mean(np.abs(
            np.expm1(preds_test[premium_mask]) - test_prices[premium_mask]
        ) / test_prices[premium_mask]) * 100
        raw_overall   = np.abs(
            np.expm1(preds_test[valid_mask]) - test_prices[valid_mask]
        ) / test_prices[valid_mask]
        overall_mape  = np.nanmean(raw_overall[np.isfinite(raw_overall)]) * 100
        ultra_mask = (test_prices > ULTRAPREMIUM_THRESHOLD) & valid_mask
        ultra_mape = np.mean(np.abs(
            np.expm1(preds_test[ultra_mask]) - test_prices[ultra_mask]
        ) / test_prices[ultra_mask]) * 100 if ultra_mask.sum() > 0 else float("nan")
        print(f"\nPremium market accuracy:")
        print(f"  MAPE overall          : {overall_mape:.1f}%  ({valid_mask.sum():,} homes)")
        print(f"  MAPE >$600k           : {premium_mape:.1f}%  ({premium_mask.sum():,} homes)")
        print(f"  MAPE >$1M (ultra)     : {ultra_mape:.1f}%  ({ultra_mask.sum():,} homes)")

    # ── Save ──────────────────────────────────────────────────────────────────
    best_name, best_model = select_best_model(models, results)
    print(f"\nBest model: {best_name}")

    pipeline = {
        "model":               best_model,
        "encoders":            encoders,
        "feature_names":       list(X_train.columns),
        "sqft_median":         encoders["sqft_median"],
        "premium_threshold":   PREMIUM_THRESHOLD,
        "training_data_path":  DATA_PATH,
    }
    joblib.dump(pipeline, f"{MODEL_PATH}/structured_pipeline.pkl")

    results["premium_mape"] = float(premium_mape) if premium_mask.sum() > 0 else None
    with open(f"{LOG_PATH}/structured_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Training complete!")
    print(f"   Model saved  → {MODEL_PATH}/structured_pipeline.pkl")
    print(f"   Metrics saved → {LOG_PATH}/structured_metrics.json")


if __name__ == "__main__":
    main()
