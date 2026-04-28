import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import OrdinalEncoder


# -----------------------------
# PREPARE DATA
# -----------------------------
def prepare_data(df, training=True, encoders=None):
    df = df.copy()

    if training:
        encoders = {}

    # Normalize house_size → sqft
    if "house_size" in df.columns and "sqft" not in df.columns:
        df = df.rename(columns={"house_size": "sqft"})

    # Target
    df["log_price"] = np.log1p(df["price"])
    y = df["log_price"]

    # Coerce core columns to numeric
    df["sqft"] = pd.to_numeric(df["sqft"], errors="coerce")
    df["bed"]  = pd.to_numeric(df["bed"],  errors="coerce")
    df["bath"] = pd.to_numeric(df["bath"], errors="coerce")

    # -------------------------
    # STRUCTURAL FEATURES
    # -------------------------
    df["bed_bath_ratio"]       = df["bed"]  / (df["bath"] + 1)
    df["sqft_per_bed"]         = df["sqft"] / (df["bed"]  + 1)
    df["bath_per_bed"]         = df["bath"] / (df["bed"]  + 1)
    df["price_per_sqft_proxy"] = df["sqft"] / (df["bed"]  + df["bath"] + 1)
    df["sqft_squared"]         = df["sqft"] ** 2

    if training:
        threshold = df["sqft"].median()
        encoders["sqft_median"] = float(threshold)
    else:
        threshold = encoders["sqft_median"]
    df["is_large_house"] = (df["sqft"] > threshold).astype(int)

    # -------------------------
    # LOT FEATURES
    # -------------------------
    if "acre_lot" in df.columns:
        df["acre_lot"] = pd.to_numeric(df["acre_lot"], errors="coerce")
        df["log_acre_lot"]  = np.log1p(df["acre_lot"].fillna(0))
        df["is_no_lot"]     = (df["acre_lot"].isna() | (df["acre_lot"] < 0.05)).astype(int)
        df["sqft_per_acre"] = df["sqft"] / (df["acre_lot"].fillna(0.01) + 0.01)
    else:
        df["log_acre_lot"]  = 0.0
        df["is_no_lot"]     = 1
        df["sqft_per_acre"] = 0.0

    # -------------------------
    # ZILLOW RICH FEATURES
    # (present when training on Zillow data; NaN-safe for Kaggle-only rows)
    # -------------------------

    # House age from year_built
    if "house_age" in df.columns:
        df["house_age"] = pd.to_numeric(df["house_age"], errors="coerce")
    elif "year_built" in df.columns:
        df["year_built"] = pd.to_numeric(df["year_built"], errors="coerce")
        df["house_age"] = 2025 - df["year_built"]
        df["house_age"] = df["house_age"].clip(0, 200)
    else:
        df["house_age"] = np.nan

    # Decade built buckets (captures renovation-wave patterns)
    df["decade_built"] = np.where(
        df["house_age"].notna(),
        ((2025 - df["house_age"].fillna(30)) // 10 * 10).clip(1900, 2020),
        np.nan
    )

    # Property type one-hot (NaN rows get all-zeros → neutral)
    PROP_TYPES = ["single_family", "condo", "townhouse", "multi_family", "manufactured"]
    if "property_type" in df.columns:
        for pt in PROP_TYPES:
            df[f"type_{pt}"] = (df["property_type"].fillna("unknown") == pt).astype(int)
    else:
        for pt in PROP_TYPES:
            df[f"type_{pt}"] = 0

    # HOA
    for col, default in [("hoa_fee", 0), ("has_hoa", 0)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default
    df["log_hoa_fee"] = np.log1p(df["hoa_fee"])

    # Garage & amenities
    for col, default in [("garage_spaces", 0), ("has_pool", 0), ("stories", 1)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default

    # School rating (10-point GreatSchools scale)
    if "school_rating" in df.columns:
        df["school_rating"] = pd.to_numeric(df["school_rating"], errors="coerce")
        # Flag missing so model can distinguish "no data" from "rated 0"
        df["has_school_rating"] = df["school_rating"].notna().astype(int)
        df["school_rating"] = df["school_rating"].fillna(5.0)  # neutral impute
    else:
        df["school_rating"]     = 5.0
        df["has_school_rating"] = 0

    # -------------------------
    # GEOGRAPHIC ENCODING
    # -------------------------
    # City
    city_col = df[["city"]] if "city" in df.columns else pd.DataFrame({"city": ["unknown"] * len(df)})
    if training:
        te_city = TargetEncoder(target_type="continuous", smooth="auto", cv=5, random_state=42)
        df["city_encoded"] = te_city.fit_transform(city_col, y).ravel()
        encoders["city_te"] = te_city
    else:
        df["city_encoded"] = encoders["city_te"].transform(city_col).ravel()

    # Zip code
    # IMPORTANT: training data stores zip_code as float64, so astype(str) produces
    # "78702.0".  At inference callers pass "78702" (string), which astype(str) keeps
    # as "78702" — a category the encoder never saw → silently falls back to global mean.
    # Fix: coerce to numeric first so BOTH paths produce "78702.0" before encoding.
    if "zip_code" in df.columns:
        df["zip_code"] = pd.to_numeric(df["zip_code"], errors="coerce")
    zip_col = df[["zip_code"]].astype(str) if "zip_code" in df.columns \
              else pd.DataFrame({"zip_code": ["unknown"] * len(df)})
    if training:
        te_zip = TargetEncoder(target_type="continuous", smooth="auto", cv=5, random_state=41)
        df["zip_encoded"] = te_zip.fit_transform(zip_col, y).ravel()
        encoders["zip_te"] = te_zip
    else:
        df["zip_encoded"] = encoders["zip_te"].transform(zip_col).ravel()

    # State
    if "state" in df.columns:
        if training:
            state_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            df["state"] = state_enc.fit_transform(df[["state"]])
            encoders["state_enc"] = state_enc
        else:
            df["state"] = encoders["state_enc"].transform(df[["state"]])
    else:
        df["state"] = -1

    # -------------------------
    # DROP RAW COLUMNS
    # -------------------------
    drop_cols = ["price", "log_price"]
    for col in ["city", "zip_code", "street", "brokered_by", "status",
                "prev_sold_date", "house_size", "property_type",
                "year_built", "address", "zpid", "last_sold_date",
                "scraped_at", "source", "data_source",
                "latitude", "longitude", "zestimate", "price_per_sqft"]:
        if col in df.columns:
            drop_cols.append(col)

    X = df.drop(columns=drop_cols)
    X = X.select_dtypes(include=["number"])
    X = X.fillna(X.median(numeric_only=True))

    return X, y, encoders

# -----------------------------
# SPLIT
# -----------------------------
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


# -----------------------------
# TRAIN MODELS
# -----------------------------
def _detect_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"  GPU detected: {name} — using CUDA for XGBoost")
            return "cuda"
    except ImportError:
        pass
    print("  No GPU detected — training on CPU")
    return "cpu"


def train_models(X_train, y_train, X_val, y_val, sample_weight=None):
    """
    Train XGBoost on log-price with auto GPU detection.

    GPU:  XGBoost uses CUDA when available (~5-10x faster than CPU).
          ~4–6 min on a mid-range GPU vs ~30 min on CPU for 3M rows.

    Hyperparameter rationale vs original:
    - min_child_weight 3 → 15   Prevents narrow splits that over-fit cheap-home
                                 density. With 2.2M rows even min_child_weight=15
                                 gives leaves with 15+ samples.
    - max_depth 6 → 5           Shallower trees are less likely to memorise the
                                 bulk cheap-home distribution before learning
                                 premium market patterns.
    - colsample_bytree 0.8→0.7  More feature randomness per tree reduces the
                                 dominance of sqft/bed (strongly correlated with
                                 cheap homes) over city/zip encodings.
    - reg_lambda 1.5 → 2.0      Stronger L2 further prevents regression to the
                                 national mean on premium inputs.
    - n_jobs removed when using GPU (CUDA handles parallelism internally).
    """
    device = _detect_device()

    xgb_params = dict(
        n_estimators=5000,        # large budget — early stopping handles the cap
        max_depth=5,              # shallower — less cheap-home overfitting
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,     # more feature randomness → geography matters more
        min_child_weight=15,      # require 15+ samples per leaf
        reg_alpha=0.3,
        reg_lambda=2.0,           # stronger L2
        early_stopping_rounds=50,
        eval_metric="rmse",
        random_state=42,
        device=device,            # "cuda" or "cpu"  (XGBoost ≥ 2.0)
    )
    # n_jobs is only meaningful on CPU; omit when GPU is used
    if device == "cpu":
        xgb_params["n_jobs"] = -1

    xgb = XGBRegressor(**xgb_params)
    xgb.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    print(f"  Best iteration: {xgb.best_iteration}  "
          f"Val RMSE: {xgb.best_score:.4f}")
    return {"xgboost": xgb}


# -----------------------------
# EVALUATE
# -----------------------------
def evaluate_models(models, X_test, y_test):
    """
    Evaluate each model on the held-out test set.

    XGBoost emits a benign device-mismatch warning when a GPU-trained booster
    predicts on a CPU DataFrame.  The results are identical — suppress it.
    """
    import warnings
    results = {}
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*device.*", category=UserWarning)
            preds = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae  = float(mean_absolute_error(y_test, preds))
        results[name] = {"rmse": rmse, "mae": mae}

    return results


# -----------------------------
# SELECT BEST
# -----------------------------
def select_best_model(models, results):
    best_name = min(results, key=lambda x: results[x]["rmse"])
    return best_name, models[best_name]