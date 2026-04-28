# Housing Price Model — Full Diagnostic & Fix Guide

---

## Root Cause Summary

After reading your source code, there are **five concrete bugs** causing your instability, not just tuning issues. Each is documented below with the exact fix.

| # | Problem | Severity | Location |
|---|---------|----------|----------|
| 1 | Target encoding uses the full dataset label, causing leakage | Critical | `structured_model.py` |
| 2 | Fusion model re-prepares features differently than the structured model trained them | Critical | `fusion_model.py → add_structured_predictions` |
| 3 | Residuals are unbounded — fusion model is free to predict ±∞ | High | `fusion_model.py → create_residual_target` |
| 4 | PCA is fit on the full dataset before the train/test split | High | `train_cnn.py` |
| 5 | `is_large_house` uses a hardcoded `0` at inference instead of the training median | Medium | `predict.py → predict_structured` |

---

## Issue 1: Target Encoding Leakage (Critical)

### The Problem

In `structured_model.py`, `prepare_data()` computes the city mean like this:

```python
city_mean = df.groupby("city")["log_price"].mean()
```

This uses the **entire dataset** — including validation and test rows — to compute the encoding. When a test row's own label contributes to its city mean, the model effectively "sees" the answer during training. This inflates training performance, makes encodings shift between folds, and causes the model to overfit encoding values that don't generalize.

When you removed this encoding for inference but kept the model trained with it, the structured model is now predicting features it was never actually taught to work without. This is why predictions swung from ~10.4 to ~13.49.

### The Fix

Use **Leave-One-Out encoding** with Gaussian noise, computed strictly within the training fold. This is the correct, leakage-free version of target encoding.

**Install:**
```bash
pip install category_encoders
```

**Replace the encoding block in `structured_model.py`:**

```python
from category_encoders import LeaveOneOutEncoder

def prepare_data(df, training=True, encoders=None):
    df = df.copy()
    df["log_price"] = np.log1p(df["price"])
    y = df["log_price"]

    # Feature engineering
    df["bed_bath_ratio"]        = df["bed"] / (df["bath"] + 1)
    df["sqft_per_bed"]          = df["sqft"] / (df["bed"] + 1)
    df["bath_per_bed"]          = df["bath"] / (df["bed"] + 1)
    df["price_per_sqft_proxy"]  = df["sqft"] / (df["bed"] + df["bath"] + 1)
    df["sqft_squared"]          = df["sqft"] ** 2
    df["is_large_house"]        = (df["sqft"] > df["sqft"].quantile(0.75)).astype(int)

    cat_cols = [c for c in ["city", "zip_code"] if c in df.columns]

    if training:
        # sigma=0.05 adds small noise to prevent overfitting the encoding
        loo = LeaveOneOutEncoder(cols=cat_cols, sigma=0.05, random_state=42)
        df[cat_cols] = loo.fit_transform(df[cat_cols], y)
        encoders = {"loo": loo}
    else:
        loo = encoders["loo"]
        df[cat_cols] = loo.transform(df[cat_cols])

    # State: use OrdinalEncoder (safe for unseen states at inference)
    from sklearn.preprocessing import OrdinalEncoder
    if training:
        state_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df["state"] = state_enc.fit_transform(df[["state"]])
        encoders["state_enc"] = state_enc
    else:
        df["state"] = encoders["state_enc"].transform(df[["state"]])

    drop_cols = ["price", "log_price"]
    X = df.drop(columns=drop_cols)
    X = X.fillna(X.median(numeric_only=True))

    return X, y, encoders
```

**Critical**: Once you fix this, **retrain the structured model completely from scratch**, then retrain the fusion model using the new structured predictions. The two must always be trained as a matched pair.

---

## Issue 2: Fusion Model Uses Wrong Feature Preparation (Critical)

### The Problem

In `fusion_model.py`, `add_structured_predictions()` re-creates features manually:

```python
# BAD — mismatches the structured model's training features
X = df[["bed", "bath", "sqft", "city"]]
X = pd.get_dummies(X, columns=["city"], drop_first=True)
model_features = structured_model.feature_names_in_
X = X.reindex(columns=model_features, fill_value=0)
```

This is the **single biggest source of your large adjustments**. The structured model was trained with LOO-encoded city values (floats), engineered ratios, and an ordinal state. When fusion feeds it raw one-hot city dummies, the model runs on completely alien features and outputs garbage predictions like 13.49. The fusion model then has to learn a correction of +2.4 just to compensate for the structured model running on wrong inputs.

### The Fix

**Save the full preprocessing pipeline alongside the model**, and reuse it everywhere — including inside `add_structured_predictions`.

**In `train_structured.py`:**
```python
import joblib

# After training
pipeline = {
    "model": best_model,
    "encoders": encoders,
    "feature_names": list(X_train.columns)
}
joblib.dump(pipeline, "outputs/models/structured_pipeline.pkl")
```

**New shared utility — `src/utils/structured_predict.py`:**
```python
import pandas as pd
import numpy as np

def predict_structured_from_row(df, pipeline):
    """
    Reusable function for making structured predictions.
    Uses the same preprocessing as training. Call this from both
    fusion training and inference.
    """
    from src.models.structured_model import prepare_data

    df = df.copy()
    # Add a dummy price column so prepare_data doesn't crash
    if "price" not in df.columns:
        df["price"] = 1.0

    X, _, _ = prepare_data(df, training=False, encoders=pipeline["encoders"])

    # Align to exact training columns
    X = X.reindex(columns=pipeline["feature_names"], fill_value=0)

    return pipeline["model"].predict(X)
```

**Use this in `fusion_model.py → add_structured_predictions`:**
```python
from src.utils.structured_predict import predict_structured_from_row

def add_structured_predictions(df, structured_pipeline):
    df = df.copy()
    df["structured_pred"] = predict_structured_from_row(df, structured_pipeline)
    return df
```

**And use the same function in `predict.py → predict_structured`:**
```python
structured_pipeline = joblib.load("outputs/models/structured_pipeline.pkl")

def predict_structured(input_dict):
    df = pd.DataFrame([input_dict])
    preds = predict_structured_from_row(df, structured_pipeline)
    return preds[0]
```

---

## Issue 3: Unbounded Residuals (High Severity)

### The Problem

```python
def create_residual_target(df):
    df["residual"] = df["log_price"] - df["structured_pred"]
    return df
```

There is no bound on what residuals can be. If the structured model mispredicts by 2 log units (e.g., due to the feature mismatch above), the fusion model trains on targets of ±2–3. It then predicts those values at inference, producing catastrophically wrong adjustments.

Even after fixing Issues 1 and 2, you should still clip residuals to protect against outliers.

### The Fix

```python
RESIDUAL_CLIP = 0.75  # ~±111% price error in raw space — very generous

def create_residual_target(df):
    df = df.copy()
    df["log_price"] = np.log1p(df["price"])
    raw_residual = df["log_price"] - df["structured_pred"]

    # Clip to prevent outlier-driven instability
    df["residual"] = raw_residual.clip(-RESIDUAL_CLIP, RESIDUAL_CLIP)

    # Log how often clipping triggers
    n_clipped = (raw_residual.abs() > RESIDUAL_CLIP).sum()
    pct = 100 * n_clipped / len(df)
    print(f"Residual clipping: {n_clipped} rows ({pct:.1f}%) clipped beyond ±{RESIDUAL_CLIP}")

    return df
```

Also clip at inference time as a safety net:

```python
# In predict.py
RESIDUAL_CLIP = 0.75

adjustment = adjustment_model.predict(feature_df)[0]
adjustment = float(np.clip(adjustment, -RESIDUAL_CLIP, RESIDUAL_CLIP))  # safety net
```

---

## Issue 4: PCA Fit on Full Dataset Before Split (High Severity)

### The Problem

In `train_cnn.py`:

```python
pca = PCA(n_components=100)
X_reduced = pca.fit_transform(X)  # X is the FULL dataset including test rows
```

The PCA is fit on all image features including those in the test set. This is a form of data leakage — the PCA learns the principal components of the test distribution, which wouldn't be available in real deployment.

### The Fix

Split first, then fit PCA on training rows only:

```python
from sklearn.model_selection import train_test_split

# After extracting all features...
feature_df = pd.DataFrame(all_features)
feature_df["image_id"] = all_image_ids

# Merge with metadata to get a train/test split that matches your structured model split
df_meta = pd.read_csv(DATA_PATH)
df_full = feature_df.merge(df_meta[["image_id"]], on="image_id")

train_ids, test_ids = train_test_split(df_full["image_id"], test_size=0.2, random_state=42)

train_mask = feature_df["image_id"].isin(train_ids)
X_train_img = feature_df.loc[train_mask].drop(columns=["image_id"])
X_test_img  = feature_df.loc[~train_mask].drop(columns=["image_id"])

# Fit PCA on TRAIN only
pca = PCA(n_components=100, random_state=42)
X_train_reduced = pca.fit_transform(X_train_img)
X_test_reduced  = pca.transform(X_test_img)

print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

joblib.dump(pca, "outputs/models/pca.pkl")
```

**On PCA dimensions**: Run the explained variance check above. If `explained_variance_ratio_.sum()` is below 0.85, increase to 150. If it's above 0.98, reduce to 50–75. For ResNet50's 2048-dim output, 100 components typically captures ~90% variance, which is appropriate.

---

## Issue 5: `is_large_house` Hardcoded to 0 at Inference (Medium)

### The Problem

In `predict.py`:
```python
df["is_large_house"] = 0  # safe default
```

The structured model was trained with `is_large_house = (sqft > median_sqft)`. At inference you always set it to 0, which means houses larger than median are misclassified. This contributes a small but consistent bias.

### The Fix

Save the training sqft median and use it at inference:

```python
# In train_structured.py, save the threshold
sqft_median = float(df["sqft"].median())
pipeline["sqft_median"] = sqft_median
joblib.dump(pipeline, "outputs/models/structured_pipeline.pkl")

# In prepare_data(), use the saved threshold at inference
if training:
    threshold = df["sqft"].median()
    encoders["sqft_median"] = float(threshold)
else:
    threshold = encoders["sqft_median"]

df["is_large_house"] = (df["sqft"] > threshold).astype(int)
```

---

## XGBoost Hyperparameter Improvements

Your current structured model (`max_depth=8`) is too deep for a tabular housing dataset with ~10 features. Deep trees memorize training patterns rather than generalizing.

```python
xgb = XGBRegressor(
    n_estimators=500,
    max_depth=5,          # was 8 — reduced to prevent overfitting
    learning_rate=0.03,   # was 0.05 — slower learning is more stable
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=5,   # ADD: prevents splits on tiny leaf nodes
    reg_alpha=0.5,        # was 0.1 — more L1 regularization
    reg_lambda=2.0,       # was 1.0 — more L2 regularization
    early_stopping_rounds=30,  # ADD: stop before overfitting
    eval_metric="rmse",
    n_jobs=-1,
    random_state=42
)

# Use early stopping properly
xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)
```

For the **fusion model**, use a Random Forest instead of XGBoost. RF is naturally more conservative on out-of-distribution inputs (it averages trees rather than boosting errors), which produces smaller, more stable adjustments:

```python
from sklearn.ensemble import RandomForestRegressor

fusion_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,           # constrain depth
    min_samples_leaf=10,   # require at least 10 samples per leaf
    max_features=0.5,      # use 50% of features per split
    random_state=42,
    n_jobs=-1
)
```

---

## Architecture Recommendation: Keep Residual Learning, But Constrain It

You asked whether to switch to direct prediction. The answer is: **keep residual learning, but add a learned confidence weight**.

Direct prediction with image + structured features together would require the fusion model to learn all the structure that the structured model already captured. Residual learning is the right idea — images should provide a *refinement*, not a full re-estimation.

The missing piece is a **confidence gate**:

```python
# During training, compute image confidence weight
# Rows where structured model is more accurate get lower image weight
structured_errors = np.abs(y_train - structured_train_preds)
image_weight = np.clip(1.0 - structured_errors / structured_errors.max(), 0.1, 1.0)

# Train fusion model on weighted residuals
fusion_model.fit(X_train, y_residual_train, sample_weight=image_weight)
```

At inference, also add a simple sanity check:
```python
def predict_price(input_dict, image_path=None):
    structured_pred = predict_structured(input_dict)

    if image_path:
        adjustment = predict_adjustment(image_path, input_dict, structured_pred)
        adjustment = float(np.clip(adjustment, -0.75, 0.75))  # hard cap
    else:
        adjustment = 0.0

    final_log = structured_pred + adjustment
    final_price = np.expm1(final_log)

    # Sanity check: reject if outside plausible range
    MIN_PRICE, MAX_PRICE = 20_000, 50_000_000
    if not (MIN_PRICE <= final_price <= MAX_PRICE):
        print(f"WARNING: Final price ${final_price:,.0f} is outside plausible range. Using structured only.")
        final_price = np.expm1(structured_pred)

    return final_price
```

---

## Ordered Fix Sequence

Apply fixes in this exact order — each step depends on the previous:

1. **Fix `prepare_data` in `structured_model.py`** — implement LOO encoding, fix `is_large_house` threshold
2. **Retrain the structured model completely** — `python src/training/train_structured.py`
3. **Fix `train_cnn.py`** — split before fitting PCA
4. **Re-extract image features and refit PCA** — `python src/training/train_cnn.py`
5. **Fix `fusion_model.py`** — use shared `predict_structured_from_row`, clip residuals
6. **Retrain the fusion model** — `python src/training/train_fusion.py`
7. **Fix `predict.py`** — use shared utility, add safety clip
8. **Validate** — structured predictions should be within ±15% of median for comparables; adjustments should be within ±0.5 log units

---

## Diagnostic Script

Run this after retraining to validate the pipeline health:

```python
# save as: validate_pipeline.py
import joblib, numpy as np, pandas as pd

pipeline = joblib.load("outputs/models/structured_pipeline.pkl")
fusion   = joblib.load("outputs/models/image_adjustment_model.pkl")

df = pd.read_csv("data/processed/structured_b_clean.csv").sample(200, random_state=1)
from src.utils.structured_predict import predict_structured_from_row

preds = predict_structured_from_row(df, pipeline)
true  = np.log1p(df["price"].values)

residuals = true - preds
print(f"Structured model residuals (should be ~N(0, 0.3)):")
print(f"  Mean:   {residuals.mean():.4f}  (target: near 0.0)")
print(f"  StdDev: {residuals.std():.4f}   (target: < 0.4)")
print(f"  Max abs: {np.abs(residuals).max():.4f} (target: < 1.5)")

mape = np.mean(np.abs(np.expm1(preds) - df["price"]) / df["price"]) * 100
print(f"\nMAPE on sample: {mape:.1f}%  (target: < 15%)")

if residuals.std() > 0.5:
    print("\n⚠️  Structured model is still unstable. Check feature encoding.")
elif mape < 15:
    print("\n✅  Structured model looks healthy. Proceed to fusion training.")
```

---

## Summary Table

| What to Change | File | Expected Impact |
|---|---|---|
| LOO encoding instead of global target mean | `structured_model.py` | Eliminates leakage, stabilizes predictions |
| Save & reuse full pipeline at inference | `structured_model.py`, `predict.py` | Fixes feature mismatch causing ±2 adjustments |
| Clip residuals to ±0.75 | `fusion_model.py` | Prevents fusion model from learning to over-correct |
| Split data before fitting PCA | `train_cnn.py` | Removes test leakage in image features |
| Use saved sqft median for `is_large_house` | `prepare_data`, `predict.py` | Removes constant bias in large homes |
| Reduce XGBoost `max_depth` to 5 | `structured_model.py` | Reduces overfitting to training distribution |
| RF instead of XGBoost for fusion | `fusion_model.py` | More conservative adjustments by nature |
| Add inference sanity check | `predict.py` | Catches any remaining runaway predictions |
