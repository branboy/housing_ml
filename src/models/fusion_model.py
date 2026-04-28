import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.utils.structured_predict import predict_structured_from_row


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(fusion_path, image_feat_path):
    df_struct = pd.read_csv(fusion_path)
    df_img = pd.read_csv(image_feat_path)

    df = pd.merge(df_struct, df_img, on="image_id")

    return df


# -----------------------------
# LOAD STRUCTURED MODEL
# -----------------------------
def load_structured_model(path):
    return joblib.load(path)


# -----------------------------
# ADD STRUCTURED PREDICTIONS
# -----------------------------
def add_structured_predictions(df, structured_pipeline):
    df = df.copy()
    df["structured_pred"] = predict_structured_from_row(df, structured_pipeline)
    return df



# -----------------------------
# CREATE RESIDUAL TARGET
# -----------------------------
RESIDUAL_CLIP = 1.0
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


# Named PCA feature columns produced by train_cnn.py
PCA_COLS = [f"pca_{i}" for i in range(200)]

# Structured context features passed alongside the image features
STRUCT_CONTEXT_COLS = ["bed", "bath", "sqft", "structured_pred"]

# -----------------------------
# PREPARE DATA
# -----------------------------
def prepare_data(df):
    """
    Build the feature matrix for the image adjustment model.

    Feature set:
      - 200 PCA image components (pca_0 … pca_199)
      - Structural context: bed, bath, sqft, structured_pred

    City/location columns are intentionally excluded.  Location context is
    already encoded implicitly via structured_pred (which uses city + zip
    target encoding).  Including CA city dummies would bake in California-
    specific market patterns and break generalization to other states.
    """
    df = df.copy()

    # Target
    y = df["residual"]

    # Select only the feature columns we care about
    keep = [c for c in PCA_COLS + STRUCT_CONTEXT_COLS if c in df.columns]
    X = df[keep]

    return X, y

# -----------------------------
# TRAIN
# -----------------------------
def train_model(X_train, y_train, sample_weight=None):
    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_weight=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


# -----------------------------
# EVALUATE
# -----------------------------
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    return {
        "rmse": rmse,
        "mae": mae
    }