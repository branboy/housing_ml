import os
import json
import joblib
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from sklearn.model_selection import train_test_split
from src.models.fusion_model import (
    load_data,
    load_structured_model,
    add_structured_predictions,
    create_residual_target,
    prepare_data,
    train_model,
    evaluate
)


# -----------------------------
# PATHS
# -----------------------------
FUSION_PATH = "data/processed/fusion_dataset.csv"
IMAGE_FEAT_PATH = "data/processed/image_features.csv"
STRUCTURED_MODEL_PATH = "outputs/models/structured_model.pkl"
STRUCTURED_PIPELINE_PATH = "outputs/models/structured_pipeline.pkl"

MODEL_PATH = "outputs/models"
LOG_PATH = "outputs/logs"


def ensure_dirs():
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)


def main():
    ensure_dirs()

    # Load
    df = load_data(FUSION_PATH, IMAGE_FEAT_PATH)
    # structured_model = load_structured_model(STRUCTURED_MODEL_PATH)
    structured_pipeline = joblib.load(STRUCTURED_PIPELINE_PATH)

    # Add structured predictions
    # df = add_structured_predictions(df, structured_model)
    df = add_structured_predictions(df, structured_pipeline)

    raw_residuals = np.log1p(df["price"]) - df["structured_pred"]
    bias = float(raw_residuals.median())  # 0.854
    print(f"Bias correction applied: +{bias:.4f}")
    df["structured_pred"] = df["structured_pred"] + bias

    # Save it for inference
    structured_pipeline["bias_correction"] = bias
    joblib.dump(structured_pipeline, STRUCTURED_PIPELINE_PATH)

    print(f"Raw residual mean:   {raw_residuals.mean():.3f}")   # near 0 = random error, far = systematic bias
    print(f"Raw residual std:    {raw_residuals.std():.3f}")
    print(f"Raw residual median: {raw_residuals.median():.3f}")

    # Create residual target
    df = create_residual_target(df)

    # Prepare
    X, y = prepare_data(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    structured_errors = np.abs(y_train.values - X_train["structured_pred"].values)
    image_weight = np.clip(1.0 - structured_errors / structured_errors.max(), 0.1, 1.0)
    model = train_model(X_train, y_train, sample_weight=image_weight)

    # Evaluate
    metrics = evaluate(model, X_test, y_test)

    print("Image Adjustment Model Metrics:")
    print(json.dumps(metrics, indent=2))

    # Save
    joblib.dump(model, f"{MODEL_PATH}/image_adjustment_model.pkl")

    with open(f"{LOG_PATH}/image_adjustment_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()