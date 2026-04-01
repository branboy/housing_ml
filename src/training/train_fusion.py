import os
import json
import joblib
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
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

MODEL_PATH = "outputs/models"
LOG_PATH = "outputs/logs"


def ensure_dirs():
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)


def main():
    ensure_dirs()

    # Load
    df = load_data(FUSION_PATH, IMAGE_FEAT_PATH)
    structured_model = load_structured_model(STRUCTURED_MODEL_PATH)

    # Add structured predictions
    df = add_structured_predictions(df, structured_model)

    # Create residual target
    df = create_residual_target(df)

    # Prepare
    X, y = prepare_data(df)

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = train_model(X_train, y_train)

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