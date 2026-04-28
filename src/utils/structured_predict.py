import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from ..models.structured_model import prepare_data

# Silence XGBoost's C++ logger for the whole process.
# The "Falling back to prediction using DMatrix due to mismatched devices" line
# comes from the C++ runtime and bypasses Python's warnings system entirely —
# filterwarnings() has no effect on it.  verbosity=0 suppresses it at source.
xgb.set_config(verbosity=0)


def predict_structured_from_row(df, pipeline):
    df = df.copy()
    if "price" not in df.columns:
        df["price"] = 1.0

    X, _, _ = prepare_data(df, training=False, encoders=pipeline["encoders"])
    X = X.reindex(columns=pipeline["feature_names"], fill_value=0)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*device.*", category=UserWarning)
        preds = pipeline["model"].predict(X)

    return preds