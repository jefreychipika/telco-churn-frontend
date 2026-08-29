"""
predictor.py
-------------
Loads the trained scikit-learn pipeline (produced by model/train_model.py)
once at startup, and exposes a single `predict(customer)` function.

Because train_model.py saved a full Pipeline (preprocessing + model
together), predicting here is just: build a one-row dataframe with the
raw column names/values, then call pipeline.predict(). No manual
one-hot-encoding or column alignment needed - that complexity lives in
the pipeline itself, defined once during training.
"""

import joblib
import pandas as pd

from config import settings
from schemas import CustomerData

# Loaded once when the module is first imported (i.e. once per backend
# process), not on every request - this keeps prediction latency low.
_pipeline = joblib.load(settings.MODEL_PATH)


# train_model.py encodes gender and these Yes/No columns to 0/1 BEFORE
# they reach the pipeline (the pipeline's ColumnTransformer only one-hot
# encodes the true multi-category columns and passes everything else
# through untouched). We must apply the exact same mapping here, or the
# pipeline will try to do maths on the string "Yes"/"Male" and fail.
_BINARY_YES_NO_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]


def _encode_binary_fields(row: dict) -> dict:
    """Mirror train_model.py's gender/Yes-No encoding for a single record."""
    encoded = dict(row)
    encoded["gender"] = 1 if row["gender"] == "Male" else 0
    for col in _BINARY_YES_NO_COLS:
        encoded[col] = 1 if row[col] == "Yes" else 0
    return encoded


def predict(customer: CustomerData) -> tuple[int, float]:
    """Run the model on one customer and return (prediction, probability).

    `prediction` is 0 or 1 (will-stay / will-churn).
    `probability` is the model's confidence that the customer will churn,
    as a float between 0 and 1.
    """
    # Apply the same encoding train_model.py applied before fitting, then
    # build a one-row dataframe with the same column names it was trained on.
    encoded_row = _encode_binary_fields(customer.model_dump())
    input_df = pd.DataFrame([encoded_row])

    prediction = int(_pipeline.predict(input_df)[0])
    probability = float(_pipeline.predict_proba(input_df)[0][1])

    return prediction, probability
