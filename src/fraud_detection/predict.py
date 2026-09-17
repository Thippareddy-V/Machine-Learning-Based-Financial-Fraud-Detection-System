import json
import sys
import joblib
import pandas as pd

from functools import lru_cache

from .config import MODEL_PATH, METADATA_PATH

# Required so joblib can reconstruct the custom feature-engineering
# class stored inside the pipeline.
from .train import FraudFeatureEngineer

# The pipeline was pickled from `train.py` run directly, which stored the
# class reference as __main__.FraudFeatureEngineer. Register it on __main__
# so joblib can find it during unpickling.
sys.modules["__main__"].FraudFeatureEngineer = FraudFeatureEngineer


@lru_cache(maxsize=1)
def load_model():
    """Load the trained fraud detection pipeline once."""
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metadata():
    """Load model metadata once."""

    with open(METADATA_PATH, "r") as file:
        return json.load(file)


def predict_transaction(transaction):
    """Predict whether a transaction is fraudulent."""

    model = load_model()
    metadata = load_metadata()

    threshold = metadata["threshold"]

    # Convert one transaction into a DataFrame
    transaction_df = pd.DataFrame([transaction])

    # Get fraud probability
    fraud_probability = float(
        model.predict_proba(transaction_df)[0, 1]
    )

    # Apply saved threshold
    prediction = int(
        fraud_probability >= threshold
    )

    return {
        "prediction": (
            "Fraud"
            if prediction == 1
            else "Legitimate"
        ),
        "fraud_probability": fraud_probability,
        "threshold": threshold,
    }