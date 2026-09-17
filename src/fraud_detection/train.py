import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .config import MODEL_DIR, MODEL_PATH, METADATA_PATH
from .data import load_data, get_X_y


class FraudFeatureEngineer:
    """Create transaction-level features used by the fraud model."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["balanceDiffOrig"] = (
            X["oldbalanceOrg"] - X["newbalanceOrig"]
        )

        X["balanceDiffDest"] = (
            X["newbalanceDest"] - X["oldbalanceDest"]
        )

        X["orig_balance_error"] = (
            X["oldbalanceOrg"] - X["amount"] - X["newbalanceOrig"]
        )

        X["dest_balance_error"] = (
            X["oldbalanceDest"] + X["amount"] - X["newbalanceDest"]
        )

        X["orig_balance_ratio"] = (
            X["amount"] / (X["oldbalanceOrg"] + 1.0)
        )

        X["dest_balance_ratio"] = (
            X["amount"] / (X["oldbalanceDest"] + 1.0)
        )

        X["orig_zero_balance"] = (
            X["oldbalanceOrg"] == 0
        ).astype(int)

        X["dest_zero_balance"] = (
            X["oldbalanceDest"] == 0
        ).astype(int)

        X["amount_log"] = (
            X["amount"].clip(lower=0).apply(lambda x: __import__("numpy").log1p(x))
        )

        return X


def build_pipeline(X):
    """Build the complete fraud detection pipeline."""

    categorical_features = ["type"]

    numerical_features = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFlaggedFraud",
        "balanceDiffOrig",
        "balanceDiffDest",
        "orig_balance_error",
        "dest_balance_error",
        "orig_balance_ratio",
        "dest_balance_ratio",
        "orig_zero_balance",
        "dest_zero_balance",
        "amount_log",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numerical",
                "passthrough",
                numerical_features,
            ),
        ]
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("features", FraudFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def find_best_threshold(y_true, probabilities):
    """Find the threshold that gives the best F1 score."""

    from sklearn.metrics import f1_score

    best_threshold = 0.5
    best_f1 = 0

    for threshold in [i / 100 for i in range(5, 100)]:
        predictions = (probabilities >= threshold).astype(int)

        score = f1_score(y_true, predictions)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold


def train_model():
    """Train and save the fraud detection pipeline."""

    print("Loading dataset...")

    df = load_data()

    X, y = get_X_y(df)

    print(f"Dataset shape: {df.shape}")
    print(f"Fraud transactions: {y.sum()}")
    print(f"Legitimate transactions: {(y == 0).sum()}")

    # 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42,
    )

    # Calculate class imbalance weight from training data
    negatives = (y_train == 0).sum()
    positives = (y_train == 1).sum()

    scale_pos_weight = negatives / positives

    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    pipeline = build_pipeline(X_train)

    # Set class imbalance weight
    pipeline.named_steps["model"].set_params(
        scale_pos_weight=scale_pos_weight
    )

    print("Training model...")

    pipeline.fit(X_train, y_train)

    # Validation probabilities
    val_probabilities = pipeline.predict_proba(X_val)[:, 1]

    threshold = find_best_threshold(
        y_val,
        val_probabilities,
    )

    print(f"Selected threshold: {threshold:.2f}")

    # Create model directory
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save complete pipeline
    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    # Save metadata
    metadata = {
        "threshold": threshold,
        "model_type": "XGBClassifier",
        "target": "isFraud",
        "random_state": 42,
    }

    with open(METADATA_PATH, "w") as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    train_model()