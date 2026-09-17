import pandas as pd

from .config import DATA_PATH


def load_data():
    """Load the fraud detection dataset."""
    return pd.read_csv(DATA_PATH)


def get_X_y(df):
    """Separate features and target."""
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    return X, y