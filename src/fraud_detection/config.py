from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Data
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "AIML Dataset.csv"


# Model
MODEL_DIR = PROJECT_ROOT / "models" / "v1"
MODEL_PATH = MODEL_DIR / "fraud_pipeline.joblib"


# Metadata
METADATA_PATH = MODEL_DIR / "metadata.json"