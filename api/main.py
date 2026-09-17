from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.fraud_detection.predict import predict_transaction


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent financial transactions",
    version="1.0.0",
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Serve frontend static files
# --------------------------------------------------

app.mount(
    "/css",
    StaticFiles(directory=FRONTEND_DIR / "css"),
    name="css",
)

app.mount(
    "/js",
    StaticFiles(directory=FRONTEND_DIR / "js"),
    name="js",
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class TransactionRequest(BaseModel):
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

    isFlaggedFraud: int = 0
    step: int = 1

    nameOrig: str = "unknown"
    nameDest: str = "unknown"


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: TransactionRequest):

    try:
        transaction = request.model_dump()

        result = predict_transaction(transaction)

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )