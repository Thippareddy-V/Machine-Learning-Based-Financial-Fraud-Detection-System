# Financial Transaction Fraud Detection & Risk Assessment System

<p align="center">
  <strong>End-to-End Machine Learning Application for Detecting Fraudulent Financial Transactions</strong>
</p>

<p align="center">
  <em>XGBoost • Scikit-learn • Pandas • FastAPI • HTML • CSS • JavaScript</em>
</p>

---

## 📌 Overview

The **Financial Transaction Fraud Detection & Risk Assessment System** is an end-to-end machine learning project that analyzes financial transactions and predicts whether a transaction is **Fraudulent** or **Legitimate**.

The project takes the model beyond a notebook and integrates the complete workflow into a web application:

**Transaction Input → Feature Engineering → Preprocessing → XGBoost → Fraud Probability → Risk Assessment → Web Dashboard**

It demonstrates the practical process of taking a machine learning model from **EDA and experimentation to API-based inference and a user-facing application**.

---

## 🎯 Objectives

- Detect potentially fraudulent financial transactions using machine learning.
- Handle severe class imbalance in fraud data.
- Create meaningful transaction-level features.
- Build a reusable preprocessing and prediction pipeline.
- Select a classification threshold using validation data.
- Expose the trained model through a **FastAPI REST API**.
- Provide a clean web interface for transaction analysis.
- Return fraud probability, threshold, and model decision.
- Maintain a modular ML project structure.

---

## 🧠 Machine Learning

### Model

The project uses **XGBoost Classifier** for binary classification.

Target:

```text
isFraud

0 → Legitimate
1 → Fraud
```

XGBoost was selected for its suitability for structured/tabular transaction data and its ability to model nonlinear relationships.

---

## 🔧 Feature Engineering

The pipeline creates transaction-level features such as:

| Feature | Description |
|---|---|
| `balanceDiffOrig` | Difference between origin balance before and after the transaction |
| `balanceDiffDest` | Difference between destination balance after and before the transaction |
| `orig_balance_error` | Difference between expected and observed origin balance |
| `dest_balance_error` | Difference between expected and observed destination balance |
| `orig_balance_ratio` | Transaction amount relative to origin balance |
| `dest_balance_ratio` | Transaction amount relative to destination balance |
| `orig_zero_balance` | Indicates whether the origin account started with zero balance |
| `dest_zero_balance` | Indicates whether the destination account started with zero balance |
| `amount_log` | Log-transformed transaction amount |

These transformations are part of the saved ML pipeline, keeping training and inference consistent.

---

## ⚖️ Class Imbalance

Fraudulent transactions represent a very small fraction of the dataset.

The training process calculates:

```text
scale_pos_weight =
number of legitimate transactions
---------------------------------
number of fraudulent transactions
```

This class weight is passed to XGBoost so the minority fraud class receives greater importance during training.

An optional **SMOTE** experiment was also explored during model development.

---

## 📊 Dataset

The project uses a PaySim-style financial transaction dataset.

### Dataset characteristics

- Approximately **6.36 million transactions**
- Highly imbalanced fraud/legitimate classes
- Target column: `isFraud`

Main columns:

```text
step
type
amount
nameOrig
oldbalanceOrg
newbalanceOrig
nameDest
oldbalanceDest
newbalanceDest
isFraud
isFlaggedFraud
```

`isFraud` is the training target and is not required from the user during prediction.

> **Note:** The dataset is intended for development/educational use. If it is too large for GitHub, store it separately and place it in the expected `data/` directory.

---

## 🏗️ Architecture

```text
                    ┌──────────────────────┐
                    │      Web Browser      │
                    │  HTML/CSS/JavaScript  │
                    └──────────┬───────────┘
                               │
                         POST /predict
                               │
                               ▼
                    ┌──────────────────────┐
                    │      FastAPI API     │
                    │      api/main.py     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Prediction Layer  │
                    │     predict.py       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Saved Pipeline    │
                    │ Feature Engineering  │
                    │   Preprocessing      │
                    │      XGBoost         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Fraud Probability    │
                    │ + Threshold          │
                    │ + Decision           │
                    └──────────────────────┘
```

### Request Flow

1. User enters transaction details.
2. JavaScript sends the transaction to `POST /predict`.
3. FastAPI validates the request.
4. The prediction layer loads the saved pipeline.
5. Feature engineering is applied.
6. Categorical and numerical preprocessing is performed.
7. XGBoost generates a fraud probability.
8. The saved threshold is applied.
9. FastAPI returns the result.
10. The frontend displays the risk assessment.

---

## 📁 Project Structure

```text
fraud-detection/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── app/
│   └── app.py
│
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
├── docs/
│   └── architecture.png
│
├── data/
│   └── AIML Dataset.csv
│
├── models/
│   └── v1/
│       ├── fraud_pipeline.joblib
│       └── metadata.json
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── src/
│   ├── __init__.py
│   └── fraud_detection/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── evaluate.py
│       ├── predict.py
│       └── train.py
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_predict.py
│
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## 🗂️ Core Components

### `notebooks/fraud_detection.ipynb`

Contains the experimentation and analysis workflow:

- EDA
- Class distribution analysis
- Feature engineering experiments
- Model experiments
- Imbalance handling
- Evaluation
- Threshold analysis

### `src/fraud_detection/train.py`

Handles:

- Dataset loading
- Train/validation/test splitting
- Feature engineering
- Preprocessing
- XGBoost configuration
- Class imbalance handling
- Threshold selection
- Model serialization
- Metadata creation

### `src/fraud_detection/predict.py`

Handles inference:

- Loads the saved pipeline
- Loads the saved threshold
- Converts a transaction into the required DataFrame format
- Generates fraud probability
- Applies the threshold
- Returns the prediction

### `src/fraud_detection/evaluate.py`

Provides reusable evaluation for:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Confusion matrix values

### `api/main.py`

FastAPI application exposing:

```text
GET  /
GET  /health
POST /predict
```

### `frontend/`

Contains the web dashboard:

- `index.html` → application structure
- `css/style.css` → styling
- `js/app.js` → API communication and UI state

### `models/v1/`

Stores:

```text
fraud_pipeline.joblib
metadata.json
```

The pipeline packages feature engineering, preprocessing, and the XGBoost model together.

---

## 🔌 API

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

### Prediction

```http
POST /predict
```

Example request:

```json
{
  "type": "TRANSFER",
  "amount": 100000,
  "oldbalanceOrg": 100000,
  "newbalanceOrig": 0,
  "oldbalanceDest": 0,
  "newbalanceDest": 100000,
  "isFlaggedFraud": 0,
  "step": 1,
  "nameOrig": "C123456789",
  "nameDest": "C987654321"
}
```

Example response:

```json
{
  "prediction": "Fraud",
  "fraud_probability": 0.999999,
  "threshold": 0.99
}
```

The displayed probability is the **model's estimated fraud probability**, not a guarantee that the transaction is actually fraudulent.

---

## 📈 Model Evaluation

The current development experiment produced these test-set results at a threshold of approximately `0.99`:

| Metric | Result |
|---|---:|
| PR-AUC | **0.997857** |
| ROC-AUC | **0.999553** |
| Precision | **0.998373** |
| Recall | **0.995942** |
| F1 Score | **0.997156** |
| False Positives | **2** |
| False Negatives | **5** |
| True Positives | **1,227** |
| True Negatives | **953,159** |

Validation PR-AUC:

```text
0.998276
```

### Evaluation note

The very high scores should not automatically be interpreted as real-world production performance. Benchmark/simulated fraud datasets can contain dataset-specific patterns. A production system would require additional validation on genuinely unseen, preferably time-separated and real-world data.

---

## 🎚️ Decision Threshold

The model produces a continuous fraud probability between:

```text
0.0 → 1.0
```

The project does not simply rely on the default `0.50` threshold. A threshold is selected using validation data and stored in:

```text
models/v1/metadata.json
```

Example:

```text
fraud_probability = 0.999
threshold = 0.99

0.999 >= 0.99
        ↓
     Fraud
```

Keeping the threshold in model metadata ensures inference uses the same decision rule selected during development.

---

## 🖥️ Web Dashboard

The frontend provides a transaction-analysis interface where users can enter transaction details and run the model.

The result dashboard can display:

- Risk score
- Model fraud probability
- Decision threshold
- Model decision
- Risk level
- Prediction status

The frontend communicates with FastAPI rather than loading the ML model directly.

```text
Frontend
   ↓
FastAPI
   ↓
Prediction Module
   ↓
ML Pipeline
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd fraud-detection
```

### 2. Create a Python environment

Windows:

```bash
python -m venv .venv
.venv\Scriptsctivate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

Place the dataset at:

```text
data/AIML Dataset.csv
```

Then run:

```bash
python -m src.fraud_detection.train
```

The trained artifacts will be generated under:

```text
models/v1/
```

---

## ▶️ Run the Application

Start FastAPI:

```bash
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Application:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

---

## 🧪 Testing

Run:

```bash
pytest tests/ -v
```

Tests are organized into:

```text
test_api.py
test_data.py
test_predict.py
```

---

## 🐳 Docker

Build:

```bash
docker build -t fraud-detection .
```

Run:

```bash
docker run -p 8000:8000 fraud-detection
```

Open:

```text
http://localhost:8000
```

---

## 🔄 Complete ML Workflow

```text
Raw Dataset
     ↓
EDA
     ↓
Feature Engineering
     ↓
Train / Validation / Test Split
     ↓
Class Imbalance Handling
     ↓
XGBoost Training
     ↓
Validation Evaluation
     ↓
Threshold Selection
     ↓
Final Test Evaluation
     ↓
Save Pipeline + Metadata
     ↓
FastAPI Inference
     ↓
Web Risk Dashboard
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Model Persistence | Joblib |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | HTML5, CSS3, JavaScript |
| Testing | Pytest |
| DevOps | Docker, GitHub Actions |

---

## 💡 What This Project Demonstrates

### Machine Learning

- Exploratory Data Analysis
- Feature engineering
- Imbalanced classification
- XGBoost
- Precision/Recall analysis
- PR-AUC and ROC-AUC
- Threshold selection
- Model evaluation

### Software Engineering

- Modular ML architecture
- Reusable preprocessing pipeline
- Model serialization
- Separation of training and inference
- REST API development
- Input validation
- Automated testing
- Docker support

### End-to-End Deployment Thinking

Instead of saving only the classifier, the project saves the complete pipeline:

```text
Raw Transaction
      ↓
Feature Engineering
      ↓
Preprocessing
      ↓
XGBoost
      ↓
Prediction
```

This reduces the risk of training/inference preprocessing mismatches.

---

## ⚠️ Limitations

This project is intended for **educational and portfolio purposes**.

- The dataset is benchmark/simulated-style data rather than live banking data.
- Performance may differ substantially on real-world transactions.
- The unusually high evaluation scores require independent validation before any production claim.
- Real fraud detection systems require investigation and monitoring workflows in addition to model predictions.
- Production deployment would require stronger security, authentication, logging, model monitoring, drift detection, and privacy controls.
- A model prediction is a risk signal, not a definitive determination of fraud.

---

## 🔮 Future Improvements

- Time-based validation
- SHAP-based model explainability
- Feature drift monitoring
- Model performance monitoring
- API authentication and authorization
- Rate limiting
- Transaction history database
- Batch prediction
- Model versioning
- Cloud deployment
- Automated retraining
- Real-time transaction streaming
- High-risk transaction alerts

---

## 👨‍💻 Project Focus

The primary goal of this project is to demonstrate an **end-to-end machine learning workflow**:

**Data → EDA → Feature Engineering → Imbalanced Classification → XGBoost → Evaluation → Model Pipeline → FastAPI → Web Application**

It combines machine learning knowledge with practical software engineering and deployment concepts.

---

## 📄 License
MIT License

Copyright (c) 2026 Thippa Reddy
