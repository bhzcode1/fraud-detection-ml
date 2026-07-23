# Fraud Detection ML Project

A production‑ready end‑to‑end credit‑card fraud detection system built with a Random Forest classifier, a FastAPI backend, and a Streamlit frontend. The project demonstrates data‑science best practices, proper ML engineering, and a clean two‑service architecture suitable for a portfolio or resume.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [API Endpoints](#api-endpoints)
- [Frontend (Streamlit)](#frontend-streamlit)
- [Results & Evaluation](#results--evaluation)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Contact](#contact)

---

## Overview
Credit‑card fraud is a rare but costly problem. This project trains a **Random Forest classifier** on the Sparkov simulated fraud dataset (~1.3 M transactions, ≈0.5‑1 % fraud rate) using engineered features (temporal, geographic, frequency‑encoded high‑cardinality fields, and Haversine distance).  
The model is evaluated with **precision, recall, and PR‑AUC** (more appropriate than accuracy for imbalanced data).  
A **FastAPI** service exposes a `/predict` endpoint that handles input validation, preprocessing, and returns a fraud probability with a risk level.  
A **Streamlit** application provides an interactive demo, complete with preset scenarios, visual explanations, and a professional UI.

---

## Features
- **Robust ML pipeline**: feature engineering, scaling, class‑weight balancing, and evaluation focused on precision/recall.
- **Production‑ready backend**: FastAPI with Pydantic validation, automatic OpenAPI docs (`/docs`), CORS support, and error handling.
- **Interactive frontend**: Streamlit app with tabbed inputs, quick‑test presets, live spinner, probability gauge, and feature‑importance explanations.
- **Clean repository layout**: separate `backend/` and `frontend/` folders, model and scaler stored in the project root, all auxiliary files (notebook, README, requirements) clearly documented.
- **Portfolio‑ready documentation**: annotated Jupyter notebook (`model.ipynb`) walking through each step, and this README.

---

## System Architecture
```
project_root/
│
├── backend/                 # FastAPI service
│   ├── main.py              # API definition, model loading, / prediction logic
│   ├── schemas.py           # Pydantic request/response models
│   └── requirements.txt     # fastapi, uvicorn, scikit-learn, pandas, numpy, joblib, pydantic
│
├── frontend/                # Streamlit demo
│   ├── app.py               # UI that calls the backend via HTTP
│   └── requirements.txt     # streamlit, requests, matplotlib, pandas, numpy
│
├── fraud_model.pkl          # Trained RandomForestClassifier
├── scalar.pkl               # Fitted StandardScaler (fit on training data)
├── fraudTrain.csv           # Original dataset (git‑ignored)
├── model.ipynb              # Fully annotated notebook (no code changes)
├── README.md                # This file
└── .gitignore               # Excludes data, venv, __pycache__, etc.
```

**Data flow**
1. User enters transaction details in the Streamlit UI.
2. Frontend sends a JSON payload to `POST /predict` on the backend.
3. Backend validates the payload with Pydantic, converts it to the 24‑feature vector (gender mapping, one‑hot for category, frequency encoding, Haversine distance), scales the appropriate columns, runs the model, and returns `{is_fraud, fraud_probability, risk_level}`.
4. Frontend displays the result with a styled card, probability bar, and (optionally) feature‑importance charts.

---

## Setup & Installation

### Prerequisites
- Python 3.9+ (tested on 3.12)
- `pip` (or `conda`)
- Git (optional)

### Step‑by‑step

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd fraud-detection-ml
   ```

2. **(Optional) Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```

3. **Install backend dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   cd ..
   ```

4. **Install frontend dependencies**
   ```bash
   cd frontend
   pip install -r requirements.txt
   cd ..
   ```

5. **Verify model files exist**  
   The project root should contain `fraud_model.pkl` and `scalar.pkl`.  
   (If you just cloned the repo, they are already present.)

### Running the Services

**Start the backend**
```bash
cd backend
uvicorn main:app --reload   # defaults to http://127.0.0.1:8000
```

**Start the frontend** (in a separate terminal)
```bash
cd frontend
streamlit run app.py
```

The Streamlit app will automatically point to `http://localhost:8000` unless you set the environment variable:
```bash
export BACKEND_URL="http://your-host:8000"   # Linux/macOS
set BACKEND_URL=http://your-host:8000       # Windows CMD
```

Open the URL shown by Streamlit (usually `http://localhost:8501`) to use the demo.

---

## Model Details
- **Algorithm**: `RandomForestClassifier` (n_estimators=100 by default, `class_weight='balanced'`, `random_state=42`).
- **Input features (24, in exact order)**:
  ```
  ['amt', 'gender', 'city_pop', 'month', 'day_of_week', 'hours', 'age',
   'category_food_dining', 'category_gas_transport', 'category_grocery_net',
   'category_grocery_pos', 'category_health_fitness', 'category_home',
   'category_kids_pets', 'category_misc_net', 'category_misc_pos',
   'category_personal_care', 'category_shopping_net', 'category_shopping_pos',
   'category_travel', 'merchant_freq', 'job_freq', 'city_freq', 'state_freq',
   'distance_km']
  ```
- **Preprocessing**:
  - `gender`: **Male** → 0, **Female** → 1.
  - `category`: one‑hot encoded (dropped first category *entertainment* → all zeros when selected).
  - `merchant`, `job`, `city`, `state`: frequency‑encoded (counts of each value in training data).
  - `distance_km`: Haversine distance between cardholder (`lat`, `long`) and merchant (`merch_lat`, `merch_long`).
  - Scaled columns (`StandardScaler` fitted **only on training data**):
    `amt`, `city_pop`, `age`, `distance_km`, `hours`, `month`, `day_of_week`,
    `merchant_freq`, `job_freq`, `city_freq`, `state_freq`.
- **Evaluation metrics** (on hold‑out test set):
  - **Precision** (of predicted frauds): proportion of flagged transactions that are truly fraudulent.
  - **Recall** (sensitivity): proportion of actual frauds caught by the model.
  - **PR‑AUC** (area under precision‑recall curve): overall ranking ability, preferred for highly imbalanced data.
  - Accuracy is **not** reported because it is misleading (>99 % accuracy can be achieved by always predicting “non‑fraud”).

For exact numbers, see the *“Results & Evaluation”* section below or the final cells of `model.ipynb`.

---

## API Endpoints
All endpoints are automatically documented at `http://127.0.0.1:8000/docs` (Swagger UI) when the backend runs.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Simple health check: `{"message":"Fraud Detection API is running"}`. |
| `GET` | `/health` | Returns `{"status":"healthy"}` for uptime monitoring. |
| `POST` | `/predict` | Accepts a JSON body matching `TransactionRequest` (see `schemas.py`). Returns `TransactionResponse`:<br>`{ "is_fraud": bool, "fraud_probability": float (0‑1), "risk_level": "Low" \| "Medium" \| "High" }`.<br>Risk levels: **Low** (<0.30), **Medium** (0.30‑0.70), **High** (≥0.70). |

**Example request (curl)**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "amt": 100.50,
        "gender": "Male",
        "city_pop": 50000,
        "month": 6,
        "day_of_week": 2,
        "hours": 14,
        "age": 35,
        "category": "grocery_net",
        "merchant_freq": 150,
        "job_freq": 200,
        "city_freq": 300,
        "state_freq": 20,
        "distance_km": 2.1
      }'
```

**Example response**
```json
{
  "is_fraud": false,
  "fraud_probability": 0.25,
  "risk_level": "Low"
}
```

**Error handling**
- Validation errors (missing fields, wrong types, out‑of‑range values) → `422 Unprocessable Entity` with details.
- Unexpected errors during prediction → `500 Internal Server Error` with a clear message.

---

## Frontend (Streamlit)
The frontend provides a polished, interactive demo:

- **Professional UI**: custom CSS, styled result cards, probability progress bar, and clear typography.
- **Input organization**: tabs for *Transaction Info*, *Location & Merchant*, and *Cardholder Info*.
- **Quick‑test presets**: buttons that auto‑fill the form with a “normal” or a “suspicious” transaction for one‑click testing.
- **Live feedback**: spinner during prediction, instant result display with color‑coded risk (red = fraud, green = legitimate).
- **Explainability**: after prediction, shows the top‑5 feature importances as a horizontal bar chart (derived from the model) and lists the exact values used for those features.
- **Environment configuration**: reads `BACKEND_URL` from environment (defaults to `http://localhost:8000`), making it trivial to point at a deployed API.
- **Footer**: includes links to GitHub and a placeholder LinkedIn.

Run as described in *Setup & Installation*.

---

## Results & Evaluation
(These values come from the final evaluation cells of `model.ipynb`; they are **not** recomputed here.)

- **Accuracy**: 0.9985 (not informative due to class imbalance).
- **Precision (fraud class)**: ~0.98  
  → When the model flags a transaction as fraudulent, about 98 % of those are truly fraudulent.
- **Recall (fraud class)**: ~0.77  
  → The model catches roughly 77 % of all actual fraudulent transactions.
- **F1‑score (fraud)**: ~0.86.
- **PR‑AUC**: ~0.935  
  → Indicates excellent ability to rank fraudulent transactions higher than legitimate ones.
- **Confusion matrix (test set)**:
  ```
  [[257790    25]
   [  352   1168]]
  ```
  - **True Negatives (TN)**: 257,790 legitimate transactions correctly identified.
  - **False Positives (FP)**: 25 legitimate transactions incorrectly flagged (low cost).
  - **False Negatives (FN)**: 352 fraudulent transactions missed (hits missed (higher cost – area for improvement).
  - **True Positives (TP)**: 1,168 frauds correctly detected.

- **Top‑5 feature importances**:
  1. `amt` (transaction amount) – highest impact; unusually large/small amounts are strong fraud signals.
  2. `hours` (hour of day) – fraud tends to cluster in certain times (e.g., late night).
  3. `merchant_freq` – how often a merchant appears; rare or very common merchants can be suspicious.
  4. `city_freq` – frequency of the transaction city.
  5. `age` – cardholder age contributes modestly.

These insights align with domain intuition and validate that the model is learning meaningful patterns.

---

## Future Improvements
- **Stratified train/test split**: add `stratify=Y` to guarantee preserved fraud rate in each split (currently omitted but noted in the notebook).
- **Cyclical encoding** for temporal features (`hour`, `month`, `day_of_week`) to capture periodicity.
- **Alternative imbalance techniques**: compare SMOTE, undersampling, different class‑weight schemes, or threshold tuning.
- **Model experimentation**: test gradient‑boosting libraries (XGBoost, LightGBM, CatBoost) or simple neural nets for potential gains.
- **Enhanced features**:
  - Replace static frequency encoding with real‑time merchant risk scores or velocity features (e.g., transaction count in last 5 min).
  - Add additional engineered fields like transaction‑to‑average‑amount ratio, or distance‑from‑home deviation.
- **Production hardening**:
  - Dockerize backend and frontend for consistent deployment.
  - Add authentication / rate‑limiting to the API.
  - Implement logging and monitoring (Prometheus/Grafana, ELK).
  - Set up CI/CD pipeline with automated testing.
- **Data & concept drift**: periodically retrain on newer data, evaluate performance over time, and implement drift detection.
- **Explainability APIs**: integrate SHAP values or LIME into the backend to return per‑prediction explanations alongside the fraud probability.

---

## License
This project is released under the **MIT License** – see the `LICENSE` file for details.

---

## Contact
- **Author**: Sangam (bhzcode01)  
- **GitHub**: [https://github.com/bhzcode01/fraud-detection-ml](https://github.com/bhzcode01/fraud-detection-ml)  
- **LinkedIn**: (add your profile URL if desired)  

Feel free to open issues, submit pull requests, or reach out for collaborations!

--- 

*Happy modeling!*
