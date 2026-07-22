import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import TransactionRequest, TransactionResponse

# Determine base directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root is one level up from backend/
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Load model and scaler
try:
    model = joblib.load(os.path.join(PROJECT_ROOT, "fraud_model.pkl"))
    scaler = joblib.load(os.path.join(PROJECT_ROOT, "scalar.pkl"))
except Exception as e:
    raise RuntimeError(f"Failed to load model or scaler: {e}")

app = FastAPI()

# Add CORS middleware to allow requests from any origin (for Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expected feature order (must match training)
FEATURE_ORDER = [
    'amt', 'gender', 'city_pop', 'month', 'day_of_week', 'hours', 'age',
    'category_food_dining', 'category_gas_transport', 'category_grocery_net',
    'category_grocery_pos', 'category_health_fitness', 'category_home',
    'category_kids_pets', 'category_misc_net', 'category_misc_pos',
    'category_personal_care', 'category_shopping_net', 'category_shopping_pos',
    'category_travel', 'merchant_freq', 'job_freq', 'city_freq', 'state_freq',
    'distance_km'
]

# Columns that were scaled during training (StandardScaler)
SCALED_COLS = [
    'amt', 'city_pop', 'age', 'distance_km', 'hours', 'month', 'day_of_week',
    'merchant_freq', 'job_freq', 'city_freq', 'state_freq'
]

# Original categories (including reference category 'entertainment')
ORIGINAL_CATEGORIES = [
    'entertainment', 'food_dining', 'gas_transport', 'grocery_net', 'grocery_pos',
    'health_fitness', 'home', 'kids_pets', 'misc_net', 'misc_pos', 'personal_care',
    'shopping_net', 'shopping_pos', 'travel'
]

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=TransactionResponse)
def predict_fraud(transaction: TransactionRequest):
    try:
        # Convert Pydantic model to dict
        data = transaction.dict()

        # Extract category for one-hot encoding
        category = data.pop("category")

        # Convert gender string to 0/1 (Male -> 0, Female -> 1)
        # Based on frontend: Male=0, Female=1
        data['gender'] = 0 if data['gender'] == "Male" else 1

        # Initialize all category columns to 0
        for cat in ORIGINAL_CATEGORIES:
            if cat == 'entertainment':
                continue  # reference category, not a column
            col_name = f'category_{cat}'
            data[col_name] = 1 if category == cat else 0

        # Create DataFrame with correct column order
        df = pd.DataFrame([data])
        # Ensure columns are in the exact order as expected by the model
        df = df[FEATURE_ORDER]

        # Scale only the specified columns
        df[SCALED_COLS] = scaler.transform(df[SCALED_COLS])

        # Make prediction
        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0]

        # Probability of fraud (class 1)
        fraud_prob = float(prediction_proba[1])
        is_fraud = bool(prediction == 1)

        # Determine risk level
        if fraud_prob < 0.3:
            risk_level = "Low"
        elif fraud_prob < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return TransactionResponse(
            is_fraud=is_fraud,
            fraud_probability=fraud_prob,
            risk_level=risk_level
        )

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid value: {str(e)}")
    except Exception as e:
        # Log the error for debugging (in production, use proper logging)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")