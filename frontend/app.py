import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# Backend URL from environment variable, default to localhost
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Define the 24 features in the exact order as used during training (for reference)
FEATURES = [
    'amt', 'gender', 'city_pop', 'month', 'day_of_week', 'hours', 'age',
    'category_food_dining', 'category_gas_transport', 'category_grocery_net',
    'category_grocery_pos', 'category_health_fitness', 'category_home',
    'category_kids_pets', 'category_misc_net', 'category_misc_pos',
    'category_personal_care', 'category_shopping_net', 'category_shopping_pos',
    'category_travel', 'merchant_freq', 'job_freq', 'city_freq', 'state_freq',
    'distance_km'
]

# Define which columns were scaled (according to the user)
SCALED_COLS = [
    'amt', 'city_pop', 'age', 'distance_km', 'hours', 'month', 'day_of_week',
    'merchant_freq', 'job_freq', 'city_freq', 'state_freq'
]

# Define the original categories (including the reference category 'entertainment')
ORIGINAL_CATEGORIES = [
    'entertainment', 'food_dining', 'gas_transport', 'grocery_net', 'grocery_pos',
    'health_fitness', 'home', 'kids_pets', 'misc_net', 'misc_pos', 'personal_care',
    'shopping_net', 'shopping_pos', 'travel'
]

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-width: 2px;
        border-style: solid;
    }
    .fraud-result {
        background-color: #fef2f2;
        border-color: #fecaca;
        border-left: 4px solid #ef4444;
    }
    .legit-result {
        background-color: #f0fdf4;
        border-color: #bbf7d0;
        border-left: 4px solid #22c55e;
    }
    .result-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .result-probability {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.375rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .preset-button {
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e5e7eb;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1rem;
        padding-top: 1rem;
    }
    .feature-importance-bar {
        background-color: #f3f4f6;
        border-radius: 0.375rem;
        padding: 1rem;
        margin-top: 1.5rem;
    }
    .feature-importance-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="FraudShield: Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header section
st.markdown('<div class="main-header">FraudShield</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time Credit Card Fraud Detection System</div>', unsafe_allow_html=True)

# GitHub badge in sidebar
with st.sidebar:
    st.markdown("### 🔗 Project Links")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/yourusername/fraud-detection-ml)")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info(
        """
        This demo uses a RandomForestClassifier trained on credit card transaction data.
        The model predicts whether a transaction is fraudulent (1) or legitimate (0).
        """
    )
    st.markdown("### 🔧 Backend Status")
    # We can add a health check indicator here
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("🟢 Backend Online")
        else:
            st.error("🔴 Backend Error")
    except:
        st.error("🔴 Backend Offline")

# Preset buttons section (outside form for immediate effect)
st.markdown("### 🚀 Quick Test Examples")
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("💳 Normal Transaction", key="normal", help="Fill form with typical legitimate transaction values", use_container_width=True):
        # Store preset values in session state
        st.session_state.preset = "normal"
        st.rerun()

with col2:
    if st.button("⚠️ Suspicious Transaction", key="suspicious", help="Fill form with potentially fraudulent transaction values", use_container_width=True):
        st.session_state.preset = "suspicious"
        st.rerun()

# Initialize session state for preset values
if 'preset_values' not in st.session_state:
    st.session_state.preset_values = {}

# Apply preset values if button was clicked
if 'preset' in st.session_state:
    if st.session_state.preset == "normal":
        st.session_state.preset_values = {
            'amt': 45.67,
            'gender': 'Male',
            'city_pop': 50000,
            'month': 6,
            'day_of_week': 2,  # Wednesday
            'hours': 14,
            'age': 35,
            'distance_km': 2.1,
            'merchant_freq': 150,
            'job_freq': 200,
            'city_freq': 300,
            'state_freq': 20,
            'category': 'grocery_net'
        }
    elif st.session_state.preset == "suspicious":
        st.session_state.preset_values = {
            'amt': 895.50,
            'gender': 'Female',
            'city_pop': 1200000,
            'month': 1,
            'day_of_week': 6,  # Sunday
            'hours': 2,
            'age': 22,
            'distance_km': 45.8,
            'merchant_freq': 2,
            'job_freq': 5,
            'city_freq': 8,
            'state_freq': 3,
            'category': 'misc_net'
        }
    # Clear the preset trigger after applying
    del st.session_state.preset

# Main form with tabs
st.markdown("### 📝 Transaction Details")
tab1, tab2, tab3 = st.tabs(["💰 Transaction Info", "📍 Location & Merchant", "👤 Cardholder Info"])

# Initialize form values with preset values or defaults
def get_value(key, default):
    return st.session_state.preset_values.get(key, default)

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        amt = st.number_input(
            "Transaction Amount ($)",
            min_value=0.0,
            value=get_value('amt', 100.0),
            step=0.01,
            key="amt"
        )
        category = st.selectbox(
            "Transaction Category",
            options=ORIGINAL_CATEGORIES,
            index=ORIGINAL_CATEGORIES.index(get_value('category', 'entertainment')),
            key="category"
        )
    with col2:
        gender = st.radio(
            "Gender",
            options=["Male", "Female"],
            horizontal=True,
            index=0 if get_value('gender', 'Male') == "Male" else 1,
            key="gender"
        )
        hours = st.slider(
            "Hour of Transaction (0-23)",
            min_value=0,
            max_value=23,
            value=get_value('hours', 12),
            key="hours"
        )

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        city_pop = st.number_input(
            "City Population",
            min_value=0,
            value=get_value('city_pop', 50000),
            step=1000,
            key="city_pop"
        )
        merchant_freq = st.number_input(
            "Merchant Frequency (1-5000)",
            min_value=1,
            max_value=5000,
            value=get_value('merchant_freq', 100),
            help="How commonly this merchant appears in training data",
            key="merchant_freq"
        )
    with col2:
        distance_km = st.number_input(
            "Distance from Home (km)",
            min_value=0.0,
            value=get_value('distance_km', 5.0),
            step=0.1,
            key="distance_km"
        )
        city_freq = st.number_input(
            "City Frequency (1-5000)",
            min_value=1,
            max_value=5000,
            value=get_value('city_freq', 100),
            help="How commonly this city appears in training data",
            key="city_freq"
        )

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            index=get_value('month', 1)-1,
            format_func=lambda x: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][x-1],
            key="month"
        )
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=get_value('age', 35),
            key="age"
        )
    with col2:
        day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(0, 7)),
            index=get_value('day_of_week', 0),
            format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
            key="day_of_week"
        )
        job_freq = st.number_input(
            "Job Title Frequency (1-5000)",
            min_value=1,
            max_value=5000,
            value=get_value('job_freq', 100),
            help="How commonly this job title appears in training data",
            key="job_freq"
        )
        state_freq = st.number_input(
            "State Frequency (1-5000)",
            min_value=1,
            max_value=5000,
            value=get_value('state_freq', 100),
            help="How commonly this state appears in training data",
            key="state_freq"
        )

# Predict button
st.markdown('<div class="tab-content">', unsafe_allow_html=True)
submitted = st.button("🔍 Analyze Transaction", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# When the form is submitted
if submitted:
    # Basic validation
    validation_errors = []
    if amt < 0:
        validation_errors.append("Transaction amount cannot be negative.")
    if age < 18 or age > 100:
        validation_errors.append("Age must be between 18 and 100.")

    if validation_errors:
        for error in validation_errors:
            st.error(error)
    else:
        # Prepare the input data for the API
        input_data = {
            "amt": amt,
            "gender": gender,
            "city_pop": city_pop,
            "month": month,
            "day_of_week": day_of_week,
            "hours": hours,
            "age": age,
            "category": category,
            "merchant_freq": merchant_freq,
            "job_freq": job_freq,
            "city_freq": city_freq,
            "state_freq": state_freq,
            "distance_km": distance_km
        }

        # Show loading spinner during prediction
        with st.spinner('🔍 Analyzing transaction for fraud patterns...'):
            try:
                # Send request to the backend
                response = requests.post(f"{BACKEND_URL}/predict", json=input_data, timeout=10)
                response.raise_for_status()  # Raise an exception for bad status codes
                result = response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the backend: {str(e)}")
                st.stop()

        # Display result from the API
        is_fraud = result["is_fraud"]
        fraud_prob = result["fraud_probability"] * 100  # Convert to percentage
        risk_level = result["risk_level"]

        # Determine result card styling
        if is_fraud:
            result_card_class = "fraud-result"
            result_title = "🚨 Fraudulent Transaction Detected"
            result_color = "#ef4444"
        else:
            result_card_class = "legit-result"
            result_title = "✅ Legitimate Transaction"
            result_color = "#22c55e"

        # Display result card
        st.markdown(f"""
        <div class="result-card {result_card_class}">
            <div class="result-title">{result_title}</div>
            <div class="result-probability">{fraud_prob:.2f}%</div>
            <div>Risk Level: {risk_level}</div>
        </div>
        """, unsafe_allow_html=True)

        # Show probability progress bar
        st.progress(int(fraud_prob))
        st.caption(f"Fraud Probability: {fraud_prob:.2f}%")

        # Display feature importances if available
        if "feature_importances" in result and result["feature_importances"]:
            st.markdown('<div class="feature-importance-bar">', unsafe_allow_html=True)
            st.markdown('<div class="feature-importance-title">🔍 Top Factors Influencing Decision</div>', unsafe_allow_html=True)

            # Prepare data for horizontal bar chart
            features = [item["feature"] for item in result["feature_importances"]]
            importances = [item["importance"] for item in result["feature_importances"]]

            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            y_pos = np.arange(len(features))
            ax.barh(y_pos, importances, color='#3b82f6')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top 5 Features by Importance')
            plt.tight_layout()

            # Display the plot
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Built with ❤️ using Streamlit & FastAPI |
    <a href="https://github.com/yourusername/fraud-detection-ml" target="_blank">GitHub</a> |
    <a href="https://linkedin.com/in/yourprofile" target="_blank">LinkedIn</a></p>
    <p>© 2026 FraudShield Demo - For portfolio purposes only</p>
</div>
""", unsafe_allow_html=True)