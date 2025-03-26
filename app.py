import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("optimized_xgboost_model.pkl")

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
        }
        .main-title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #4B0082;
        }
        .sub-text {
            text-align: center;
            font-size: 18px;
            color: #444;
        }
        .stButton>button {
            background-color: #ffcc00;
            color: black;
            border-radius: 10px;
            font-size: 18px;
            padding: 10px;
            width: 100%;
        }
        .stNumberInput>div>input {
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown('<h1 class="main-title">🎬 Movie Revenue Prediction App 💰</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Enter movie details below to predict the revenue!</p>', unsafe_allow_html=True)

# Layout using columns for better structure
col1, col2 = st.columns(2)

with col1:
    budget = st.number_input("🎥 Enter Budget ($):", min_value=0, value=50000000, step=1000000)
    score = st.number_input("⭐ IMDb Score (1-10):", min_value=1.0, max_value=10.0, value=7.0, step=0.1)

with col2:
    votes = st.number_input("📊 Number of Votes:", min_value=0, value=100000, step=1000)
    runtime = st.number_input("⏳ Movie Runtime (minutes):", min_value=30, max_value=240, value=120, step=5)

# Centering the button
st.markdown("<br>", unsafe_allow_html=True)  # Add some space

# Predict revenue when the button is clicked
if st.button("🚀 Predict Revenue"):
    input_data = np.array([[budget, votes, score, runtime]])
    prediction = model.predict(input_data)
    st.success(f"🎥 Predicted Gross Revenue: **${prediction[0]:,.2f}**")  
