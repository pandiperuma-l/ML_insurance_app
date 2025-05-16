# app.py
import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("insurance_model.pkl")

st.title("💰 Insurance Charges Prediction")

# Input form
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Prediction
if st.button("Predict"):
    user_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])
    prediction = model.predict(user_data)[0]
    st.success(f"Predicted Insurance Charge: ${prediction:,.2f}")
