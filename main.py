import streamlit as st
import pandas as pd
import numpy as np
import joblib

model_logreg = joblib.load("model_logistic_regression.pkl")
model_rf = joblib.load("model_random_forest.pkl")
model_ensemble = joblib.load("model_ensemble.pkl")

st.title("Heart Disease Prediction App")
st.write("Menggunakan Logistic Regression, Random Forest, dan Ensemble Voting")

st.sidebar.header("Input Data Pasien")

def user_input():
    data = {
        "Age": st.sidebar.slider("Age", 20, 100, 50),
        "Sex": st.sidebar.selectbox("Sex", ["M", "F"]),
        "ChestPainType": st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"]),
        "RestingBP": st.sidebar.slider("RestingBP", 80, 200, 120),
        "Cholesterol": st.sidebar.slider("Cholesterol", 100, 600, 200),
        "FastingBS": st.sidebar.selectbox("Fasting Blood Sugar", [0, 1]),
        "RestingECG": st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"]),
        "MaxHR": st.sidebar.slider("Max HR", 60, 200, 150),
        "ExerciseAngina": st.sidebar.selectbox("Exercise Angina", ["Y", "N"]),
        "Oldpeak": st.sidebar.slider("Oldpeak", -2.0, 6.0, 1.0),
        "ST_Slope": st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])
    }
    return pd.DataFrame([data])

input_df = user_input()

st.subheader("Input Data")
st.write(input_df)

choose_model = st.selectbox(
    "Pilih Model untuk Prediksi",
    ["Logistic Regression", "Random Forest", "Ensemble Voting"]
)

if st.button("Prediksi"):
    if choose_model == "Logistic Regression":
        pred = model_logreg.predict(input_df)[0]
    elif choose_model == "Random Forest":
        pred = model_rf.predict(input_df)[0]
    else:
        pred = model_ensemble.predict(input_df)[0]

    st.subheader("Hasil Prediksi")
    st.write("Berpotensi Mengidap Penyakit Jantung" if pred == 1 else "Tidak Berpotensi Penyakit Jantung")
