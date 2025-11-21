import streamlit as st
import pandas as pd
import numpy as np
import joblib

scaler = joblib.load("scaler.pkl")
model_logreg = joblib.load("model_logistic_regression.pkl")
model_rf = joblib.load("model_random_forest.pkl")
model_ensemble = joblib.load("model_ensemble.pkl")

st.title("Heart Disease Prediction App")
st.write("Aplikasi Machine Learning untuk memprediksi penyakit jantung menggunakan 3 model: Logistic Regression, Random Forest, dan Ensemble Voting.")

st.sidebar.header("Pilih Model")
model_choice = st.sidebar.selectbox(
    "Model yang digunakan:",
    ("Logistic Regression", "Random Forest", "Ensemble Voting")
)

st.sidebar.write("Akurasi Model:")
st.sidebar.write("• Logistic Regression: > 90%")
st.sidebar.write("• Random Forest: > 90%")
st.sidebar.write("• Ensemble Voting: > 90%")

st.header("Input Data Baru")

age = st.number_input("Age", 20, 100, 45)
sex = st.selectbox("Sex", ("M", "F"))
chest_pain = st.number_input("Chest Pain Type", 0, 3, 1)
bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.number_input("Fasting Blood Sugar", 0, 1, 0)
restecg = st.number_input("Resting ECG", 0, 2, 0)
maxhr = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.number_input("Exercise Induced Angina", 0, 1, 0)
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.number_input("ST Slope", 0, 2, 1)

sex = 1 if sex == "M" else 0

input_data = np.array([
    age, sex, chest_pain, bp, chol, fbs, restecg,
    maxhr, exang, oldpeak, slope
]).reshape(1, -1)

scaled_data = scaler.transform(input_data)

if st.button("Prediksi"):
    if model_choice == "Logistic Regression":
        pred = model_logreg.predict(scaled_data)
    elif model_choice == "Random Forest":
        pred = model_rf.predict(input_data)
    else:
        pred = model_ensemble.predict(scaled_data)

    if pred[0] == 1:
        st.error("Hasil Prediksi: Berpotensi Mengalami Penyakit Jantung ❗")
    else:
        st.success("Hasil Prediksi: Tidak Berpotensi Penyakit Jantung ✔")

st.write("---")
st.write("Developed for Deadline Jam 12 Malam ⚡")
