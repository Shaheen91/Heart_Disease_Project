import streamlit as st
import pandas as pd
import joblib
import os

# ===============================
# Load trained pipeline
# ===============================
model_path = os.path.join(os.path.dirname(__file__), "../models/final_model.pkl")
pipeline = joblib.load(model_path)

st.title("💓 Heart Disease Prediction App")
st.write("Enter patient data to predict heart disease class.")

# -------------------------------
# Single Patient Prediction
# -------------------------------
st.subheader("Single Patient Input")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=[0, 1])
cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.selectbox("Resting ECG", options=[0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope", options=[0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3) colored by fluoroscopy", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])

if st.button("Predict"):
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })

    prediction = pipeline.predict(input_data)[0]
    probabilities = pipeline.predict_proba(input_data)[0]

    st.success(f"Predicted Heart Disease Class: {prediction}")

    # Show probabilities table with index
    st.dataframe(pd.DataFrame(probabilities, index=range(len(probabilities)), columns=["Probability"]).style.format({"Probability": "{:.2%}"}))
