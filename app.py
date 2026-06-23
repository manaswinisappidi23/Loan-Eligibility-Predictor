import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("loan_model.pkl")

st.title("🏦 Loan Eligibility Predictor")

# User Inputs

gender = st.selectbox("Gender", ["Male", "Female"])

married = st.selectbox("Married", ["Yes", "No"])

dependents = st.selectbox("Dependents", [0, 1, 2])

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0
)

loan_term = st.number_input(
    "Loan Term",
    value=360
)

credit_history = st.selectbox(
    "Credit History",
    [1, 0]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Rural", "Semiurban"]
)

if st.button("Predict"):

    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 0 if education == "Graduate" else 1
    self_employed = 1 if self_employed == "Yes" else 0

    if property_area == "Rural":
        property_area = 0
    elif property_area == "Semiurban":
        property_area = 1
    else:
        property_area = 2

    input_data = pd.DataFrame([[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")