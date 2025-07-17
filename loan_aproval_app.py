
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model and scaler
model = joblib.load('Loan_aproval_model.pkl')
scaler = joblib.load('scaler.pkl')  # Remove this line if no scaler is used

# App title
st.title('🏦 Loan Approval Prediction App')

# Sidebar info
st.sidebar.title("🔍 About")
st.sidebar.image("yash_profile.jpg", width=150)
st.sidebar.markdown("**Developer**: Yash Patil")
st.sidebar.markdown("**Project**: Loan Approval Prediction")
st.sidebar.markdown("[📂 Dataset Source](https://drive.google.com)")

# Contact info
st.sidebar.markdown("📧 **Email**: [yashpatil877@gmail.com](mailto:yashpatil877@gmail.com)")
st.sidebar.markdown("🔗 **LinkedIn**: [yash-patil-560933b9](https://www.linkedin.com/in/yash-patil-560933b9)")

st.write("Enter the applicant's details below to predict whether the loan will be **Approved** or **Rejected**.")

# User inputs
gender = st.selectbox('👤 Gender', ['Male', 'Female'])
married = st.selectbox('💍 Married', ['Yes', 'No'])
dependents = st.selectbox('👶 Number of Dependents', ['0', '1', '2', '3+'])
education = st.selectbox('🎓 Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('💼 Self-Employed', ['Yes', 'No'])
applicant_income = st.number_input('💰 Applicant Income (Monthly in ₹)', min_value=0.0, step=100.0)
coapplicant_income = st.number_input('🤝 Coapplicant Income', min_value=0.0, step=100.0)
loan_amount = st.number_input('🏦 Loan Amount (in ₹1000s)', min_value=1.0, step=1.0)
loan_term = st.number_input('📅 Loan Term (in days)', min_value=1.0, step=1.0)
credit_history = st.selectbox('🧾 Credit History', ['Good (1)', 'Bad (0)'])
property_area = st.selectbox('🏘 Property Area', ['Urban', 'Semiurban', 'Rural'])

# Preprocessing inputs
dependents = 3 if dependents == '3+' else int(dependents)
credit_history = 1.0 if credit_history == 'Good (1)' else 0.0

# Manual one-hot encoding based on training features
input_dict = {
    'Dependents': dependents,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_term,
    'Credit_History': credit_history,

    'Gender_Female': 1 if gender == 'Female' else 0,
    'Gender_Male': 1 if gender == 'Male' else 0,

    'Married_No': 1 if married == 'No' else 0,
    'Married_Yes': 1 if married == 'Yes' else 0,

    'Education_Graduate': 1 if education == 'Graduate' else 0,
    'Education_Not Graduate': 1 if education == 'Not Graduate' else 0,

    'Self_Employed_No': 1 if self_employed == 'No' else 0,
    'Self_Employed_Yes': 1 if self_employed == 'Yes' else 0,

    'Property_Area_Rural': 1 if property_area == 'Rural' else 0,
    'Property_Area_Semiurban': 1 if property_area == 'Semiurban' else 0,
    'Property_Area_Urban': 1 if property_area == 'Urban' else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
if st.button('🔍 Predict Loan Approval'):
    prediction = model.predict(input_scaled)[0]
    result = '✅ Approved' if prediction == 1 else '❌ Rejected'
    st.success(f"Loan Status Prediction: **{result}**")
