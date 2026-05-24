import streamlit as st
import joblib 
import numpy as np
import pandas as pd

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Customer Churn Predictor")
st.write("Enter customer details to predict if they will churn")

st.subheader("Customer Information")

col1,col2=st.columns(2)

with col1:
    gender=st.selectbox("Gender",["Male","Female"])
    senior_citizen=st.selectbox("Senior Citizen",["No","Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

with col2:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)
    multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])

st.subheader("Services & Contract")

col3, col4 = st.columns(2)

with col3:
    tech_support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])

with col4:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ])

#PREPROCESSING-label encoding and one-hot encoding what we did in collab notebook


def preprocess(data):
    data['gender'] = 1 if data['gender'] == 'Male' else 0
    data['SeniorCitizen'] = 1 if data['SeniorCitizen'] == 'Yes' else 0
    data['Partner'] = 1 if data['Partner'] == 'Yes' else 0
    data['Dependents'] = 1 if data['Dependents'] == 'Yes' else 0
    data['PhoneService'] = 1 if data['PhoneService'] == 'Yes' else 0
    data['PaperlessBilling'] = 1 if data['PaperlessBilling'] == 'Yes' else 0

    data['MultipleLines_No phone service'] = 1 if data['MultipleLines'] == 'No phone service' else 0
    data['MultipleLines_Yes'] = 1 if data['MultipleLines'] == 'Yes' else 0

    data['InternetService_Fiber optic'] = 1 if data['InternetService'] == 'Fiber optic' else 0
    data['InternetService_No'] = 1 if data['InternetService'] == 'No' else 0

    data['OnlineSecurity_No internet service'] = 1 if data['OnlineSecurity'] == 'No internet service' else 0
    data['OnlineSecurity_Yes'] = 1 if data['OnlineSecurity'] == 'Yes' else 0

    data['OnlineBackup_No internet service'] = 1 if data['OnlineBackup'] == 'No internet service' else 0
    data['OnlineBackup_Yes'] = 1 if data['OnlineBackup'] == 'Yes' else 0

    data['DeviceProtection_No internet service'] = 1 if data['DeviceProtection'] == 'No internet service' else 0
    data['DeviceProtection_Yes'] = 1 if data['DeviceProtection'] == 'Yes' else 0

    data['TechSupport_No internet service'] = 1 if data['TechSupport'] == 'No internet service' else 0
    data['TechSupport_Yes'] = 1 if data['TechSupport'] == 'Yes' else 0


    data['StreamingTV_No internet service'] = 1 if data['StreamingTV'] == 'No internet service' else 0
    data['StreamingTV_Yes'] = 1 if data['StreamingTV'] == 'Yes' else 0

    data['StreamingMovies_No internet service'] = 1 if data['StreamingMovies'] == 'No internet service' else 0
    data['StreamingMovies_Yes'] = 1 if data['StreamingMovies'] == 'Yes' else 0

    data['Contract_One year'] = 1 if data['Contract'] == 'One year' else 0
    data['Contract_Two year'] = 1 if data['Contract'] == 'Two year' else 0

    data['PaymentMethod_Credit card (automatic)'] = 1 if data['PaymentMethod'] == 'Credit card (automatic)' else 0
    data['PaymentMethod_Electronic check'] = 1 if data['PaymentMethod'] == 'Electronic check' else 0
    data['PaymentMethod_Mailed check'] = 1 if data['PaymentMethod'] == 'Mailed check' else 0

    features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'MultipleLines_No phone service', 'MultipleLines_Yes',
        'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
        'OnlineBackup_No internet service', 'OnlineBackup_Yes',
        'DeviceProtection_No internet service', 'DeviceProtection_Yes',
        'TechSupport_No internet service', 'TechSupport_Yes',
        'StreamingTV_No internet service', 'StreamingTV_Yes',
        'StreamingMovies_No internet service', 'StreamingMovies_Yes',
        'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    return pd.DataFrame([[data[f] for f in features]], columns=features)


if st.button("Predict Churn"):

    raw_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'PaperlessBilling': paperless_billing,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaymentMethod': payment_method
    }
    input_df = preprocess(raw_data)
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Display result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f" This customer is likely to CHURN (Probability: {probability:.1%})")
    else:
        st.success(f"This customer is likely to STAY (Probability of churn: {probability:.1%})")