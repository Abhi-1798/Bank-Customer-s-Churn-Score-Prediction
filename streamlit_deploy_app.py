import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ------------------ Authentication ------------------
USERNAME = "admin"
PASSWORD = "password123"  # In real apps, don't hardcode like this

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.authenticated = True
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    return joblib.load("XGBClass_model_bundle.pkl")

bundle = load_model()
ohe = bundle['ohe']
selected_features = bundle['selected_features']
model = bundle['xgbf']

# ------------------ App Interface ------------------
st.title("Customer Churn Prediction")

st.subheader("Enter Customer Details")

# Replace with actual categorical & numerical columns
data_input = {
    "Gender": st.selectbox("Gender", ["Male", "Female"]),
    "Geography": st.selectbox("Geography", ["France", "Germany", "Spain"]),
    "Age": st.number_input("Age", min_value=18, max_value=100, step=1),
    "Balance": st.number_input("Balance", min_value=0.0),
    "NumOfProducts": st.number_input("Num Of Products", min_value=1, max_value=4, step=1),
    "IsActiveMember": st.selectbox("Is Active Member", [0, 1]),
    "EstimatedSalary": st.number_input("Estimated Salary", min_value=0.0)
}

# Convert input to DataFrame
input_df = pd.DataFrame([data_input])

# Apply OneHotEncoder (assumes ohe was fit on full dataframe earlier)
ohe_cols = ohe.get_feature_names_out()
categorical_cols = ohe.feature_names_in_
ohe_df = pd.DataFrame(ohe.transform(input_df[categorical_cols]).toarray(), columns=ohe_cols)

# Combine with numeric columns
numeric_df = input_df.drop(columns=categorical_cols)
processed_df = pd.concat([numeric_df, ohe_df], axis=1)

# Select only the features used during training
final_input = processed_df[selected_features]

# Make prediction
if st.button("Predict"):
    prediction = model.predict(final_input)[0]
    st.success(f"Predicted Churn Risk Score: {prediction}")
