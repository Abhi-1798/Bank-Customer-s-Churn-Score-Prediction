import streamlit as st
import joblib
import pickle
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
    with open("XGBClass_model_bundle.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_model()
ohe = bundle['ohe']
selected_features = bundle['selected_features']
model = bundle['xgbf']

# ------------------ App Interface ------------------
st.title("Customer Churn Prediction")

st.subheader("Enter Customer Details")

# Define input fields based on dataset columns
data_input = {
    "age": st.number_input("Age", min_value=10, max_value=100, step=1),
    "gender": st.selectbox("Gender", ["F", "M", "Unknown"]),
    "region_category": st.selectbox("Region Category", ["City", "Town", "Village", "Unknown"]),
    "membership_category": st.selectbox("Membership Category", ["Basic Membership", "Silver Membership", "Gold Membership", "Platinum Membership", "Premium Membership", "No Membership"]),
    "joined_through_referral": st.selectbox("Joined Through Referral", ["Yes", "No", "?"]),
    "preferred_offer_types": st.selectbox("Preferred Offer Types", ["Credit/Debit Card Offers", "Gift Vouchers/Coupons", "Without Offers"]),
    "medium_of_operation": st.selectbox("Medium of Operation", ["Desktop", "Smartphone", "Both", "?"]),
    "internet_option": st.selectbox("Internet Option", ["Wi-Fi", "Mobile_Data", "Fiber_Optic"]),
    "days_since_last_login": st.number_input("Days Since Last Login", min_value=0, max_value=28),
    "avg_time_spent": st.number_input("Average Time Spent (In seconds)", min_value=0.0),
    "avg_transaction_value": st.number_input("Average Transaction Value", min_value=0.0),
    "avg_frequency_login_days": st.number_input("Avg Frequency Login Days", min_value=0.0),
    "points_in_wallet": st.number_input("Points in Wallet", min_value=0.0),
    "used_special_discount": st.selectbox("Used Special Discount", ["Yes", "No"]),
    "offer_application_preference": st.selectbox("Offer Application Preference", ["Yes", "No"]),
    "past_complaint": st.selectbox("Past Complaint", ["Yes", "No"]),
    "complaint_status": st.selectbox("Complaint Status", ["Solved", "Unsolved", "Solved in Follow-up", "Not Applicable", "No Information Available"]),
    "feedback": st.selectbox("Feedback", [
        "No reason specified", "User Friendly Website", "Reasonable Price", "Quality Customer Care",
        "Products always in Stock", "Poor Customer Service", "Poor Product Quality", "Poor Website", "Too many ads"
    ])
}

# Convert input to DataFrame
input_df = pd.DataFrame([data_input])

# Identify categorical columns from OHE
categorical_cols = ohe.feature_names_in_

# Apply OneHotEncoder
encoded_df = pd.DataFrame(ohe.transform(input_df[categorical_cols]).toarray(), columns=ohe.get_feature_names_out())

# Drop categorical columns and combine with encoded features
numeric_df = input_df.drop(columns=categorical_cols)
processed_df = pd.concat([numeric_df, encoded_df], axis=1)

# Ensure only selected features are used for prediction
final_input = processed_df[selected_features]

# Make prediction
if st.button("Predict"):
    
    prediction = model.predict(final_input)[0]
    st.success(f"Predicted Churn Risk Score: {prediction}")
