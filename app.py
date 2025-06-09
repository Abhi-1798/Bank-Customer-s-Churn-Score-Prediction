import streamlit as st
import pickle
import pandas as pd
import streamlit_authenticator as stauth

# ----- USER AUTHENTICATION SETUP -----
names = ['Abhishek Baranwal']
usernames = ['abhishek']
passwords = ['1234567']  # plain text, for demo only

hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(
    {"usernames": {
        usernames[0]: {
            "name": names[0],
            "password": hashed_passwords[0]
        }
    }},
    "bank_churn_app", "a_secure_random_key_2025", cookie_expiry_days=1
)

name, auth_status, username = authenticator.login('Login', 'main')

if auth_status:
    st.sidebar.success(f"Welcome, {name} 👋")
    authenticator.logout('Logout', 'sidebar')

    # Load the model and encoder
    with open("XGBClass_model_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
        model = bundle["model"]
        ohe = bundle["encoder"]
        selected_features = bundle["selected_features"]

    # Streamlit UI
    st.title("📊 Customer Churn Prediction")

    st.markdown("### Enter Customer Details:")

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

    input_df = pd.DataFrame([data_input])

    # Identify categorical columns used during training
    categorical_cols = list(ohe.feature_names_in_)

    # Encode categorical columns
    encoded_df = pd.DataFrame(ohe.transform(input_df[categorical_cols]).toarray(), columns=ohe.get_feature_names_out())

    # Drop categorical columns from original input and concatenate with encoded columns
    numeric_df = input_df.drop(columns=categorical_cols)
    processed_df = pd.concat([numeric_df, encoded_df], axis=1)

    # Select only the features used in training
    final_input = processed_df[selected_features]

    if st.button("Predict"):
        prediction = model.predict(final_input)[0]
        st.success(f"Predicted Churn Risk Score: {prediction}")

elif auth_status is False:
    st.error("Username or password is incorrect")
elif auth_status is None:
    st.warning("Please enter your username and password")
