# 🏦 Bank Customer Churn Prediction

This project focuses on building and deploying a machine learning model to predict customer churn score in the banking sector.
By analyzing customer demographics, usage behavior, and feedback, the model identifies customers who are likely to leave, enabling proactive retention strategies.

---

## 📌 Project Overview

- **Objective:** Predict whether a customer will churn based on various behavioral and demographic factors.
- **Model Used:** XGBoost Classifier
- **Deployment:** Streamlit app hosted on Render
- **Dataset:** Bank customer dataset with features like gender, region, membership category, medium of operation, internet option, complaint status, and feedback.

---

## 🚀 Key Features

- ✅ Performed data preprocessing including handling missing values and encoding categorical variables using `OneHotEncoder`.
- ✅ Engineered relevant features and handled a special case for `churn_risk_score = -1` (customers not likely to churn).
- ✅ Trained an **XGBoost classifier** with selected features to maximize prediction accuracy.
- ✅ Developed an **interactive Streamlit dashboard** with slicers for key customer segments (gender, region, etc.).
- ✅ Visualized key churn insights using bar plots, pie charts, and heatmaps.
- ✅ Deployed the model and app on **Render** for live user interaction.

---

## 🧠 Tech Stack

- **Language:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, streamlit  
- **Deployment Platform:** Render  
- **Model Serialization:** `joblib` for saving XGBoost model with preprocessing pipeline  

---

## 📦 How to Use

- Login using your credentials
- Enter feature values in the input form
- Click Predict Demand
- View the predicted value immediately

## 📁 Deployment
The app is live and accessible at: 🔗 https://bank-customer-churn-eda-deployment.onrender.com/

**User Name:** abhishek 
**Password:** 1234567
