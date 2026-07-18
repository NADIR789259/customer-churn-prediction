
import streamlit as st
import pandas as pd
from pathlib import Path
import pickle



def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# -------------------------------
# Load Model and Scaler
# -------------------------------


model = pickle.load(open("models/customer_churn_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Customer Churn Prediction")
st.write("Enter customer details to predict whether the customer is likely to churn.")

# -------------------------------
# User Inputs
# -------------------------------

credit_score = st.number_input("Credit Score", 350, 850, 650)

country = st.selectbox(
    "Country",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

age = st.slider("Age", 18, 92, 35)

tenure = st.slider("Tenure", 0, 10, 5)

balance = st.number_input("Balance", min_value=0.0, value=50000.0)

products_number = st.selectbox(
    "Number of Products",
    [1,2,3,4]
)

credit_card = st.selectbox(
    "Has Credit Card?",
    [0,1]
)

active_member = st.selectbox(
    "Is Active Member?",
    [0,1]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=70000.0
)

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict"):

    # One-Hot Encoding

    country_germany = 1 if country == "Germany" else 0
    country_spain = 1 if country == "Spain" else 0
    gender_male = 1 if gender == "Male" else 0

    input_data = pd.DataFrame([{
        "credit_score": credit_score,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "products_number": products_number,
        "credit_card": credit_card,
        "active_member": active_member,
        "estimated_salary": estimated_salary,
        "country_Germany": country_germany,
        "country_Spain": country_spain,
        "gender_Male": gender_male
    }])

    # Scaling
    numerical_columns = [
        "credit_score",
        "age",
        "tenure",
        "balance",
        "products_number",
        "estimated_salary"
    ]

    input_data[numerical_columns] = scaler.transform(
        input_data[numerical_columns]
    )

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    st.write(f"**Churn Probability:** {probability*100:.2f}%")
    
    # Footer
    

st.markdown(
    """
    <div style="text-align:center; color:#9CA3AF; font-size:14px; padding:20px 0; line-height:1.8;">
        <strong style="font-size:16px;">Developed by Nadir Khan ❤️</strong><br>
        <span>Python • Machine Learning • Streamlit</span>
    </div>
    """,
    unsafe_allow_html=True
)