import streamlit as st
import requests
import pandas as pd


st.header("Insurance Premium Category Prediction System")

with st.form(key="person"):
    bmi = st.number_input("Body-Mass Index of the person", min_value=10)
    age_group = st.selectbox("Choose the age_group of the person", options=["young", "adult", "middle_aged", "senior"])
    lifestyle_risk= st.selectbox("Choose the lifestyle risk of the person", options=["low", "medium", "high"])
    city_tier = st.selectbox("Choose the city tier of the person", options=["1", "2", "3"])
    income_lpa = st.number_input("Salary (in LPA) of the person")
    occupation = st.selectbox("Choose the occupation of the person", options=["freelancer", "retired", "student", "private_job", "unemployed", "business_owner", "government_job"])

    submit_button = st.form_submit_button("Submit")

url = "http://127.0.0.1:8000/check/"

if submit_button:
    input_data = {
    "bmi": bmi,
    "age_group": age_group,
    "lifestyle_risk": lifestyle_risk,
    "city_tier": city_tier,
    "income_lpa": income_lpa,
    "occupation": occupation
    }

    try:
        response = requests.post(url, json = input_data)
        result = response.json()
        if "Insurance Premium Category" in result:
            st.success(f"Insurance Premium Category: {result['Insurance Premium Category']}")
        else:
            st.warning("Unexpected response format.")

    except Exception:
        st.error(Exception)
