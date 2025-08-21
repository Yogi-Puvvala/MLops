import streamlit as st
import joblib
import gdown
import pickle
import datetime
import numpy as np
import pandas as pd

st.header("Duration Calculation")

# Google Drive file ID (from your link)
FILE_ID = "15E2OcEtRy0u_wHUdB_AHoPg7uJAhk_x-"
url = f"https://drive.google.com/uc?id={FILE_ID}"


# Output file name
output = "Duration_Calculation_model.pkl"

# Download
gdown.download(url, output, quiet=False)

model = joblib.load(output)

# Independent features: trip_distance, total_amount, PULocationID, DOLocationID, time_sin, time_cos

def calculateDistance():
    if test_df.notnull().values.all():
        duration = model.predict(test_df)
        st.success(f"Predicted Duration: {round(duration[0])} minutes")
    else:
        st.error("Please fill in all fields before submitting.")

with st.form(key = "Duration_Calculation_Form"):
    trip_distance = st.number_input("Enter Trip Distance")
    total_amount = st.number_input("Enter Total Amount")
    PULocationID = st.number_input("Enter PickUp-LocationID")
    DOLocationID = st.number_input("Enter DropOff-LocationID")
    pickup_time = st.time_input("Enter PickUp Time")

    hour = pickup_time.hour
    minute = pickup_time.minute
    time_numeric = hour + minute / 60
    time_sin = np.sin(2* np.pi* time_numeric / 24)
    time_cos = np.cos(2 * np.pi * time_numeric / 24)

    test = {"trip_distance": [trip_distance],
        "total_amount": [total_amount],
        "PULocationID": [PULocationID],
        "DOLocationID": [DOLocationID],
        "time_sin": [time_sin],
        "time_cos": [time_cos]
    }

    test_df = pd.DataFrame(test)

    submitted = st.form_submit_button("Calculate Duration")

if submitted:
    calculateDistance()
