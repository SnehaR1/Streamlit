import pandas as pd
import numpy as np
import streamlit as st
from prediction import predict
st.title("Diabetes Prediction")
st.header("Hello...Welcome to diabetes prediction forum!")
st.markdown("Please Enter The Following Informations")
col1, col2 = st.columns(2)


with col1:
    Age = st.slider("Enter your Age", 21, 81)
    BMI = st.slider("BMI", 0, 68)
    pregnancies = st.slider("Pregnancies", 0, 20)
    SkinThickness = st.slider("SkinThickness", 0, 99)

with col2:
    BloodPressure = st.slider("BloodPressure", 0, 180)
    Glucose = st.slider("Glucose Level", 0, 200)
    Insulin = st.slider("Insulin", 0, 180)
    DiabetesPedigreeFunction = st.slider(
        "DiabetesPedigreeFunction", 0.08, 2.42)

if st.button("CHECK RESULTS"):
    result = predict(
        np.array([[pregnancies, Glucose, SkinThickness, BloodPressure, DiabetesPedigreeFunction, Insulin, BMI, Age]]))
    if result == 0:
        st.text(" YOU ARE NOT DIABETIC...CONGRATULATIONS!!!")
    else:
        st.text("SORRY...,YOU ARE DIABETIC")
