import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Configure the page
st.set_page_config(page_title="Diabetes Prediction", page_icon="⚕", layout="wide")
st.title("Diabetes Prediction App")

# Apply custom CSS for styling
st.markdown("""
    <style>
    .centered-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        margin-top: 50px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 24px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# Centered container for inputs
with st.container():
    st.markdown("<div class='centered-container'>", unsafe_allow_html=True)
    st.markdown("### Enter Patient Information")

    pregnancies = st.number_input("Pregnancies", min_value=0, value=0)
    glucose = st.number_input("Glucose Level", min_value=0.0, value=120.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=70.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
    insulin = st.number_input("Insulin Level", min_value=0.0, value=80.0)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.0)
    age = st.number_input("Age", min_value=0, value=30)
    st.markdown("</div>", unsafe_allow_html=True)

# Prediction button and result display
if st.button("Predict"):
    input_data = scaler.transform([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.markdown("<h3 style='color: red; text-align: center;'>The patient is likely to have diabetes.</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: green; text-align: center;'>The patient is unlikely to have diabetes.</h3>", unsafe_allow_html=True)

# Add additional notes
st.markdown("""
    <div style='text-align: center; margin-top: 50px;'>
    *Important Notes:*
    - Glucose Level refers to the concentration of glucose in the blood.
    - Blood Pressure indicates the force of blood against the walls of your arteries.
    - Skin Thickness is measured at the tricep area to estimate body fat.
    - Insulin level shows the amount of insulin in the blood.
    - BMI is a measure of body fat based on height and weight.
    - Diabetes Pedigree Function indicates the genetic predisposition to diabetes.
    - Age plays a crucial role in the likelihood of diabetes.
    </div>
    """, unsafe_allow_html=True)