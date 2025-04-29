import streamlit as st
import numpy as np
import tensorflow as tf
import os

# ===== LOAD TFLITE MODEL =====
model_path = "model_stroke_prediction.tflite"
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found.")
    st.stop()

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("Stroke Prediction App (with TFLite)")

# ===== INPUT USER =====
age = st.number_input('Age', min_value=0, max_value=120)
hypertension = st.selectbox('Hypertension (0=No, 1=Yes)', [0, 1])
heart_disease = st.selectbox('Heart Disease (0=No, 1=Yes)', [0, 1])
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0)
bmi = st.number_input('BMI', min_value=0.0)

# Gender
gender = st.selectbox('Gender', ['Female', 'Male', 'Other'])
gender_Male = 1 if gender == 'Male' else 0
gender_Other = 1 if gender == 'Other' else 0

# Marital status
ever_married = st.selectbox('Ever Married?', ['No', 'Yes'])
ever_married_Yes = 1 if ever_married == 'Yes' else 0

# Work type
work_type = st.selectbox('Work Type', ['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed'])
wt_Govt_job = 1 if work_type == 'Govt_job' else 0
wt_Never_worked = 1 if work_type == 'Never_worked' else 0
wt_Private = 1 if work_type == 'Private' else 0
wt_Self_employed = 1 if work_type == 'Self-employed' else 0

# Residence type
residence = st.selectbox('Residence Type', ['Rural', 'Urban'])
residence_Urban = 1 if residence == 'Urban' else 0

# Smoking status
smoking = st.selectbox('Smoking Status', ['Unknown', 'formerly smoked', 'never smoked', 'smokes'])
smoke_formerly = 1 if smoking == 'formerly smoked' else 0
smoke_never = 1 if smoking == 'never smoked' else 0
smoke_current = 1 if smoking == 'smokes' else 0

# ===== GABUNGKAN INPUT MENJADI ARRAY SESUAI URUTAN FITUR =====
input_data = np.array([[
    age, hypertension, heart_disease, avg_glucose_level, bmi,
    gender_Male, gender_Other,
    ever_married_Yes,
    wt_Govt_job, wt_Never_worked, wt_Private, wt_Self_employed,
    residence_Urban,
    smoke_formerly, smoke_never, smoke_current
]], dtype=np.float32)

# ===== PREDIKSI =====
if st.button("Predict"):
    if input_data.shape != tuple(input_details[0]['shape']):
        st.error(f"Shape mismatch: model expects {input_details[0]['shape']}, got {input_data.shape}")
        st.stop()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction = output[0][0]

    st.subheader("Prediction Result")
    st.write(f"Stroke Risk Score: **{prediction:.2f}**")
    if prediction > 0.5:
        st.error("High risk of stroke!")
    else:
        st.success("Low risk of stroke.")
