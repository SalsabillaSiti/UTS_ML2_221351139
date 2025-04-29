import streamlit as st
import numpy as np
import tensorflow as tf
import os

# Load model TFLite
model_path = os.path.join(os.path.dirname(__file__), "model_stroke_prediction.tflite")
interpreter = tf.lite.Interpreter(model_path=model_path)

# Ambil detail input/output tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("Stroke Prediction with TFLite")

# Form input
age = st.number_input('Age', min_value=0, max_value=120)
hypertension = st.selectbox('Hypertension', [0, 1])
heart_disease = st.selectbox('Heart Disease', [0, 1])
avg_glucose_level = st.number_input('Average Glucose Level')
bmi = st.number_input('BMI')

if st.button("Predict"):
    # Buat data input sesuai model (misal 5 fitur)
    input_data = np.array([[age, hypertension, heart_disease, avg_glucose_level, bmi]], dtype=np.float32)

    # Set input ke model
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Jalankan prediksi
    interpreter.invoke()

    # Ambil output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0][0]

    if prediction > 0.5:
        st.error(f"High risk of stroke ({prediction:.2f})")
    else:
        st.success(f"Low risk of stroke ({prediction:.2f})")
