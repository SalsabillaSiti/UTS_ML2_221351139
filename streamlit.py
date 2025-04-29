import streamlit as st
import numpy as np
import tensorflow as tf
import os
if not os.path.exists("model_stroke.tflite"):
    st.error("Model file 'model_stroke.tflite' not found. Please upload or place it in the correct directory.")
    print("File exists:", os.path.exists("model_stroke.tflite"))

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="models/model_stroke.tflite")
interpreter.allocate_tensors()

# Ambil detail input/output tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul aplikasi
st.title("Stroke Prediction with TFLite")

# Form input
age = st.number_input('Age', min_value=0, max_value=120)
hypertension = st.selectbox('Hypertension (0=No, 1=Yes)', [0, 1])
heart_disease = st.selectbox('Heart Disease (0=No, 1=Yes)', [0, 1])
avg_glucose_level = st.number_input('Average Glucose Level')
bmi = st.number_input('BMI')

if st.button("Predict"):
    # Buat input data array, sesuaikan jumlah fiturnya dengan model
    input_data = np.array([[age, hypertension, heart_disease, avg_glucose_level, bmi]], dtype=np.float32)

    # Debug info (optional)
    st.write("Expected input shape from model:", input_details[0]['shape'])
    st.write("Input data shape:", input_data.shape)

    # Set input dan jalankan model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Ambil output hasil prediksi
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0][0]  # asumsi output scalar

    # Tampilkan hasil
    st.subheader("Prediction Result:")
    st.write(f"Stroke Risk Score: **{prediction:.2f}**")
    if prediction > 0.5:
        st.error("High risk of stroke!")
    else:
        st.success("Low risk of stroke.")
