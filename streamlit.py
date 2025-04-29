import streamlit as st
import numpy as np
import tensorflow as tf
import os

# Pastikan model file .tflite tersedia
model_path = "model_stroke_prediction.tflite"
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please upload or place it in the correct directory.")
    st.stop()

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("Stroke Prediction App (TFLite)")

# Ambil input user
age = st.number_input('Age', min_value=0, max_value=120)
hypertension = st.selectbox('Hypertension (0=No, 1=Yes)', [0, 1])
heart_disease = st.selectbox('Heart Disease (0=No, 1=Yes)', [0, 1])
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0)
bmi = st.number_input('BMI', min_value=0.0)

if st.button('Predict'):
    input_data = np.array([[age, hypertension, heart_disease, avg_glucose_level, bmi]], dtype=np.float32)

    expected_shape = input_details[0]['shape']
    if input_data.shape != tuple(expected_shape):
        st.error(f"Shape mismatch: model expects {expected_shape}, got {input_data.shape}")
        st.stop()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction = output[0][0]

    st.write(f"Stroke Risk Score: **{prediction:.2f}**")
    if prediction > 0.5:
        st.error("High risk of stroke.")
    else:
        st.success("Low risk of stroke.")
