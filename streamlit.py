import streamlit as st
import numpy as np
import tensorflow as tf

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="model_stroke_prediction.tflite")
interpreter.allocate_tensors()

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
    # Buat data input sesuai model
    input_data = np.array([[age, hypertension, heart_disease, avg_glucose_level, bmi]], dtype=np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Ambil hasil prediksi
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0][0]

    # Tampilkan hasil
    st.write("Prediction result:", round(prediction, 2))
    if prediction > 0.5:
        st.error("High risk of stroke!")
    else:
        st.success("Low risk of stroke.")
