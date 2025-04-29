import streamlit as st
import numpy as np
import tensorflow as tf

# Load model (asumsi model ANN kamu sudah disimpan di file .h5)
model = tf.keras.models.load_model('model_stroke_prediction.h5')

# Judul halaman
st.title("Stroke Prediction App")

# Form inputan
age = st.number_input('Age', min_value=0, max_value=120)
hypertension = st.selectbox('Hypertension', [0, 1])
heart_disease = st.selectbox('Heart Disease', [0, 1])
avg_glucose_level = st.number_input('Average Glucose Level')
bmi = st.number_input('BMI')

# Tambahkan input lain sesuai kebutuhan dataset-mu

# Button untuk prediksi
if st.button('Predict'):
    input_data = np.array([[age, hypertension, heart_disease, avg_glucose_level, bmi]])
    prediction = model.predict(input_data)

    if prediction[0][0] > 0.5:
        st.error('High risk of stroke.')
    else:
        st.success('Low risk of stroke.')
