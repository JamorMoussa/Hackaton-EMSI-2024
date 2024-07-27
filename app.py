# Save this script as app.py
import streamlit as st
import numpy as np

from hack_emsi.models import Model, models_name

st.title("Crop Prediction Model")

st.subheader("Input Features")

features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

input_data = []
cols = st.columns(2)

for i, feature in enumerate(features):
    with cols[i % 2]:
        value = st.number_input(f"Enter {feature}", step=1.0)
        input_data.append(value)
    
with cols[len(features) % 2]:
    model_choice = st.selectbox("Choose the model", models_name)

input_data = np.array(input_data).reshape(1, -1)

if st.button("Predict"):
    
    model_id = Model.get_model_id(model_choice)

    model_name, model = Model.from_pretrained(model_id)

    prediction = model.predict(input_data)
    
    st.subheader("Prediction")
    
    st.write(f"The predicted label is: {prediction[0]}")

