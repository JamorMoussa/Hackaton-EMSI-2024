# Save this script as app.py
import streamlit as st
import numpy as np
from hack_emsi.models import Model, models_name, LoadLabelEncoder

# Load the label encoder
enc_name, enc = LoadLabelEncoder.from_pretrained()

# Set the title and subtitle
st.title("Crop Prediction Model")
st.subheader("Input Features")

# Define the input features
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
input_data = []

# Create input fields for each feature
cols = st.columns(2)
for i, feature in enumerate(features):
    with cols[i % 2]:
        value = st.number_input(f"Enter {feature}", step=1.0)
        input_data.append(value)

# Create a dropdown for model selection
with cols[len(features) % 2]:
    model_choice = st.selectbox("Choose the model", models_name)

# Convert input data to a numpy array
input_data = np.array(input_data).reshape(1, -1)

# Make prediction when the button is clicked
if st.button("Predict"):
    model_id = Model.get_model_id(model_choice)
    model_name, model = Model.from_pretrained(model_id)
    prediction = model.predict(input_data)

    # Style the prediction output
    st.markdown(
        f"""
        <div style='background-color: #f9f9f9; padding: 10px; border-radius: 5px; text-align: center;'>
            <p style='font-size: 24px; color: #000;'>Recommended Crop</p>
            <h2 style='color: #4CAF50;'>{enc.classes_[prediction[0]]}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
