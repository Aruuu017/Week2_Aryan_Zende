import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
try:
    with open("crop_model.pkl", "rb") as file:
        model = pickle.load(file)
    
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

except FileNotFoundError:
    st.error("❌ Model or Scaler file not found. Please train and save them first.")
    st.stop()

# Streamlit UI
st.title("🌱 Crop Recommendation System")
st.write("Enter soil and weather conditions to get the best crop recommendation.")

# User Input Fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=100, value=50)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=100, value=50)
K = st.number_input("Potassium (K)", min_value=0, max_value=100, value=50)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)

# Predict Button
if st.button("Recommend Crop"):
    # Create numpy array from input (only N, P, K, Temperature)
    input_features = np.array([[N, P, K, temperature]])

    # Apply scaling using the loaded StandardScaler
    input_scaled = scaler.transform(input_features)

    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Display the recommended crop
    st.success(f"✅ Recommended Crop: {prediction[0]} 🌾")
