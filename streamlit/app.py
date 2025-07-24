import streamlit as st
import torch
import numpy as np
import pickle
from mlc_model import ANNModel

# Load scaler and model once at start
with open(r'E:\vs_code_dsa\ANN\ANN_Multi_class_Classification\scaler.pickle', 'rb') as f:
    scaler_var = pickle.load(f)

model = ANNModel(input_size=11, output_size=3)
model.load_state_dict(torch.load(r'E:\vs_code_dsa\ANN\ANN_Multi_class_Classification\model.pth'))
model.eval()

st.title("Wine Quality Input Form")
st.write("Enter the features below to predict wine quality")

# Sliders for features your model expects (10 features)
fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.4, 0.1)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 2.0, 0.7, 0.01)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.0, 0.01)
residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 1.9, 0.1)
chlorides = st.slider("Chlorides", 0.01, 0.2, 0.076, 0.001)
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1.0, 50.0, 11.0, 1.0)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 10.0, 200.0, 34.0, 1.0)
density = st.slider("Density", 0.9900, 1.0100, 0.9978, 0.0001, format="%.4f")
pH = st.slider("pH", 2.5, 4.5, 3.51, 0.01)
sulphates = st.slider("Sulphates", 0.3, 2.0, 0.56, 0.01)
alcohol = st.slider("Alcohol", 7.0, 15.0, 9.4, 0.1)

if st.button("Submit"):
    # Prepare input list in exact feature order your model was trained on
    features = [
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]
    
    # If alcohol was part of training input, include it, and update input_size=11 for model
    # For now, excluding alcohol since input_size=10
    
    # Convert to numpy array, reshape for scaler (1 sample, n features)
    features_np = np.array(features).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler_var.transform(features_np)
    
    # Convert to torch tensor
    input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
    
    predicted_class_idx = torch.argmax(outputs, dim=1).item()
    result_map = { 0:'Average', 1: 'High', 2 :'Low'}
    st.write("Prediction (class index):", result_map.get(predicted_class_idx))
    
    # Optionally map to class name if you have mapping
    
    st.write("Scaled Input Features:")
    st.write(features_scaled)
