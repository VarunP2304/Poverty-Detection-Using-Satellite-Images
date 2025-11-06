import streamlit as st
import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import joblib

# --- 1. Load Pre-computed Model and Set Up ---
st.title("Poverty Prediction from Satellite Images")

@st.cache_resource
def load_prediction_pipeline():
    # Load the trained KMeans model
    model_path = "kmeans_model.joblib"
    kmeans_model = joblib.load(model_path)
    
    # --- Definitive fix: ensure model centers are float64 ---
    kmeans_model.cluster_centers_ = kmeans_model.cluster_centers_.astype(np.float64)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load ResNet50 for extracting features
    feature_extractor = models.resnet50(weights='IMAGENET1K_V1').to(device).eval()
    
    transform_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return kmeans_model, feature_extractor, transform_pipeline, device

with st.spinner('Loading prediction model...'):
    kmeans, model, transform, device = load_prediction_pipeline()
st.success("Model loaded successfully!")

classes = {0: "Poor", 1: "Rich", 2: "Middle Class"} 

# --- 2. Main Streamlit Application Logic ---
st.header("Upload an Image for Prediction")
uploaded_file = st.file_uploader("Choose a satellite image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Use 'width=None' for default handling, or 'width="stretch"' to fill container
    st.image(image, caption='Uploaded Image.', width=None)

    with st.spinner('Analyzing the image...'):
        img_array = np.array(image)

        if len(img_array.shape) < 3 or img_array.shape[2] == 1:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        img_transformed = transform(img_array).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feature = model(img_transformed)
            
        # --- Definitive fix: ensure input feature is float64 ---
        feature_final = feature.cpu().numpy().flatten().astype(np.float64)
        
        prediction = kmeans.predict([feature_final])[0]

    st.write(f"### Predicted Poverty Level: **{classes[prediction]}**")