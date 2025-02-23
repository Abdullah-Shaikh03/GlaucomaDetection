import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from Model import HybridNet
import plotly.graph_objects as go
from io import BytesIO
import matplotlib.pyplot as plt
from skimage import filters, measure
from scipy.ndimage import binary_fill_holes

def calculate_cdr(image):
    """
    Calculate Cup-to-Disc Ratio from fundus image
    Returns CDR value and segmented image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Detect optic disc
    disc_thresh = filters.threshold_otsu(enhanced)
    disc_mask = enhanced > disc_thresh
    disc_mask = binary_fill_holes(disc_mask)
    
    # Find the largest connected component (optic disc)
    labels = measure.label(disc_mask)
    regions = measure.regionprops(labels)
    if not regions:
        return 0, None, None
    
    disc_region = max(regions, key=lambda x: x.area)
    disc_mask = labels == disc_region.label
    disc_area = disc_region.area
    
    # Detect cup region (brighter part within disc)
    cup_thresh = filters.threshold_otsu(enhanced[disc_mask])
    cup_mask = enhanced > cup_thresh
    cup_mask = cup_mask & disc_mask
    cup_mask = binary_fill_holes(cup_mask)
    
    # Calculate cup area
    cup_area = np.sum(cup_mask)
    
    # Calculate CDR
    cdr = cup_area / disc_area if disc_area > 0 else 0
    
    # Create visualization
    visualization = np.zeros((*gray.shape, 3), dtype=np.uint8)
    visualization[disc_mask] = [0, 255, 0]  # Green for disc
    visualization[cup_mask] = [255, 0, 0]   # Red for cup
    
    return cdr, visualization, {'disc_area': disc_area, 'cup_area': cup_area}

def classify_cdr(cdr):
    """
    Classify glaucoma risk based on CDR value
    """
    if cdr < 0.4:
        return "Normal", "Low Risk"
    elif 0.4 <= cdr < 0.7:
        return "Suspicious", "Medium Risk"
    else:
        return "Abnormal", "High Risk"

def load_model(model_path):
    """Load the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridNet(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()
    return model, device

def preprocess_image(image):
    """Preprocess the input image for model prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return preprocess(image)

def get_prediction(model, image_tensor, device):
    """Get model prediction and confidence score"""
    with torch.no_grad():
        model = model.to(device)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        confidence = torch.max(probabilities).item()
    
    return prediction.item(), confidence

def main():
    st.set_page_config(page_title="Glaucoma Detection System", layout="wide")
    
    st.title("Glaucoma Detection System")
    st.markdown("""
    This application analyzes fundus images for glaucoma detection using:
    1. Deep Learning Model Analysis
    2. Cup-to-Disc Ratio (CDR) Calculation
    """)
    
    st.sidebar.title("About")
    st.sidebar.info("""
    Hybrid analysis system combining:
    - Deep Learning (Accuracy: 90.13%)
    - CDR Analysis
    - Clinical Risk Assessment
    """)
    
    uploaded_file = st.file_uploader("Choose a fundus image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and process image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Calculate CDR
        cdr, cdr_visualization, measurements = calculate_cdr(image_np)
        cdr_status, risk_level = classify_cdr(cdr)
        
        # Model prediction
        model, device = load_model("../models/best_model.pth")
        image_tensor = preprocess_image(image)
        prediction, confidence = get_prediction(model, image_tensor, device)
        
        # Display results in columns
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Fundus Image", use_column_width=True)
        
        with col2:
            st.subheader("CDR Analysis")
            st.image(cdr_visualization, caption="Cup (Red) and Disc (Green) Segmentation", use_column_width=True)
        
        with col3:
            st.subheader("Measurements")
            st.metric("Cup-to-Disc Ratio (CDR)", f"{cdr:.3f}")
            st.metric("CDR Status", cdr_status)
            st.metric("Risk Level", risk_level)
        
        # Combined Analysis Results
        st.subheader("Combined Analysis Results")
        
        # Create two columns for deep learning and CDR results
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.markdown("### Deep Learning Analysis")

            result = "Glaucoma Detected" if prediction == 1 else "No Glaucoma Detected"
            confidence_percent = confidence * 100
            
            fig1 = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence_percent,
                title = {'text': f"Model Confidence"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "gray"},
                        {'range': [75, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            st.plotly_chart(fig1)
        
        with result_col2:
            st.markdown("### CDR Analysis")
            fig2 = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = cdr * 100,
                title = {'text': f"Cup-to-Disc Ratio"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            st.plotly_chart(fig2)
        
        # Final Assessment
        st.subheader("Final Assessment")
        
        # Combine both analyses for final assessment
        dl_risk = "High" if confidence_percent > 90 else "Medium" if confidence_percent > 70 else "Low"
        final_risk = "High" if (dl_risk == "High" and risk_level == "High Risk") else \
                    "Low" if (dl_risk == "Low" and risk_level == "Low Risk") else "Medium"
        
        st.markdown(f"""
        ### Risk Factors:
        1. **Deep Learning Analysis**: {result} (Confidence: {confidence_percent:.1f}%)
        2. **CDR Analysis**: {cdr_status} (CDR: {cdr:.3f})
        3. **Overall Risk Level**: {final_risk}
        
        **Important Notice**: 
        - This is an automated screening tool and should not replace professional medical diagnosis.
        - Please consult with an ophthalmologist for proper diagnosis and treatment.
        - CDR is one of many factors in glaucoma diagnosis and should be considered alongside other clinical findings.
        """)

if __name__ == "__main__":
    main()