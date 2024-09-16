# Import required modules
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import os
import json
import logging
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration and sample image path
sample_images_folder = "./images/sample_images"
st.set_page_config(
    page_title="Fish Detector",
    page_icon="🐟",
    layout="wide"
)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load the Faster R-CNN model
model_path = os.getenv("MODEL_PATH", "./models/custom_resnet_fasterrcnn_2.pt")
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please check your setup.")
    st.stop()

# Load the full Faster R-CNN model
model = torch.load(model_path)
model.eval()  # Set the model to evaluation mode

# Define image preprocessing transforms for Faster R-CNN
preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor (Faster R-CNN expects a tensor)
])

# Function to load sample images
def load_sample_images():
    sample_images = []
    for filename in os.listdir(sample_images_folder):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(sample_images_folder, filename)
            sample_images.append(image_path)
    return sample_images

# Title and description for Streamlit app
st.title("🐟 Fish or No Fish Detector")
st.write("""
Is there a fish 🐟 or not? Upload one or more images to detect vulnerable marine ecosystems (corals, crinoids, sponges, and fish). 

Uses the [**FathomNet VME Model**](https://huggingface.co/FathomNet/vulnerable-marine-ecosystems). 
Based on a custom Faster R-CNN model for object detection.  
""")

# Sidebar setup
st.sidebar.header("Models Parameter Settings")
confidence = st.sidebar.slider("Detection Confidence Threshold", 0.0, 1.0, 0.5)

st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.image("./images/logos/nmfs-opensci-logo3.png", width=100, caption="NOAA Open Science")
with col2:
    st.image("./images/logos/FathomNet_black_bottomText_400px.png", width=100, caption="FathomNet")

# Optional font for labels
try:
    font = ImageFont.truetype("arial.ttf", 20)
except IOError:
    font = ImageFont.load_default()

# Function to map labels to meaningful categories
LABELS_MAP = {
    1: "Fish",  # Assuming label 1 is for fish, adjust according to your data
    # Add more labels here as needed
}

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, labels, scores):
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        # Map the label to a meaningful name
        label_text = LABELS_MAP.get(label, f"Label {label}")
        # Use a thicker red line for bounding boxes
        draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=5)
        # Add the label and score with a larger font
        draw.text((box[0], box[1] - 10), f"{label_text}: {score:.2f}", fill="yellow", font=font)

# Define the prediction function to return bounding box data and number of fish detections
def run(image_path):
    logging.info(f"Running model prediction on {image_path}")
    try:
        # Load and preprocess image
        image = Image.open(image_path)
        
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        img_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

        # Perform inference using Faster R-CNN
        with torch.no_grad():
            outputs = model(img_tensor)

        # Extract bounding box data
        boxes = outputs[0]['boxes'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()

        # Filter detections by confidence threshold
        valid_boxes = boxes[scores >= confidence]
        valid_labels = labels[scores >= confidence]
        valid_scores = scores[scores >= confidence]

        fish_count = 0
        metadata = []
        for i, (box, label, score) in enumerate(zip(valid_boxes, valid_labels, valid_scores)):
            if label == 0:  # Assuming "Fish" is label 0
                fish_count += 1
            metadata.append({
                "box": box.tolist(),    # Convert NumPy array to list
                "label": int(label),    # Convert np.int64 to int
                "score": float(score)   # Convert np.float32 to float
            })

        # Draw bounding boxes on the image
        draw_boxes(image, valid_boxes, valid_labels, valid_scores)

        # Return processed image and metadata
        return image, {"fish_count": fish_count, "boxes": metadata}

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        st.error(f"An error occurred during prediction: {e}")
        return None, None

# Reusable function to handle multiple image uploads and display results
def process_images(uploaded_files):
    all_detections = []
    result_images = []
    summary_data = []
    temp_dir = tempfile.gettempdir()

    for uploaded_file in uploaded_files:
        if isinstance(uploaded_file, str):  # Check if it's a sample image path
            image_path = uploaded_file
            image = Image.open(image_path)
        else:
            image = Image.open(uploaded_file)
            image_path = os.path.join(temp_dir, f"{uploaded_file.name}")
            image.save(image_path)

        st.write(f"Detecting in {os.path.basename(image_path)}...")
        with st.spinner('Running detection...'):
            result_image, detection_metadata = run(image_path)

        if result_image is not None:
            result_images.append((result_image, os.path.basename(image_path)))
            all_detections.append(detection_metadata)

            summary_data.append({
                "image_name": os.path.basename(image_path),
                "fish_detected": detection_metadata["fish_count"] > 0,
                "fish_count": detection_metadata["fish_count"]
            })

            # Display images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.image(result_image, caption=f"Detection Results - {os.path.basename(image_path)}", use_column_width=True)

            st.success(f"Detection completed for {os.path.basename(image_path)} successfully! 🐟")

        else:
            st.warning(f"No marine ecosystems detected in {os.path.basename(image_path)}.")

    st.session_state["all_detections"] = all_detections
    return summary_data

# Function to display a summary table
def display_summary(summary_data):
    if summary_data:
        df = pd.DataFrame(summary_data)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Summary of Detections")
            st.table(df[["image_name", "fish_count"]])

        if st.session_state.get("all_detections"):
            json_data = json.dumps(st.session_state["all_detections"], indent=4)
            st.download_button(
                label="Download Results as JSON & Reset",
                data=json_data,
                file_name="all_detections.json",
                mime="application/json",
                key="download_json_bottom"
            )

# Image uploader with multiple file support
st.markdown('<div class="custom-file-uploader">', unsafe_allow_html=True)
uploaded_files = st.file_uploader("Choose image(s)...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Add the functionality for the "Try it with Sample Images" button
use_sample_images = st.button("Or Auto Run Using Sample Images", key="sample_button")
if use_sample_images:
    sample_images = load_sample_images()
    st.session_state['use_sample_images'] = True
    for sample_image in sample_images:
        st.session_state.setdefault('uploaded_files', []).append(sample_image)
    st.session_state['run_automatically'] = True

# Display the Run, Clear, and Download buttons with enhanced styling
if uploaded_files or st.session_state.get('uploaded_files'):
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")

    with col1:
        run_button = st.button("Click to Run", key="run_button")

    # Run the detection
    if run_button:
        summary_data = process_images(uploaded_files or st.session_state['uploaded_files'])
        display_summary(summary_data)

    # Clear the results
    if st.button("Clear Results", key="clear_button"):
        st.session_state.clear()

    if st.session_state.get("all_detections"):
        with col3:
            json_data = json.dumps(st.session_state["all_detections"], indent=4)
            st.download_button(
                label="Download Results as JSON & Reset",
                data=json_data,
                file_name="all_detections.json",
                mime="application/json",
                key="download_json"
            )
