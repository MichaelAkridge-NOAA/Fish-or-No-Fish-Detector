import streamlit as st
import io
import os
import torch
import logging
from ultralytics import YOLO
from google.cloud import storage
from PIL import Image
import numpy as np
import cv2
import tempfile

st.set_page_config(
    page_title="Fish Detector",
    page_icon="🐟",
    layout="wide"
)

# Configure logging
logging.basicConfig(filename='yolo_fish_detection.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Google Cloud Storage client
client = storage.Client()
bucket_name = "nmfs_odp_pifsc"

# Default input and output GCS directories
DEFAULT_INPUT_FOLDER_GCS = "PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/large_2016_dataset/raw/"
DEFAULT_OUTPUT_IMAGES_GCS = "PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/large_2016_dataset/images/"
DEFAULT_OUTPUT_LABELS_GCS = "PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/large_2016_dataset/labels/"
DEFAULT_VERIFICATION_IMAGES_GCS = "PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/large_2016_dataset/verification/"

# Check if CUDA is available and load the large model (YOLOv8x) to CUDA if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"Using device: {device}")

# Load the YOLO model from the downloaded location
large_model = YOLO("/app/yolov8n_fish_trained.pt")
large_model = large_model.to(device)

# Define GCS bucket
bucket = client.bucket(bucket_name)

# Function to read images directly from GCS and save to a temporary file
def read_image_from_gcs_and_save(image_blob):
    try:
        img_bytes = image_blob.download_as_bytes()
        img = Image.open(io.BytesIO(img_bytes))

        # Create a temporary file that will be automatically deleted
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            img.save(temp_file.name)
            temp_file_path = temp_file.name
        return temp_file_path
    except Exception as e:
        logging.error(f"Failed to read image {image_blob.name}: {e}")
        return None

# Function to process images, run inference, and save results
def process_images_from_gcs(input_folder_gcs, output_images_gcs, output_labels_gcs, verification_images_gcs, confidence, max_display_count=5):
    blobs = client.list_blobs(bucket_name, prefix=input_folder_gcs)
    processed_count = 0
    display_count = 0
    
    for blob in blobs:
        if not blob.name.endswith(('.jpg', '.png')):
            continue
        
        # Read and save the image to a temporary file, then pass the path to the model
        temp_image_path = read_image_from_gcs_and_save(blob)
        if temp_image_path is None:
            logging.error(f"Failed to download and save {blob.name}")
            continue
        
        img_name = os.path.basename(blob.name)
        
        try:
            # Check if the image is valid
            image = cv2.imread(temp_image_path)
            if image is None or image.shape[0] == 0 or image.shape[1] == 0:
                logging.error(f"Invalid image dimensions for {img_name}")
                continue

            with st.spinner(f'Processing {img_name}...'):
                # Use the temporary file path for inference
                results = large_model.predict(temp_image_path, conf=confidence)

            logging.info(f"Processing {img_name}, found {len(results[0].boxes) if results[0].boxes is not None else 0} instances.")
            
            # Display side-by-side images (Limit to max_display_count)
            if display_count < max_display_count:
                # Load the original image to show it in Streamlit
                original_img = Image.open(temp_image_path)
                processed_img = results[0].plot()
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_img, caption=f"Original Image - {img_name}", use_column_width=True)
                with col2:
                    st.image(processed_img, caption=f"Detection Results - {img_name}", use_column_width=True)
                display_count += 1

        except cv2.error as e:
            logging.error(f"OpenCV error while processing {img_name}: {e}")
            st.error(f"Failed to process {img_name}: {e}")

        except Exception as e:
            logging.error(f"Failed to process {img_name}: {e}")
            st.error(f"Failed to process {img_name}")

        finally:
            # Remove the temporary file to free up space
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    st.success("🎉 Dataset preparation complete!")
    display_verification_images(verification_images_gcs)

# Function to display verification images from GCS
def display_verification_images(verification_images_gcs):
    st.write("Verification Images:")
    blobs = client.list_blobs(bucket_name, prefix=verification_images_gcs)
    for blob in blobs:
        if blob.name.endswith(('.jpg', '.png')):
            img = read_image_from_gcs(blob)
            if img is not None:
                st.image(img, caption=os.path.basename(blob.name), use_column_width=True)

# Streamlit UI
st.title("🐟 Google Cloud Fish Detector - NODD App")

# Add description with links to the repository and model
st.markdown("""
**Welcome to the Google Cloud Fish Detector - NODD App!**
This application leverages advanced object detection models to identify fish in images stored on Google Cloud. 

🔗 **[GitHub Repository](https://github.com/MichaelAkridge-NOAA/Fish-or-No-Fish-Detector/tree/MOUSS_2016/google-cloud-shell)**  
🧠 **[YOLOv11 Fish Detector Model on Hugging Face](https://huggingface.co/akridge/yolo11-fish-detector-grayscale)**
""")

# Sidebar configuration
st.sidebar.title("🐟 Fish Detection Settings")
confidence = st.sidebar.slider("Detection Confidence Threshold", 0.0, 1.0, 0.15)

# Use columns for better layout
col1, col2 = st.columns(2)
with col1:
    input_folder_gcs = st.text_input("📂 Input Folder GCS Path", DEFAULT_INPUT_FOLDER_GCS)
    output_images_gcs = st.text_input("🖼️ Output Images GCS Path", DEFAULT_OUTPUT_IMAGES_GCS)
with col2:
    output_labels_gcs = st.text_input("📝 Output Labels GCS Path", DEFAULT_OUTPUT_LABELS_GCS)
    verification_images_gcs = st.text_input("✅ Verification Images GCS Path", DEFAULT_VERIFICATION_IMAGES_GCS)

# Start processing button
with st.expander("🔄 Start Processing"):
    if st.button("🚀 Process Images"):
        process_images_from_gcs(input_folder_gcs, output_images_gcs, output_labels_gcs, verification_images_gcs, confidence, max_display_count=5)

# Apply custom CSS for improved styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        padding: 10px;
        border-radius: 5px;
        font-size: 18px;
        font-weight: bold;
        background-color: #007BFF;
        color: white;
        border: none;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)
