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

# Configure logging
logging.basicConfig(filename='yolo_fish_detection.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Google Cloud Storage client
client = storage.Client()
bucket_name = "nmfs_odp_pifsc"

# Default input and output GCS directories
#DEFAULT_INPUT_FOLDER_GCS = "PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/small_test_set/raw/"
#DEFAULT_OUTPUT_IMAGES_GCS = "PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/small_test_set/test/images/"
#DEFAULT_OUTPUT_LABELS_GCS = "PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/small_test_set/test/labels/"
#DEFAULT_VERIFICATION_IMAGES_GCS = "PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/small_test_set/test/verification/"
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

# Function to read images directly from GCS
def read_image_from_gcs(image_blob):
    img_bytes = image_blob.download_as_bytes()
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img)

# Function to save labels to GCS
def save_yolo_labels_to_gcs(bucket, label_path, content):
    blob = bucket.blob(label_path)
    blob.upload_from_string(content)
    logging.info(f"Uploaded {label_path} to GCS.")

# Function to save images to GCS
def save_image_to_gcs(bucket, image_path, img_array):
    _, img_encoded = cv2.imencode('.jpg', img_array)
    blob = bucket.blob(image_path)
    blob.upload_from_string(img_encoded.tobytes(), content_type='image/jpeg')
    logging.info(f"Uploaded {image_path} to GCS.")

# Function to draw bounding boxes using YOLO's plot method and save for verification
def draw_and_save_verification_image(results, output_path):
    result_image = results[0].plot()
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    save_image_to_gcs(bucket, output_path, result_image)

# Function to display verification images from GCS
def display_verification_images(verification_images_gcs):
    st.write("Verification Images:")
    blobs = client.list_blobs(bucket_name, prefix=verification_images_gcs)
    for blob in blobs:
        if blob.name.endswith(('.jpg', '.png')):
            img = read_image_from_gcs(blob)
            st.image(img, caption=os.path.basename(blob.name), use_column_width=True)

# Function to process images, run inference, and save results
def process_images_from_gcs(input_folder_gcs, output_images_gcs, output_labels_gcs, verification_images_gcs):
    blobs = client.list_blobs(bucket_name, prefix=input_folder_gcs)
    processed_count = 0
    
    for blob in blobs:
        if not blob.name.endswith(('.jpg', '.png')):
            continue
        
        img = read_image_from_gcs(blob)
        img_name = os.path.basename(blob.name)
        results = large_model(img)
        logging.info(f"Processing {img_name}, found {len(results[0].boxes) if results[0].boxes is not None else 0} instances.")
        
        output_image_path = f"{output_images_gcs}{img_name}"
        save_image_to_gcs(bucket, output_image_path, img)
        
        img_height, img_width = img.shape[:2]
        fish_class_index = 0
        labels_content = ""
        
        if results[0].boxes is not None:
            for i, cls in enumerate(results[0].boxes.cls.cpu().numpy()):
                if cls == fish_class_index:
                    bbox = results[0].boxes.xyxy[i].cpu().numpy()
                    x_center = (bbox[0] + bbox[2]) / 2 / img_width
                    y_center = (bbox[1] + bbox[3]) / 2 / img_height
                    width = (bbox[2] - bbox[0]) / img_width
                    height = (bbox[3] - bbox[1]) / img_height
                    labels_content += f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        
        if labels_content:
            label_path = f"{output_labels_gcs}{img_name.replace('.jpg', '.txt').replace('.png', '.txt')}"
            save_yolo_labels_to_gcs(bucket, label_path, labels_content)

        if processed_count < 5:
            verification_image_path = f"{verification_images_gcs}{img_name}"
            draw_and_save_verification_image(results, verification_image_path)
            processed_count += 1

    st.success("🎉 Dataset preparation complete!")
    display_verification_images(verification_images_gcs)

# Streamlit UI
st.title("🐟 Google Cloud Fish Detector - NODD App")

# Add description with links to the repository and model
st.markdown("""
**Welcome to the Google Cloud Fish Detector - NODD App!**
This application leverages advanced object detection models to identify fish in images stored on Google Cloud. 

🔗 **[GitHub Repository](https://github.com/MichaelAkridge-NOAA/Fish-or-No-Fish-Detector/tree/MOUSS_2016/google-cloud-shell)**  
🧠 **[YOLOv11 Fish Detector Model on Hugging Face](https://huggingface.co/akridge/yolo11-fish-detector-grayscale)**
""")

# Use columns for better layout
col1, col2 = st.columns(2)

with col1:
    # User inputs for GCS paths
    input_folder_gcs = st.text_input("📂 Input Folder GCS Path", DEFAULT_INPUT_FOLDER_GCS)
    output_images_gcs = st.text_input("🖼️ Output Images GCS Path", DEFAULT_OUTPUT_IMAGES_GCS)

with col2:
    output_labels_gcs = st.text_input("📝 Output Labels GCS Path", DEFAULT_OUTPUT_LABELS_GCS)
    verification_images_gcs = st.text_input("✅ Verification Images GCS Path", DEFAULT_VERIFICATION_IMAGES_GCS)

# Add a button in an expanded section
with st.expander("🔄 Start Processing"):
    if st.button("🚀 Process Images"):
        process_images_from_gcs(input_folder_gcs, output_images_gcs, output_labels_gcs, verification_images_gcs)
