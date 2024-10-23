import streamlit as st
import io
import os
import torch
from ultralytics import YOLO
from google.cloud import storage
from PIL import Image
import numpy as np
import cv2

# Initialize Google Cloud Storage client
client = storage.Client()
bucket_name = "nmfs_odp_pifsc"

# Define input and output GCS directories
input_folder_gcs = "PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/small_test_set/raw/"
output_images_gcs = "PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/small_test_set/test/images/"
output_labels_gcs = "PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/small_test_set/test/labels/"
verification_images_gcs = "PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/small_test_set/test/verification/"

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
    st.write(f"Uploaded {label_path} to GCS.")

# Function to save images to GCS
def save_image_to_gcs(bucket, image_path, img_array):
    _, img_encoded = cv2.imencode('.jpg', img_array)
    blob = bucket.blob(image_path)
    blob.upload_from_string(img_encoded.tobytes(), content_type='image/jpeg')
    st.write(f"Uploaded {image_path} to GCS.")

# Function to draw bounding boxes using YOLO's plot method and save for verification
def draw_and_save_verification_image(results, output_path):
    result_image = results[0].plot()
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    save_image_to_gcs(bucket, output_path, result_image)

# Function to process images, run inference, and save results
def process_images_from_gcs():
    blobs = client.list_blobs(bucket_name, prefix=input_folder_gcs)
    processed_count = 0
    
    for blob in blobs:
        if not blob.name.endswith(('.jpg', '.png')):
            continue
        
        img = read_image_from_gcs(blob)
        img_name = os.path.basename(blob.name)
        results = large_model(img)
        st.write(f"Processing {img_name}, found {len(results[0].boxes) if results[0].boxes is not None else 0} instances.")
        
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

    st.write("Dataset preparation complete!")

# Streamlit UI
st.title("YOLO Fish Detection - Streamlit App")
if st.button("Start Processing"):
    process_images_from_gcs()
