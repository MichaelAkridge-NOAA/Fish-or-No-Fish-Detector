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
from skimage.color import gray2rgb

st.set_page_config(
    page_title="Fish Detector",
    page_icon="🐟",
    layout="wide"
)

# Configure logging
logging.basicConfig(filename='yolo_fish_detection.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Google Cloud Storage client
client = storage.Client()

# Default GCS bucket name
DEFAULT_BUCKET_NAME = "nmfs_odp_pifsc"

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

# Function to read and preprocess images directly from GCS
def read_image_from_gcs(image_blob):
    try:
        img_bytes = image_blob.download_as_bytes()
        img = Image.open(io.BytesIO(img_bytes))

        # Ensure the image is in RGB format
        img = img.convert("RGB")

        # Convert to a NumPy array
        img_array = np.array(img)

        # Check if the image is not already in (H, W, 3) format
        if len(img_array.shape) == 2:  # Grayscale image
            img_array = gray2rgb(img_array)  # Use skimage to convert grayscale to RGB
        elif img_array.shape[-1] == 1:  # Single channel grayscale
            img_array = gray2rgb(img_array.squeeze())  # Squeeze to remove extra dimension and convert

        return img_array
    except Exception as e:
        logging.error(f"Failed to read image {image_blob.name}: {e}")
        return None

# Function to save labels to GCS
def save_yolo_labels_to_gcs(bucket, label_path, content):
    blob = bucket.blob(label_path)
    blob.upload_from_string(content)
    logging.info(f"Uploaded {label_path} to GCS.")

# Function to save images to GCS
def save_image_to_gcs(bucket, image_path, img_array):
    try:
        _, img_encoded = cv2.imencode('.jpg', img_array)
        blob = bucket.blob(image_path)
        blob.upload_from_string(img_encoded.tobytes(), content_type='image/jpeg')
        logging.info(f"Uploaded {image_path} to GCS.")
    except Exception as e:
        logging.error(f"Failed to save image {image_path} to GCS: {e}")

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
            if img is not None:
                st.image(img, caption=os.path.basename(blob.name), use_column_width=True)

# Function to process images, run inference, and save results
def process_images_from_gcs(bucket_name, input_folder_gcs, output_images_gcs, output_labels_gcs, verification_images_gcs, confidence, max_display_count=5):
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket_name, prefix=input_folder_gcs)
    processed_count = 0
    display_count = 0
    
    for blob in blobs:
        if not blob.name.endswith(('.jpg', '.png')):
            continue
        
        img = read_image_from_gcs(blob)
        if img is None:
            continue
        
        # Convert to OpenCV-compatible format (BGR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        img_name = os.path.basename(blob.name)
        
        with st.spinner(f'Processing {img_name}...'):
            try:
                # Add a batch dimension if the model expects a batch
                img_batch = np.expand_dims(img, axis=0)  # Add batch dimension
                results = large_model(img_batch, conf=confidence)
            except Exception as e:
                st.error(f"Failed to process {img_name}: {e}")
                logging.error(f"Failed to process {img_name}: {e}")
                continue
        
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

        # Display side-by-side images (Limit to max_display_count)
        if display_count < max_display_count:
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption=f"Original Image - {img_name}", use_column_width=True)
            with col2:
                processed_img = results[0].plot()
                st.image(processed_img, caption=f"Detection Results - {img_name}", use_column_width=True)
            display_count += 1

    st.success("🎉 Dataset preparation complete!")
    display_verification_images(verification_images_gcs)

# Streamlit UI
st.title("🐟 Google Cloud Fish Detector - NODD App")

# Add description with links to the repository and model
st.markdown("""
**Welcome to the Google Cloud Fish Detector - NODD App!**
Identify fish in images stored on Google Cloud. 

🔗 **[GitHub Repository](https://github.com/MichaelAkridge-NOAA/Fish-or-No-Fish-Detector/)**  
🧠 **[YOLO11 Fish Detector Model on Hugging Face](https://huggingface.co/akridge/yolo11-fish-detector-grayscale)**
""")

# Sidebar configuration
st.sidebar.title("🐟 Fish Detection Settings")
st.sidebar.markdown("""
For more information:
- Contact: Michael.Akridge@NOAA.gov
- Visit the [GitHub repository](https://github.com/MichaelAkridge-NOAA/Fish-or-No-Fish-Detector/)
""")
confidence = st.sidebar.slider("Detection Confidence Threshold", 0.0, 1.0, 0.15)

# Bucket name input with default value
bucket_name = st.sidebar.text_input("🪣 GCS Bucket Name", DEFAULT_BUCKET_NAME)

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
        process_images_from_gcs(bucket_name, input_folder_gcs, output_images_gcs, output_labels_gcs, verification_images_gcs, confidence, max_display_count=5)

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
