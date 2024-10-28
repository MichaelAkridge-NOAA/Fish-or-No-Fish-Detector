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
from datetime import datetime
import sqlite3
import uuid

st.set_page_config(
    page_title="Fish Detector",
    page_icon="🐟",
    layout="wide"
)

# Configure logging to use StreamHandler
log_stream = io.StringIO()
logging.basicConfig(stream=log_stream, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Google Cloud Storage client
client = storage.Client()
bucket_name = "nmfs_odp_pifsc"

# Default input and output GCS directories
DEFAULT_INPUT_FOLDER_GCS = "PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/large_2016_dataset/raw/"
DEFAULT_OUTPUT_IMAGES_GCS = "PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/large_2016_dataset/datasetv2/images/"
DEFAULT_OUTPUT_LABELS_GCS = "PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/large_2016_dataset/datasetv2/labels/"

# SQLite database file
DB_FILE = "processed_images.db"
BACKUP_INTERVAL = 1000  # Sync the database every 1,000 images processed

# Check if CUDA is available and load the large model (YOLOv8x) to CUDA if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"Using device: {device}")

# Load the YOLO model from the downloaded location
large_model = YOLO("/app/yolov8n_fish_trained.pt")
large_model = large_model.to(device)

# Define GCS bucket
bucket = client.bucket(bucket_name)

# Initialize SQLite database connection
def initialize_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Update the table schema to store additional details
    cursor.execute('''CREATE TABLE IF NOT EXISTS images (
                        image_name TEXT PRIMARY KEY,
                        processed BOOLEAN,
                        detections BOOLEAN,
                        confidence REAL,
                        processed_timestamp TEXT,
                        job_id TEXT,
                        batch_id INTEGER)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        total_images INTEGER,
                        completed_batches INTEGER)''')
    conn.commit()
    return conn

# Load processed images from SQLite
def load_processed_images_db():
    conn = initialize_db()
    cursor = conn.cursor()
    cursor.execute("SELECT image_name, detections FROM images WHERE processed = 1")
    processed_images = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return processed_images

# Restore cumulative counters from the database
def restore_cumulative_counters():
    conn = initialize_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), SUM(detections) FROM images WHERE processed = 1")
    processed_count, detections_count = cursor.fetchone()
    conn.close()
    return processed_count or 0, detections_count or 0

# Scan and create 10 dynamic batches
def scan_and_create_batches(input_folder_gcs):
    blobs = client.list_blobs(bucket_name, prefix=input_folder_gcs)
    image_list = [blob.name for blob in blobs if blob.name.endswith(('.jpg', '.png'))]
    
    conn = initialize_db()
    cursor = conn.cursor()

    # Retrieve unprocessed images
    cursor.execute("SELECT image_name FROM images WHERE processed = 0")
    existing_unprocessed = set(row[0] for row in cursor.fetchall())
    unprocessed_images = [img for img in image_list if img not in existing_unprocessed]

    total_unprocessed = len(unprocessed_images)
    if total_unprocessed == 0:
        st.warning("No unprocessed images found.")
        return

    # Split into 10 even batches
    batch_size = max(1, total_unprocessed // 10)
    batches = [unprocessed_images[i:i + batch_size] for i in range(0, total_unprocessed, batch_size)]
    
    # Create a new job ID
    job_id = str(uuid.uuid4())
    
    # Save each image to the database with its batch_id
    for batch_id, batch in enumerate(batches):
        for image_name in batch:
            cursor.execute("INSERT OR IGNORE INTO images (image_name, processed, job_id, batch_id) VALUES (?, 0, ?, ?)",
                           (image_name, job_id, batch_id))
    
    conn.commit()
    conn.close()

    st.success(f"Job created with ID {job_id}, Total Unprocessed Images: {total_unprocessed}, Batches: {len(batches)}")

# Update SQLite with processed image and details
def update_processed_images_db(image_name, detections, confidence):
    conn = initialize_db()
    cursor = conn.cursor()
    processed_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT OR REPLACE INTO images (image_name, processed, detections, confidence, processed_timestamp) VALUES (?, 1, ?, ?, ?)",
                   (image_name, detections, confidence, processed_timestamp))
    conn.commit()
    conn.close()

# Backup SQLite database to GCS in multiple locations
def backup_db_to_gcs():
    try:
        db_blob1 = bucket.blob("PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/large_2016_dataset/datasetv2/logs/processed_images.db")
        db_blob1.upload_from_filename(DB_FILE)
        logging.info("SQLite database backed up to GCS (logs folder).")
        
        db_blob2 = bucket.blob("PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/large_2016_dataset/datasetv2/database/processed_images.db")
        db_blob2.upload_from_filename(DB_FILE)
        logging.info("SQLite database backed up to GCS (database folder).")
        
    except Exception as e:
        logging.error(f"Failed to backup SQLite database to GCS: {e}")

# Function to read images directly from GCS and save to a temporary file
def read_image_from_gcs_and_save(image_blob):
    try:
        img_bytes = image_blob.download_as_bytes()
        img = Image.open(io.BytesIO(img_bytes))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            img.save(temp_file.name)
            temp_file_path = temp_file.name
        return temp_file_path
    except Exception as e:
        logging.error(f"Failed to read image {image_blob.name}: {e}")
        return None

# Function to process images, run inference, and save results
def process_batch(job_id, batch_id, output_images_gcs, output_labels_gcs, confidence):
    conn = initialize_db()
    cursor = conn.cursor()

    # Get the list of images for the batch
    cursor.execute("SELECT image_name FROM images WHERE job_id = ? AND batch_id = ? AND processed = 0", (job_id, batch_id))
    image_names = [row[0] for row in cursor.fetchall()]
    
    if not image_names:
        st.warning("No images to process for this batch.")
        return

    # Load cumulative counters
    cumulative_processed, cumulative_detections = restore_cumulative_counters()
    session_processed = 0
    session_detections = 0
    sync_count = 0

    # Streamlit UI to display counters
    processed_images_placeholder = st.empty()
    images_with_detections_placeholder = st.empty()

    for image_name in image_names:
        blob = bucket.blob(image_name)
        temp_image_path = read_image_from_gcs_and_save(blob)
        if temp_image_path is None:
            continue
        
        try:
            image = cv2.imread(temp_image_path)
            if image is None or image.shape[0] == 0 or image.shape[1] == 0:
                continue
            
            results = large_model.predict(temp_image_path, conf=confidence)
            has_detections = results[0].boxes is not None and len(results[0].boxes) > 0
            detected_confidence = max([box.conf[0] for box in results[0].boxes]) if has_detections else 0.0
            
            # Track session stats
            session_processed += 1
            sync_count += 1
            if has_detections:
                session_detections += 1

            # Update the database for each image processed
            update_processed_images_db(image_name, has_detections, detected_confidence)

            # Save results to GCS if there are detections
            if has_detections:
                output_image_gcs_path = f"{output_images_gcs}{os.path.basename(image_name)}"
                upload_to_gcs(temp_image_path, output_image_gcs_path)

                label_path = temp_image_path.replace(".jpg", ".txt")
                image_height, image_width, _ = image.shape
                save_yolo_format_labels(results, label_path, image_width, image_height)
                output_label_gcs_path = f"{output_labels_gcs}{os.path.basename(image_name).replace('.jpg', '.txt')}"
                upload_to_gcs(label_path, output_label_gcs_path)

            # Sync database every 1,000 images
            if sync_count >= BACKUP_INTERVAL:
                conn.commit()
                backup_db_to_gcs()
                sync_count = 0

            # Update Streamlit UI counters
            processed_images_placeholder.metric("Processed Images (Total / Session)", f"{cumulative_processed + session_processed} / {session_processed}")
            images_with_detections_placeholder.metric("Images with Detections (Total / Session)", f"{cumulative_detections + session_detections} / {session_detections}")

        except Exception as e:
            logging.error(f"Failed to process {image_name}: {e}")

        finally:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
    
    # Final sync and backup
    conn.commit()
    backup_db_to_gcs()
    conn.close()
    st.success("Batch processing complete.")
    
# Streamlit UI Elements
st.title("🐟 Google Cloud Fish Detector - NODD App 3.0")

# Add description with links to the repository and model
st.markdown("""
**Welcome to the Google Cloud Fish Detector - NODD App!**
This application leverages advanced object detection models to identify fish in images stored on Google Cloud. 

🔗 **[GitHub Repository](https://github.com/MichaelAkridge-NOAA/Fish-or-No-Fish-Detector/tree/MOUSS_2016/google-cloud-shell)**  
🧠 **[YOLOv11 Fish Detector Model on Hugging Face](https://huggingface.co/akridge/yolo11-fish-detector-grayscale)**
""")

# Sidebar configuration
st.sidebar.title("🐟 Fish Detection Settings")
st.sidebar.markdown("""
For more information:
- Contact: Michael.Akridge@NOAA.gov
- Visit the [GitHub repository](https://github.com/MichaelAkridge-NOAA/Fish-or-No-Fish-Detector/)
""")
confidence = st.sidebar.slider("Detection Confidence Threshold", 0.0, 1.0, 0.65)

# Job Management UI
if st.sidebar.button("Scan & Divide Unprocessed Into 10 Batches"):
    scan_and_create_batches(DEFAULT_INPUT_FOLDER_GCS)

# Load existing job IDs from the database
conn = initialize_db()
cursor = conn.cursor()
cursor.execute("SELECT DISTINCT job_id FROM images")
job_ids = [row[0] for row in cursor.fetchall()]
conn.close()

if job_ids:
    job_id = st.sidebar.selectbox("Select Job ID", job_ids)
    batch_id = st.sidebar.number_input("Batch ID (0-9)", min_value=0, max_value=9)

    if st.sidebar.button("Start Processing Selected Batch"):
        process_batch(job_id, batch_id, DEFAULT_OUTPUT_IMAGES_GCS, DEFAULT_OUTPUT_LABELS_GCS, confidence)
else:
    st.sidebar.warning("No jobs available. Scan for unprocessed images to create a new job.")
