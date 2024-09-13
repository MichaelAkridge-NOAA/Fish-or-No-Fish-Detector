# Import required modules
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import json
import logging
import tempfile
import pandas as pd
import matplotlib.pyplot as plt

sample_images_folder = "./images/sample_images"
st.set_page_config(
    page_title="Fish Detector",
    page_icon="🐟",
    layout="wide"
)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load the YOLO model
model_path = os.getenv("MODEL_PATH", "./models/best.pt")
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please check your setup.")
    st.stop()

model = YOLO(model_path)

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
Based on the [Ultralytics YOLOv8x Model](https://github.com/ultralytics/ultralytics) for its object detection capabilities and trained by [FathomNet](https://fathomnet.org) on vulnerable marine ecosystems.  
""")

# Custom CSS for button and uploader alignment
st.markdown("""
    <style>
    .custom-file-uploader {
        display: flex;
        align-items: center;
        margin-top: -10px; /* Adjust to move button closer */
        justify-content: flex-start;
    }
    .css-1cpxqw2 {
        flex-grow: 1;  /* Let file uploader take remaining space */
    }
    .sample-button {
        font-size: 14px;
        padding: 8px;
        background-color: #007BFF;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin-left: 10px;
        height: 38px; /* Ensure button matches uploader height */
    }
    .sample-button:hover {
        background-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)

# Custom CSS for default button styling
st.markdown("""
    <style>
    .stButton>button, .stDownloadButton>button {
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
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with title, credits, and model details
st.sidebar.title("🐟 Fish or No Fish Detector")
st.sidebar.markdown("""
For more information:
- Contact: Michael.Akridge@NOAA.gov
- Visit the [GitHub repository](https://github.com/MichaelAkridge-NOAA/Fish-or-No-Fish-Detector/)
""")

# Sidebar with dynamic confidence slider
st.sidebar.header("Models Parameter Settings")
confidence = st.sidebar.slider("Detection Confidence Threshold", 0.0, 1.0, 0.15)
final_confidence = st.sidebar.slider("Final Yes/No Confidence Threshold", 0.0, 1.0, 0.5)

st.sidebar.markdown("---")

# Sidebar logos arranged in columns
col1, col2 = st.sidebar.columns(2)
with col1:
    st.image("./images/logos/nmfs-opensci-logo3.png", width=100, caption="NOAA Open Science")
with col2:
    st.image("./images/logos/FathomNet_black_bottomText_400px.png", width=100, caption="FathomNet")

# Prediction kwargs
PREDICT_KWARGS = {
    "conf": confidence,
}

# Define the prediction function to return bounding box data and number of fish detections
def run(image_path):
    logging.info(f"Running model prediction on {image_path}")
    try:
        results = model.predict(image_path, **PREDICT_KWARGS)

        # Extract bounding box data using valid attributes
        boxes = []
        fish_count = 0
        confidences = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_label = model.names[class_id]

            confidences.append(conf)

            if class_label == "Fish" and conf > confidence:
                fish_count += 1

            boxes.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": conf,
                "class_id": class_id,
                "class_label": class_label
            })

        metadata = {
            "image_name": os.path.basename(image_path),
            "bounding_boxes": boxes,
            "fish_count": fish_count,
            "confidences": confidences
        }

        result_image = results[0].plot()[:, :, ::-1]

        return result_image, metadata
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        st.error("An error occurred during prediction.")
        return None, None

# Reusable function to handle multiple image uploads and display results
def process_images(uploaded_files):
    all_detections = []
    result_images = []
    summary_data = []
    confidences = []
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

            confidences.extend(detection_metadata["confidences"])

            # Display fish status
            fish_detected = detection_metadata['fish_count'] > 0
            fish_status = f"<b><span style='color: green; font-size: 24px;'>YES</span></b> 🐟" if fish_detected else f"<b><span style='color: red; font-size: 24px;'>NO</span></b>"

            st.markdown(f"**Summary for {os.path.basename(image_path)}:** Fish detected: {fish_status}", unsafe_allow_html=True)

            # Display images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption=f"Uploaded Image - {os.path.basename(image_path)}", use_column_width=True)
            with col2:
                st.image(result_image, caption=f"Detection Results - {os.path.basename(image_path)}", use_column_width=True)

            st.success(f"Detection completed for {os.path.basename(image_path)} successfully! 🐟")

        else:
            st.warning(f"No marine ecosystems detected in {os.path.basename(image_path)}.")

    st.session_state["all_detections"] = all_detections
    return summary_data, confidences

# Function to display a summary table and scatter plot side by side with image labels
def display_summary(summary_data, confidences):
    if summary_data:
        df = pd.DataFrame(summary_data)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Summary of Detections")
            st.table(df[["image_name", "fish_count"]])

        with col2:
            st.subheader("Fish Detection Confidence Levels")
            fig, ax = plt.subplots()
            confidence_index = 0

            for i, row in df.iterrows():
                num_confidences_for_image = len([c for c in confidences[confidence_index:confidence_index + row["fish_count"]]])

                for j in range(num_confidences_for_image):
                    if confidence_index < len(confidences):
                        ax.scatter(confidence_index, confidences[confidence_index], c='blue')
                        ax.text(confidence_index, confidences[confidence_index], row['image_name'], 
                                fontsize=10, ha='center', va='bottom', rotation=0)
                        confidence_index += 1

            ax.axhline(final_confidence, color='red', linestyle='--', label=f'Final Threshold ({final_confidence})')
            ax.set_xlabel('Detections')
            ax.set_ylabel('Confidence Level')
            ax.legend(loc='lower left')
            st.pyplot(fig)

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
# Image uploader with multiple file support
st.markdown('<div class="custom-file-uploader">', unsafe_allow_html=True)
uploaded_files = st.file_uploader("Choose image(s)...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Check if files are uploaded, hide the "Auto Run with Sample Images" button if they are
if not uploaded_files and not st.session_state.get('use_sample_images', False):
    use_sample_images = st.button("Or Auto Run Using Sample Images", key="sample_button")
else:
    use_sample_images = None
st.markdown('</div>', unsafe_allow_html=True)

# Add the functionality for the "Try it with Sample Images" button
if use_sample_images:
    sample_images = load_sample_images()
    st.session_state['use_sample_images'] = True
    for sample_image in sample_images:
        st.session_state.setdefault('uploaded_files', []).append(sample_image)
    st.session_state['run_automatically'] = True

# Display the Run, Clear, and Download buttons with enhanced styling
if uploaded_files or st.session_state.get('uploaded_files'):
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")

    if not st.session_state.get('use_sample_images', False):
        with col1:
            run_button = st.button("Click to Run", key="run_button")
    else:
        run_button = None

    # Initialize clear_button to None to avoid NameError
    clear_button = None

    # Conditionally hide the "Clear Results" button while processing
    with col2:
        if not st.session_state.get('processing', False):
            clear_button = st.button("Clear Results", key="clear_button")

    # Run automatically if triggered by the sample images button or the run button
    if run_button or st.session_state.get('run_automatically'):
        st.session_state['processing'] = True  # Set the processing flag
        summary_data, confidences = process_images(uploaded_files or st.session_state['uploaded_files'])
        display_summary(summary_data, confidences)
        st.session_state['processing'] = False  # Reset the processing flag after processing is done
        st.session_state['run_automatically'] = False
        st.session_state['use_sample_images'] = False

    # Now this check will work, even if clear_button is not defined earlier
    if clear_button:
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
