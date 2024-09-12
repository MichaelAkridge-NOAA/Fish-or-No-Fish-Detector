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

st.set_page_config(
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

# Title and description for Streamlit app
st.title("🐟 Fish or No Fish Detector")
st.write("""
Is there a fish 🐟 or not? Upload one or more images to detect vulnerable marine ecosystems (corals, crinoids, sponges, and fish). 

Uses the [**FathomNet VME Model**](https://huggingface.co/FathomNet/vulnerable-marine-ecosystems). 
Based on the [Ultralytics YOLOv8x Model](https://github.com/ultralytics/ultralytics) for its object detection capabilities and trained by [FathomNet](https://fathomnet.org) on vulnerable marine ecosystems.  
""")

# Custom CSS to style buttons
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

st.sidebar.markdown("""
---
""")

# Sidebar logos arranged in columns
col1, col2 = st.sidebar.columns(2)
with col1:
    st.image("./logos/nmfs-opensci-logo3.png", width=100, caption="NOAA Open Science")
with col2:
    st.image("./logos/FathomNet_black_bottomText_400px.png", width=100, caption="FathomNet")

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
        fish_count = 0  # To count the number of fish detected in this image
        confidences = []  # Store confidences for plotting later
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # xyxy format
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_label = model.names[class_id]  # Get class label from model names

            # Store confidence for plotting
            confidences.append(conf)

            # Check if the detection is for a fish and above the confidence threshold
            if class_label == "Fish" and conf > confidence:
                fish_count += 1

            boxes.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": conf,
                "class_id": class_id,
                "class_label": class_label  # Add class label to metadata
            })

        # Prepare metadata and bounding box information for this image
        metadata = {
            "image_name": os.path.basename(image_path),
            "bounding_boxes": boxes,
            "fish_count": fish_count,  # Add fish count to metadata
            "confidences": confidences  # Include confidences for plotting
        }

        # Use YOLO's built-in plot method to draw bounding boxes on the image
        result_image = results[0].plot()[:, :, ::-1]  # Convert to RGB for display in Streamlit

        return result_image, metadata  # Return the processed image and metadata
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        st.error("An error occurred during prediction.")
        return None, None

# Reusable function to handle multiple image uploads and display results
def process_images(uploaded_files):
    all_detections = []  # To store all detections for JSON export
    result_images = []  # To store images with bounding boxes
    summary_data = []   # To store summary info for each image
    confidences = []  # For storing all confidences for plotting

    # Get the system's temporary directory
    temp_dir = tempfile.gettempdir()

    # Loop over each uploaded file
    for uploaded_file in uploaded_files:
        if uploaded_file.type.startswith('image/'):
            image = Image.open(uploaded_file)

            # Save the image to a temporary file in the system's temp directory
            temp_file_path = os.path.join(temp_dir, f"{uploaded_file.name}")
            with open(temp_file_path, "wb") as temp_image:
                temp_image.write(uploaded_file.getbuffer())

            # Run YOLO detection
            st.write(f"Detecting in {uploaded_file.name}...")
            with st.spinner('Running detection...'):
                result_image, detection_metadata = run(temp_file_path)
            
            if result_image is not None:
                # Add this image's detection data and image to the lists
                result_images.append((result_image, uploaded_file.name))
                all_detections.append(detection_metadata)

                # Add summary data for charting and table
                summary_data.append({
                    "image_name": uploaded_file.name,
                    "fish_detected": detection_metadata["fish_count"] > 0,
                    "fish_count": detection_metadata["fish_count"]
                })

                # Collect all confidences for scatter plot
                confidences.extend(detection_metadata["confidences"])

                # Display summary on the same line with emoji, bold, and larger colored YES/NO
                fish_detected = detection_metadata['fish_count'] > 0
                fish_status = f"<b><span style='color: green; font-size: 24px;'>YES</span></b> 🐟" if fish_detected else f"<b><span style='color: red; font-size: 24px;'>NO</span></b>"

                # Display formatted summary
                st.markdown(f"**Summary for {uploaded_file.name}:** Fish detected: {fish_status}", unsafe_allow_html=True)

                # Create two columns to show images side by side
                col1, col2 = st.columns(2)

                # Display the uploaded image in the first column
                with col1:
                    st.image(image, caption=f"Uploaded Image - {uploaded_file.name}", use_column_width=True)

                # Display the detection result in the second column
                with col2:
                    st.image(result_image, caption=f"Detection Results - {uploaded_file.name}", use_column_width=True)
                
                st.success(f"Detection completed for {uploaded_file.name} successfully! 🐟")
            else:
                st.warning(f"No marine ecosystems detected in {uploaded_file.name}.")
        else:
            st.error(f"Please upload a valid image file (PNG, JPG, JPEG) for {uploaded_file.name}.")
    
    # Store the detection data in session state to avoid re-running
    st.session_state["all_detections"] = all_detections

    # Return the summary data for charting
    return summary_data, confidences

# Function to display a summary table and scatter plot side by side with image labels
# Function to display a summary table and scatter plot side by side with image labels
def display_summary(summary_data, confidences):
    if summary_data:
        # Convert summary data into a DataFrame
        df = pd.DataFrame(summary_data)

        # Strip file extensions from image names
        df['image_name'] = df['image_name'].apply(lambda x: os.path.splitext(x)[0])

        # Create two columns: one for the table and one for the scatter plot
        col1, col2 = st.columns(2)

        # Display the table with fish counts in the first column
        with col1:
            st.subheader("Summary of Detections")
            st.table(df[["image_name", "fish_count"]])

        # Create a scatter plot of confidence levels in the second column
        with col2:
            st.subheader("Fish Detection Confidence Levels")
            fig, ax = plt.subplots()

            # Loop through confidences and plot each with an image label
            for i, conf in enumerate(confidences):
                image_name = df.iloc[i // len(df), 0]  # Associate image name with each confidence
                ax.scatter(i, conf, c='blue')
                ax.text(i, conf, image_name, fontsize=8, ha='right', rotation=45)  # Rotate labels for better readability

            # Add threshold line and labels
            ax.axhline(final_confidence, color='red', linestyle='--', label=f'Final Threshold ({final_confidence})')
            ax.set_xlabel('Detections')
            ax.set_ylabel('Confidence Level')
            ax.legend(loc='upper right')
            st.pyplot(fig)

        # Add download button next to summary at the bottom
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
uploaded_files = st.file_uploader("Choose image(s)...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Display the Run, Clear, and Download buttons with enhanced styling
if uploaded_files:
    # Create three columns with equal width
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")
    
    # Align the buttons next to each other
    with col1:
        run_button = st.button("Run", key="run_button")
    with col2:
        clear_button = st.button("Clear Results", key="clear_button")

    # Run the detection only when the "Run" button is clicked
    if run_button and uploaded_files:
        summary_data, confidences = process_images(uploaded_files)
        
        # Display the summary table and scatter plot side by side
        display_summary(summary_data, confidences)

    # Clear the results and session state when the "Clear" button is clicked
    if clear_button:
        st.session_state.clear()  # Clear all session state
        st.session_state.detection_completed = False  # Reset the flag

    # JSON download button
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