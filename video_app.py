# Import required modules
import streamlit as st
import torch
from torchvision import transforms
import os
import json
import logging
import tempfile
import cv2
import numpy as np
from PIL import Image

# Enable wide layout
st.set_page_config(layout="wide")

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load the custom Faster R-CNN model from the specified path
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

# Title and description for Streamlit app
st.title("🐟 Fish or No Fish Detector - Video (Object Detection)")
st.write("""
Is there a fish 🐟 or not? Upload videos to detect fish in each frame.

This application uses your **custom Faster R-CNN model** for video object detection.
""")

# Sidebar for frame skip control
frame_skip = st.sidebar.slider("Frame Skip (process every nth frame)", 1, 30, 1)

# Initialize session state for detection_completed if it doesn't exist
if "detection_completed" not in st.session_state:
    st.session_state.detection_completed = False

# Initialize session state for all_detections
if "all_detections" not in st.session_state:
    st.session_state.all_detections = []

# Function to process video with Faster R-CNN object detection and frame skip
def process_video_with_detection(uploaded_video, frame_skip):
    # Save uploaded video to a temp file
    temp_video_path = os.path.join(tempfile.gettempdir(), uploaded_video.name)
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    # Open video using OpenCV
    cap = cv2.VideoCapture(temp_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.sidebar.write(f"Video Frame Rate: {frame_rate} FPS")
    st.sidebar.write(f"Total Frames: {total_frames}")

    # Set up the video writer to save the processed video
    output_video_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    # Progress bar and frame counter above the video frame
    progress_bar = st.progress(0)
    frame_counter = st.empty()  # Placeholder to display frame count

    # Prepare to display video frames in real-time
    frame_display = st.empty()  # Placeholder for real-time frame display
    frame_count = 0

    # List to store detection results for each frame
    all_detections = []

    # Loop through video frames
    while cap.isOpened():
        ret, frame = cap.read()

        # If we can't grab a frame, we're at the end of the video
        if not ret:
            break

        # Process every nth frame according to frame_skip
        if frame_count % frame_skip == 0 or frame_count == total_frames - 1:
            # Convert the frame to RGB using OpenCV, then to tensor for Faster R-CNN
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img_tensor = preprocess(Image.fromarray(frame_rgb)).unsqueeze(0)  # Convert to tensor and add batch dim

            # Perform inference using the Faster R-CNN model
            with torch.no_grad():
                outputs = model(img_tensor)
            
            # Extract the bounding boxes, labels, and scores from the output
            boxes = outputs[0]['boxes'].cpu().numpy()
            labels = outputs[0]['labels'].cpu().numpy()
            scores = outputs[0]['scores'].cpu().numpy()

            # Filter out detections with low confidence (adjust threshold as needed)
            threshold = 0.5
            valid_boxes = boxes[scores >= threshold]
            valid_labels = labels[scores >= threshold]
            valid_scores = scores[scores >= threshold]

            # Draw bounding boxes on the frame
            for box, label, score in zip(valid_boxes, valid_labels, valid_scores):
                x1, y1, x2, y2 = box
                label_name = "Fish" if label == 1 else "Other"  # Adjust label based on your classes
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label_name} ({score:.2f})", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Write the annotated frame to the output video
            out.write(frame)

            # Display the processed frame in real-time, resized to a smaller width
            frame_display.image(frame, channels="BGR", width=400)

            # Add detection metadata for the current frame
            all_detections.append({
                "frame": frame_count + 1,
                "detections": [
                    {"box": box.tolist(), "label": int(label), "score": float(score)}
                    for box, label, score in zip(valid_boxes, valid_labels, valid_scores)
                ]
            })

        # Update frame count and progress bar
        progress_percentage = int((frame_count / total_frames) * 100)
        progress_bar.progress(progress_percentage)

        # Display frame counter (e.g., "Processed 150/300 frames")
        frame_counter.text(f"Processed {frame_count + 1} / {total_frames} frames")

        frame_count += 1

    cap.release()
    out.release()  # Save the video file

    # Store the JSON results in the session state for download
    st.session_state["all_detections"] = all_detections
    st.session_state["output_video_path"] = output_video_path  # Store the video path for download

# Function to provide a download link for the processed video
def download_video_button():
    if "output_video_path" in st.session_state:
        video_file = st.session_state["output_video_path"]
        with open(video_file, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

# JSON download button for video results
def download_video_json_button():
    if "all_detections" in st.session_state and st.session_state["all_detections"]:
        # Convert the detections to JSON format
        json_data = json.dumps(st.session_state["all_detections"], indent=4)

        # Display the download button
        st.download_button(
            label="Download Video Results as JSON",
            data=json_data,
            file_name="video_detections.json",
            mime="application/json"
        )

# Main logic for handling video upload and running object detection
uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_video:
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        run_button = st.button("Run", key="run_button")
    with col2:
        clear_button = st.button("Clear Results", key="clear_button")

    # Run the detection only when the "Run" button is clicked
    if run_button and uploaded_video:
        process_video_with_detection(uploaded_video, frame_skip)
        st.session_state.detection_completed = True

    # Show the download button after processing
    if st.session_state.detection_completed:
        with col3:
            download_video_button()

    # Clear the results
    if clear_button:
        st.session_state.clear()  # Clear all session state
        st.session_state.detection_completed = False
