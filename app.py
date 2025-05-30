# Import required modules
import streamlit as st
import cv2
import tempfile
import os
from collections import defaultdict
import numpy as np
import torch
from ultralytics import YOLO
import json
import logging
import shutil
import plotly.graph_objects as go
import time

# Enable wide layout
st.set_page_config(layout="wide")

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load the YOLO model
model_path = os.getenv("MODEL_PATH", "./model/fish_no_fish_v1.pt")
model = YOLO(model_path)
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please check your setup.")
    st.stop()


# Title and description for Streamlit app
st.title("🐟 Fish or No Fish Detector - Video")
st.write("""
Is there a fish 🐟 or not? Upload videos to detect fish
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

# Sidebar with logos, dynamic confidence slider, and frame skip control
st.sidebar.title("🐟 Fish or No Fish Detector")
st.sidebar.markdown("""
For more information:
- Contact: Michael.Akridge@NOAA.gov
- Visit the [GitHub repository](https://github.com/MichaelAkridge-NOAA/Fish-or-No-Fish-Detector/)
""")

# Confidence slider
st.sidebar.header("Model Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.15)

# Frame skip slider for video processing
frame_skip = st.sidebar.slider("Frame Skip (process every nth frame)", 1, 30, 1)

st.sidebar.markdown("""---""")

# Initialize session state for detection_completed if it doesn't exist
if "detection_completed" not in st.session_state:
    st.session_state.detection_completed = False

# Initialize session state for all_detections
if "all_detections" not in st.session_state:
    st.session_state.all_detections = []

# Initialize track history for visualization
track_history = defaultdict(lambda: [])

# Prediction kwargs with tracking configuration
PREDICT_KWARGS = {
    "conf": confidence,
    "tracker": "tracker/custom_tracker.yaml",  # Use custom tracker configuration
    "persist": True,  # Persist tracking across frames
    "verbose": False,
    "stream": True,  # Enable streaming mode to prevent memory accumulation
    "device": 0 if torch.cuda.is_available() else 'cpu',  # Use GPU if available
    "show": False,  # Don't show the native YOLO window
    "vid_stride": frame_skip  # Use frame skip at YOLO level for better performance
}

# Function to process video with ByteTrack and optimized settings
def process_video_with_tracks(uploaded_video, frame_skip):
    try:
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

            # Resize frame to improve processing speed
            frame = cv2.resize(frame, (1280, int(1280 / frame_width * frame_height)))

            # Process every nth frame according to frame_skip
            if frame_count % frame_skip == 0 or frame_count == total_frames - 1:
                results = model.track(frame, persist=True, tracker="./tracker/custom_tracker.yaml")  # Using ByteTrack for tracking
                annotated_frame = results[0].plot()

                boxes = results[0].boxes.xywh.cpu()  # Get the boxes in xywh format

                # Check if tracking IDs are available
                if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()  # Get the list of track IDs
                else:
                    track_ids = [None] * len(boxes)  # Fallback if no track IDs are available

                # Loop over each detected object and its track ID
                frame_detections = []
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box  # Extract the center x, y, width, and height

                    # Fetch confidence and class ID directly from the results object
                    conf = results[0].boxes.conf[0].cpu().item()  # Get the confidence score
                    class_id = int(results[0].boxes.cls[0].cpu().item())  # Get the class ID

                    # Only process if the class is "fish" (adjust class ID accordingly)
                    if class_id == 0:  # Assuming class 0 is fish, adjust if necessary
                        frame_detections.append({
                            "x": float(x),
                            "y": float(y),
                            "width": float(w),
                            "height": float(h),
                            "confidence": conf,
                            "class_id": class_id,
                            "track_id": track_id
                        })

                # Log the detections for this frame
                all_detections.append({
                    "frame": frame_count + 1,
                    "detections": frame_detections
                })

                # Write the annotated frame to the output video
                out.write(annotated_frame)

                # Display the processed frame in real-time
                frame_display.image(annotated_frame, channels="RGB", width=600)

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

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        raise e

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
            label="Download Fish Detection Results as JSON",
            data=json_data,
            file_name="fish_detections.json",
            mime="application/json"
        )

# Function to provide a zip download link for detected frames
def download_detected_frames_button():
    if "detected_frames_dir" in st.session_state:
        detected_frames_dir = st.session_state["detected_frames_dir"]
        zip_path = shutil.make_archive(detected_frames_dir, 'zip', detected_frames_dir)
        with open(zip_path, "rb") as f:
            st.download_button(
                label="Download Detected Frames as Zip",
                data=f,
                file_name="detected_fish_frames.zip",
                mime="application/zip"
            )

# Function to create the detection timeline plot
def create_detection_timeline(all_detections):
    if not all_detections:
        return None
    
    # Group detections by frame
    detections_by_frame = {}
    unique_tracks = set()
    
    # Process all detections to get frame-by-frame counts
    for detection in all_detections:
        frame = detection["frame"]
        track_id = detection["track_id"]
        
        if frame not in detections_by_frame:
            detections_by_frame[frame] = set()
            
        if track_id != -1:  # Only count tracked detections
            detections_by_frame[frame].add(track_id)
            unique_tracks.add(track_id)
    
    # Create timeline data
    frames = sorted(detections_by_frame.keys())
    fish_counts = [len(detections_by_frame[frame]) for frame in frames]
    
    # Create the line plot
    fig = go.Figure()
    
    # Add the main line
    fig.add_trace(go.Scatter(
        x=frames,
        y=fish_counts,
        mode='lines+markers',
        name='Fish Count',
        line=dict(color='#007BFF', width=2),
        marker=dict(
            size=6,
            color='#007BFF',
            symbol='circle'
        ),
        hovertemplate='Frame: %{x}<br>Fish Count: %{y}<extra></extra>'
    ))
    
    # Update layout with improved styling
    fig.update_layout(
        title=dict(
            text='Fish Detection Timeline',
            x=0.5,
            xanchor='center',
            font=dict(size=24)
        ),
        xaxis_title='Frame Number',
        yaxis_title='Number of Fish',
        hovermode='x unified',
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='#e0e0e0',
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='#e0e0e0',
            rangemode='nonnegative'  # Ensure y-axis starts at 0 or above
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        height=500  # Fixed height for better visibility
    )
    
    return fig, len(unique_tracks)

# Main logic for handling video upload and running detection
uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_video:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Video")
        original_frame_placeholder = st.empty()
    with col2:
        st.subheader("Processed Video")
        processed_frame_placeholder = st.empty()
    
    # Add control buttons in a row
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 1, 1])
    
    with ctrl_col1:
        run_button = st.button("Run", key="run_button")
    with ctrl_col2:
        clear_button = st.button("Clear Results", key="clear_button")
    with ctrl_col3:
        stop_button = st.button("Stop", key="stop_button")

    # Initialize stop flag in session state
    if "stop_processing" not in st.session_state:
        st.session_state.stop_processing = False

    if stop_button:
        st.session_state.stop_processing = True

    # Run the detection only when the "Run" button is clicked
    if run_button and uploaded_video:
        st.session_state.stop_processing = False
        
        # Stats placeholder
        stats_placeholder = st.empty()
        fps_placeholder = st.sidebar.empty()
        progress_bar = st.progress(0)
        
        temp_video_path = os.path.join(tempfile.gettempdir(), uploaded_video.name)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        cap = cv2.VideoCapture(temp_video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        output_video_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

        frame_count = 0
        all_detections = []
        try:
            prev_time = time.time()            # Process video with tracking using streaming mode
            for result in model.track(source=temp_video_path, **PREDICT_KWARGS):
                if st.session_state.stop_processing:
                    st.warning("Processing stopped by user")
                    break
                    
                if frame_count % frame_skip == 0 or frame_count == total_frames - 1:
                    # Get and display original frame
                    frame = result.orig_img
                    original_frame_placeholder.image(frame, channels="BGR")
                    
                    # Get the annotated frame with YOLO's built-in visualization
                    annotated_frame = result.plot()  # This includes boxes, labels, and tracks

                    # Update track history and draw trails
                    if result.boxes is not None and result.boxes.id is not None:
                        boxes = result.boxes.xywh.cpu()
                        track_ids = result.boxes.id.int().cpu().tolist()
                        
                        # Draw existing tracks first
                        for track_id, track in track_history.items():
                            if len(track) > 1:
                                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                                cv2.polylines(annotated_frame, [points], 
                                            isClosed=False, 
                                            color=(0, 255, 0),  # Green color
                                            thickness=2)
                        
                        # Update track history with new positions
                        for box, track_id in zip(boxes, track_ids):
                            x, y = float(box[0]), float(box[1])  # Get center x, y coordinates
                            track = track_history[track_id]
                            track.append((x, y))
                            if len(track) > 30:  # Keep 30 frames of history
                                track.pop(0)

                    # Display processed frame with annotations
                    processed_frame_placeholder.image(annotated_frame, channels="BGR")
                    
                    # Save the annotated frame to video file
                    out.write(annotated_frame)

                    # Process tracking results for our data collection
                    if result.boxes is not None and len(result.boxes):
                        boxes = result.boxes.cpu().numpy()
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            track_id = int(box.id[0]) if box.id is not None else -1
                            conf = float(box.conf[0])
                            
                            if conf >= confidence:
                                # Store detection data
                                detection_data = {
                                    "frame": frame_count,
                                    "track_id": track_id,
                                    "confidence": conf,
                                    "bbox": [x1, y1, x2, y2]
                                }
                                all_detections.append(detection_data)

                    # Calculate and display FPS
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time)
                    prev_time = curr_time
                    fps_placeholder.metric("FPS", f"{fps:.2f}")

                    # Update progress and stats
                    progress_percentage = int((frame_count / total_frames) * 100)
                    progress_bar.progress(progress_percentage)
                    
                    current_tracks = set(d["track_id"] for d in all_detections if d["track_id"] != -1)
                    stats_placeholder.markdown(f"""
                    ### Live Statistics
                    - Current Fish Count: {len(current_tracks)}
                    - Processed Frames: {frame_count}
                    - FPS: {fps:.2f}
                    """)
                    
                frame_count += 1

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            raise e
        finally:
            out.release()
            try:
                os.remove(temp_video_path)
            except:
                pass

        st.session_state["all_detections"] = all_detections
        st.session_state["output_video_path"] = output_video_path
        st.session_state.detection_completed = True

    # Show the download buttons and timeline after processing
    if st.session_state.detection_completed:
        with ctrl_col3:
            download_video_button()
        with ctrl_col4:
            download_video_json_button()
            
        # Add the timeline visualization
        if "all_detections" in st.session_state and st.session_state["all_detections"]:
            st.subheader("Fish Detection Timeline")
            timeline_fig, total_unique_fish = create_detection_timeline(st.session_state["all_detections"])
            if timeline_fig:
                st.plotly_chart(timeline_fig, use_container_width=True)
                
                # Add final statistics with more detail
                max_frame = max(d["frame"] for d in st.session_state["all_detections"])
                total_detections = len(st.session_state["all_detections"])
                
                st.markdown(f"""
                ### Final Detection Statistics
                - Total unique fish tracked: {total_unique_fish}
                - Total frames analyzed: {max_frame}
                - Total number of detections: {total_detections}
                - Average detections per frame: {total_detections / max_frame:.2f}
                """)

    # Clear the results
    if clear_button:
        st.session_state.clear()  # Clear all session state
        st.session_state.detection_completed = False
