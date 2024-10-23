# Dockerfile
FROM python:3.9-slim

# Install required packages
RUN apt-get update && apt-get install -y wget && \
    pip install streamlit ultralytics google-cloud-storage pillow opencv-python-headless

# Set the working directory
WORKDIR /app

# Download the YOLO model
RUN wget -O /app/yolov8n_fish_trained.pt https://storage.googleapis.com/nmfs_odp_pifsc/PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/models/yolov8n_fish_trained_small_dataset/yolov8n_fish_trained.pt

# Copy the app code
COPY app.py /app/app.py

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
