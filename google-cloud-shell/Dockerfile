# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install streamlit ultralytics google-cloud-storage pillow opencv-python-headless scikit-image

# Set the working directory
WORKDIR /app

# Download the YOLO model
RUN wget -O /app/yolov8n_fish_trained.pt https://storage.googleapis.com/nmfs_odp_pifsc/PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/models/yolov8n_fish_trained_small_dataset/yolov8n_fish_trained.pt

# Copy the app code and startup script
COPY app.py /app/app.py
COPY start.sh /app/start.sh

# Make sure the startup script is executable
RUN chmod +x /app/start.sh

# Expose Streamlit default port
EXPOSE 8080

# Use the startup script as the entrypoint
ENTRYPOINT ["/app/start.sh"]
