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

# Attempt to download the SQLite database if it exists
RUN wget -O /app/processed_images.db https://storage.googleapis.com/nmfs_odp_pifsc/PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/large_2016_dataset/database/processed_images.db || echo "Database not found, proceeding without it."

# Copy the app code
COPY app.py /app/app.py

# Expose Streamlit default port
EXPOSE 8501

# Define a volume to persist the SQLite database
VOLUME ["/app"]

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
