# Stage 1: Install dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies for OpenCV and YOLO with no recommended packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy the models directory from your project
COPY models /app/models

# Stage 2: Build final image
FROM python:3.10-slim

WORKDIR /app

# Install necessary system dependencies for OpenCV in final image as well
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the app code and the models folder
COPY . .

# Add a non-root user for security
RUN addgroup --system appgroup && adduser --system appuser && adduser appuser appgroup
USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]

