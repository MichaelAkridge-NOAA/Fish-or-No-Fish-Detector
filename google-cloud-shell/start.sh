#!/bin/bash

# Attempt to download the SQLite database if it exists
echo "Checking for existing database..."
wget -O /app/processed_images.db https://storage.googleapis.com/nmfs_odp_pifsc/PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/large_2016_dataset/database/processed_images.db || echo "Database not found, proceeding without it."

# Start the Streamlit app
streamlit run app.py --server.port=8080 --server.enableCORS=false --server.enableXsrfProtection=false
