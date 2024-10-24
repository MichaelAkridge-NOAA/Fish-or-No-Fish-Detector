# Google Cloud Fish Detector NODD App
This app runs a Streamlit application that processes images from Google Cloud Storage (GCS) using a pre-trained YOLO11 fish detection model. The application can perform fish detection on images stored in GCS and save the results back to GCS in yolo format.
- Model link: https://huggingface.co/akridge/yolo11-fish-detector-grayscale

## Contact
- michael.akridge@noaa.gov

## Step 1. Start the Google Cloud Shell 
[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FMichaelAkridge-NOAA%2FFish-or-No-Fish-Detector&cloudshell_git_branch=MOUSS_2016&cloudshell_print=cloud-shell-readme.txt&cloudshell_workspace=google-cloud-shell&cloudshell_tutorial=TUTORIAL.md)

## Step 2: This will ask to clone the repo:
- Github Repo: https://github.com/MichaelAkridge-NOAA/Fish-or-No-Fish-Detector/tree/MOUSS_2016/google-cloud-shell

## Step 3: Once cloned, run the docker setup
- Once repo is cloned run:
```
docker compose up
```
## Step 4. Visit app in browser and input location to data

## Step 5. Authorized Access via Shell Window

## Step 6. View Progress & output in the app, your google cloud bucket, and or shell window

## Features
- **Automated Image Processing**: Reads images from a specified GCS bucket, runs YOLO-based object detection, and saves the output.
- **Customizable Paths**: Users can specify input and output directories directly from the Streamlit interface.
- **Verification Images**: The app generates and displays verification images with bounding boxes for detected objects.

## Requirements
- **Google Cloud Shell**: Run this directly in Google Cloud Shell for easy authentication.
- **Google Cloud Storage**: For storage of images and results
## Auto Build Notes
- see github workflow

## Manual Build Notes
```
#docker build -t gcs-fish-detector .
docker build --no-cache -t gcs-fish-detector .
docker tag gcs-fish-detector michaelakridge326/gcs-fish-detector:latest
docker push michaelakridge326/gcs-fish-detector:latest
# other helpful commands
docker rm google-cloud-shell-streamlit_app-1
docker image prune -a
```
### More info on Google Cloud Shell:
- https://cloud.google.com/shell/docs/how-cloud-shell-works
- https://cloud.google.com/shell/docs/open-in-cloud-shell
- https://cloud.google.com/shell/docs/configuring-cloud-shell#environment_customization****
