# Welcome to the Google Cloud Fish Detector App!

For more details, visit the Github here:
- https://github.com/MichaelAkridge-NOAA/Fish-or-No-Fish-Detector

And view the model here: 
- https://huggingface.co/akridge/yolo11-fish-detector-grayscale

## Step 1: Running the Environment
In your terminal <walkthrough-cloud-shell-icon></walkthrough-cloud-shell-icon> below, run the following command:
```
docker compose up
```

## Step 2: Access the App
Once the container is running, you can visit it by::
- Clicking the Web Preview button <walkthrough-web-preview-icon></walkthrough-web-preview-icon>
- and select "Preview on port 8080" To Access your Environment

### FAQ: How to Tell if its running?
You should see a message in the terminal once it's ready, similar to:
"streamlit-app  | [I 2024-06-11 02:01:34.895 ServerApp] Server 2.14.1 is running at:"

## Congratulations
That's it! As you can see it's very easy to get started with Google Cloud Shell and container images.
<walkthrough-conclusion-trophy></walkthrough-conclusion-trophy>

### Next steps:
Explore using your google cloud environment with data on google cloud for faster performance. 
- Our PIFSC Google Cloud Data Archive: https://console.cloud.google.com/storage/browser/nmfs_odp_pifsc

### More info: 
### What is Docker Compose?

Docker Compose is a tool that allows you to define and manage Docker applications. With a simple YAML file, you can configure your application's services, networks, and volumes.

### Why use Docker Compose?

- Simplifies management
- Provides a single command to start all services
- Facilitates reproducible environments

### This command will:
- Pull the 'gcs-fish-detector' image from the Docker Hub: https://hub.docker.com/repository/docker/michaelakridge326/gcs-fish-detector/general
- Creates and starts the app container.
- Maps port 8080 on the shell host to port 8080 in the container.
- And starts up the app
