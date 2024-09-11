# Yes Fish / No Fish Detector
Is there a fish 🐟 or not?  Detect vulnerable marine ecosystems(corals, crinoids, sponges, and fish.) Uses the [FathomNet VME Model](https://huggingface.co/FathomNet/vulnerable-marine-ecosystems) for object detection in marine ecosystems, specifically focusing on corals, crinoids, sponges, and fish. It is built on top of the **Ultralytics YOLOv8x** model, providing fast and accurate object detection capabilities.

## Features
- Upload one or more images to detect vulnerable marine ecosystems.
- Adjust the confidence threshold for predictions.
- Download all detection and bounding box results in JSON format.

### Installation

1. Clone the repository:
    ```
    git clone https://github.com/MichaelAkridge-NOAA/Yes-Fish-No-Fish-Detector.git
    cd Yes-Fish-No-Fish-Detector
    ```
2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Download the pre-trained model from [FathomNet VME Model](https://huggingface.co/FathomNet/vulnerable-marine-ecosystems/blob/main/best.pt) and save it in the `./models/` directory.

### Running the App

Run the Streamlit app with the following command:
```
streamlit run app.py
```
----------
#### Disclaimer
This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project content is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.

##### License 
See the [LICENSE.md](./LICENSE.md) for details on this code.

##### Credit, Models & Licences
- For [FathomNet VME Model](https://huggingface.co/FathomNet/vulnerable-marine-ecosystems), see their [license](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/cc-by-4.0.md) for more details
    - Based on Ultralytics YOLOv8x Model, see their [license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) for more details.  
