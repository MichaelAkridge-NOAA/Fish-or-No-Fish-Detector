# Fish Detection Model Card

## Model Overview
This model was trained to detect fish in underwater **Grayscale Imagery** using the YOLOv8n architecture. The model leverages **unsupervised learning** techniques to learn patterns and identify fish without relying on fully labeled datasets.

- **Model Architecture**: YOLOv8n
- **Task**: Object Detection (Fish Detection)
- **Footage Type**: Grayscale(Black-and-White) Underwater Footage
- **Classes**: Fish

## Model Weights
The model's weights can be found [here](./yolov8n_fish_trained.pt)

## Training Data
- **Dataset**: `fish_dataset.zip` consisting of Grayscale(black-and-white) underwater footage.
- **Training/Validation Split**: The dataset was split into 80% training and 20% validation.
- **Classes**: The model was trained on a single class (fish).
- **Learning Approach**: Unsupervised learning, meaning the model identified patterns in the data without needing detailed annotations for all images.

## Training Configuration
- **Model Weights File**: `yolov8n_fish_trained.pt`
- **Number of Epochs**: 50
- **Learning Rate**: 0.001
- **Batch Size**: 16
- **Image Size**: 416x416

## Training Metrics
Below are the key metrics from the model evaluation on the validation set:

- **Precision**: 0.863
- **Recall**: 0.869
- **mAP50**: 0.936
- **mAP50-95**: 0.856

## Validation Results
### Training and Validation Losses
![Training and Validation Losses](./train/results.png)

### Confusion Matrix
![Confusion Matrix](./train/confusion_matrix.png)

### Precision-Recall Curve
![Precision-Recall Curve](./train/PR_curve.png)

### F1 Score Curve
![F1 Score Curve](./train/F1_curve.png)


## How to Use the Model

To use the trained model, follow these steps:

1. **Load the Model**:
   ```python
   from ultralytics import YOLO

   # Load the model
   model = YOLO("yolov8n_fish_trained.pt")

Limitations
The model was trained on black-and-white underwater footage, and may not generalize well to color images or videos with different lighting conditions.
The unsupervised learning nature of this model may lead to some incorrect detections, particularly in noisy environments where it may confuse other underwater objects for fish.
Images with complex backgrounds, occlusions, or poor resolution may affect the model's performance.

Ethical Considerations
The unsupervised learning approach could lead to biases in detections, especially in new environments or types of marine life that were not represented in the training dataset.
This model should not be used in critical applications without thorough validation to ensure it doesn't miss key detections or produce incorrect results in sensitive scenarios.
Consider the potential environmental or societal impact when using the model for marine conservation or research, and ensure that the detections are verified.

### Additional Notes:
- **Grayscale Imagery**: The model may perform better on grayscale images and might not generalize well to color underwater footage or images with different lighting conditions.
- **Unsupervised Learning**: Since using an unsupervised approach, it's worth noting that this can make the model more flexible but also more prone to errors or misclassifications without annotated data.


