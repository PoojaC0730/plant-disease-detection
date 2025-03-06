# Plant Disease Detection

 An end-to-end deep learning system for detecting plant diseases from leaf images. This repository includes a comprehensive PyTorch-based training pipeline using transfer learning (VGG19, ResNet34, DenseNet121) and a user-friendly Streamlit web application for real-time disease diagnosis.

## Overview

This project consists of:
1. A PyTorch-based deep learning system that compares performance of different models (VGG19, ResNet34, DenseNet121) for plant disease classification
2. A Streamlit web application that allows users to upload leaf images and get instant disease diagnoses

## Dataset

The model is trained on the [PlantVillage dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village), which contains images of plant leaves with various diseases as well as healthy leaves.

The dataset includes the following plants and their associated diseases:
- Pepper (Bacterial Spot, Healthy)
- Potato (Early Blight, Healthy, Late Blight)
- Tomato (Bacterial Spot, Early Blight, Healthy, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Mosaic Virus, Yellow Leaf Curl Virus)

## Model Training

The `train_models.ipynb` script trains and evaluates various CNN architectures:
- Transfer learning with VGG19, ResNet34, and DenseNet121
- Comparison with a simple CNN trained from scratch
- Data augmentation to improve model generalization
- Balanced sampling to handle class imbalance

## Web Application

The Streamlit app (`app.py`) provides:
- User-friendly interface for uploading leaf images
- Selection between different trained models
- Visual presentation of prediction results
- Information about detected diseases and treatment recommendations

## Installation and Usage

### Prerequisites
- Python 3.8 or higher
- PyTorch
- Torchvision
- Streamlit
- Other dependencies listed in requirements.txt

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/plant-disease-detection.git
   cd plant-disease-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
### Training Models
To train the models yourself:
 - Download the dataset and upload on your Google Drive
 - Run the Train_Models.ipynb file in Colab

### Running the Web App
To run the Streamlit application:
 - Download the saved model weights and save it in the same directory as app.py  
```bash
cd app
streamlit run app.py
```

## Results

The DenseNet121 model achieved the highest validation accuracy of 97.5%, followed by ResNet34 (96.8%) and VGG19 (95.3%). The model trained from scratch achieved only 85.7% accuracy.

## Future Work

- Add more plant species and diseases
- Add mobile application support
- Improve treatment recommendations with agricultural expert input

## License

This project is licensed under the MIT License - see the LICENSE file for details.
