# Brain_tumer_MRI_classification_Resnet50_pytourch #

This project aims to classify brain tumors using a deep learning model, ResNet50, based on MRI images. The model is trained on a dataset containing different types of brain tumors, such as Glioma, Meningioma, Neurocitoma, and Schwannoma, and other types of lesions, including Abscesses, Cysts, and Encephalopathies.

# Dataset #
The dataset contains MRI images of patients with various types of brain tumors and other lesions. The dataset is organized into subfolders, each named after the corresponding class label. Supported image formats include JPEG, PNG, and TIFF.

# Setup #


1. Clone the repository: git clone https://github.com/TEERAWAT/Brain_tumer_MRI_classification_Resnet50_pytourch
.git

2. Install the required dependencies: !pip install flask torch torchvision pillow

# Usage #
To start the Flask API, run the following command: python app.py

The API will be accessible at http://127.0.0.1:5000/.

To make a prediction, send a POST request with an image file to the /predict endpoint

## Have a good day##

