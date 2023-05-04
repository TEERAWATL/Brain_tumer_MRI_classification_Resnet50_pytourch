#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import io
import torch
import torch.nn as nn
import torchvision  # Add this import
from torchvision import transforms
from PIL import Image
import imghdr
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the saved model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
state_dict = torch.load('resnet50_brain_tumor.pth', map_location=torch.device('cpu')) #for cpu
loaded_resnet50 = torchvision.models.resnet50(weights=None)
num_ftrs = loaded_resnet50.fc.in_features
num_classes = 17
loaded_resnet50.fc = nn.Linear(num_ftrs, num_classes)
loaded_resnet50.load_state_dict(state_dict)
loaded_resnet50.to(device)
loaded_resnet50.eval()

# Preprocess function
def preprocess_image(image: Image.Image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

@app.route('/')
def home():
    return 'Welcome to the Brain Tumor Classification API!'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Load image from request
        file = request.files['file']
        
        # Check if the file is an allowed image format
        file_type = imghdr.what(file)
        allowed_file_types = {'jpeg', 'png', 'tiff'}
        if file_type not in allowed_file_types:
            return jsonify({'error': 'Invalid image format. Please upload a JPEG, PNG, or TIFF file.'}), 400

        img = Image.open(file.stream)

        # Preprocess image and predict
        preprocessed_img = preprocess_image(img).to(device)
        with torch.no_grad():
            output = loaded_resnet50(preprocessed_img)
            prediction = torch.argmax(output, dim=1).item()

        # Get the class name
        # Get the class name
        index_to_label = {
            0: 'Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1',
            1: 'Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1C+',
            2: 'Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T2',
            3: 'Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1',
            4: 'Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1C+',
            5: 'Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T2',
            6: 'NORMAL T1',
            7: 'NORMAL T2',
            8: 'Neurocitoma (Central - Intraventricular, Extraventricular) T1',
            9: 'Neurocitoma (Central - Intraventricular, Extraventricular) T1C+',
            10: 'Neurocitoma (Central - Intraventricular, Extraventricular) T2',
            11: 'Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1',
            12: 'Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1C+',
            13: 'Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T2',
            14: 'Schwannoma (Acustico, Vestibular - Trigeminal) T1',
            15: 'Schwannoma (Acustico, Vestibular - Trigeminal) T1C+',
            16: 'Schwannoma (Acustico, Vestibular - Trigeminal) T2'
        }

        class_name = index_to_label[prediction]

        # Return result as JSON
        return jsonify({'class_name': class_name})

if __name__ == "__main__":
    app.run()
