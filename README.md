# Barrier-Free Map Project

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Details](#model-details)
    - [EfficientNet](#efficientnet)
    - [Training Process](#training-process)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgements](#acknowledgements)

## Introduction 
This is **Barrier-Free Map Project** Project conducted as the final project for the Open Source Programming course at the Department of Artificial Intelligence Engineering, Sookmyung Women's University.   

Using an image classification model built with **EfficientNet**, we classify restaurant entrance photos to determine the presence of stairs.

This project is created to facilitate smoother daily life for wheelchair users. Through this map, wheelchair users can easily fine restaurants that they can access.

## Features
The model was trained through image recognition and allowed the model to distinguish between step images.

Through the image recognition model, the model distinguishes the step image to determine whether the wheelchair is an accessible entrance or not.

## Installation
EfficientNet Image Classification
This project aims to perform image classification using the EfficientNet-B0 model, pretrained on ImageNet, and fine-tuned on your custom dataset.

Requirements:
Python 3.8+, PyTorch, torchvision, tqdm, numpy, scikit-learn, matplotlib, PIL

**1. Mount Google Drive (if using Google Colab)**
<br>from google.colab import drive
<br>drive.mount('/content/drive')
<br>cd /content/drive/MyDrive/forimagedetect/

**2. Install Required Libraries**
<br>!pip install torchinfo
<br>!pip install torchvision
<br>!pip install tqdm

**3. Import Libraries and Set Device**
<br>import warnings
<br>warnings.filterwarnings(action = "ignore")
<br>import time
<br>from tqdm import tqdm
<br>import matplotlib.pyplot as plt
<br>from PIL import Image

<br>import numpy as np
<br>import torch
<br>from torch import nn, optim
<br>from torchinfo import summary
<br>from torchvision.models import efficientnet_b0
<br>from torchvision import transforms
<br>from torchvision.datasets import ImageFolder
<br>from torch.utils.data import DataLoader, random_split
<br>import torchvision.transforms.functional as F
<br>from sklearn.metrics import accuracy_score
<br>import sys

## Usage

## Model Details
### - EfficientNet
### - Training Process

## Contributing

## License

## Acknowledgements
