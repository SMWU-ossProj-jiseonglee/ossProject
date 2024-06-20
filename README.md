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
from google.colab import drive
drive.mount('/content/drive')
cd /content/drive/MyDrive/forimagedetect/

**2. Install Required Libraries**
!pip install torchinfo
!pip install torchvision
!pip install tqdm

**3. Import Libraries and Set Device**
import warnings
warnings.filterwarnings(action = "ignore")
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import torch
from torch import nn, optim
from torchinfo import summary
from torchvision.models import efficientnet_b0
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.functional as F
from sklearn.metrics import accuracy_score
import sys

## Usage

## Model Details
### - EfficientNet
### - Training Process

## Contributing

## License

## Acknowledgements
