# Barrier-Free Map Project

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Model Details](#model-details)
    - [EfficientNet](#efficientnet)
    - [Training Process](#training-process)
5. [Contributing](#contributing)
6. [License](#license)
7. [Acknowledgements](#acknowledgements)

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

## Model Details
### - EfficientNet
### - Training Process

## Contributing
We welcome contributions to improve this project. Whether it's fixing bugs, adding new features, improving documentation, or optimizing performance, your help is appreciated!

### How to Contribute
**1. Fork the Repository**
<br>**2. Clone Your Fork**`
<br>**3. Create a New Branch**
<br>**4. Make Your Changes**
<br>**5. Test Your Changes**
<br>**6. Commit Your Changes**
<br>**7. Push to Your Fork** 
<br>**8. Create a Pull Request**

### Contributing Guidelines
To ensure a smooth collaboration process, please follow these guidelines:
- **Code Style:**
    - Follow the existing code style and structure.
    - Write clean and readable code with comments where necessary.
- **Commits:**
    - Make atomic commits that clearly state the purpose of the change.
    - Avoid large, monolithic commits.
- **Pull Requests:**
    - Ensure your pull request is based on the latest code from the `main` branch.
    - Provide a detailed description of what your pull request does.
- **Issues:**
    - Before starting work on a new feature or bug fix, please check if there's already an existing issue or discussion.
    - If you find an issue, feel free to comment on it or open a new issue to discuss your idea.

## License
This project is licensed under the MIT License.

## Acknowledgements
