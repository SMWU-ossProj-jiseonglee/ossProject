# Barrier-Free Map Project

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Data Preparation](#Data-Preparation)
5. [Model Details](#model-details)
    - [EfficientNet](#efficientnet)
    - [Training Process](#training-process)
    - [Yolov5](#Yolov5)
    - [Web Service](#web-service)
6. [Contributing](#contributing)
7. [License](#license)

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
<br>**2. Install Required Libraries**
<br>!pip install torchinfo
<br>!pip install torchvision
<br>!pip install tqdm
<br>**3. Import Libraries and Set Device**

**Dataset Preparation:**
<br>The data collected through data crawling and direct photos were processed in Roboflow and labeled one by one directly to create a dataset of the desired shape.
## Data Preparation

###  Steps for Image Crawling

1. **Download AutoCrawler**
   - You can download AutoCrawler [here](https://github.com/YoongiKim/AutoCrawler).

2. **Install Dependencies**
   - Ensure you have Python installed, then install the required dependencies:
     ```sh
     pip install -r requirements.txt
     ```

3. **Run AutoCrawler**
   - Configure `keywords.txt` with the keywords for images you want to crawl.
   - Run the crawler:
     ```sh
     python main.py
     ```

4. **Collect Images**
   - The images will be saved in the `download` directory, organized by keyword.
     
### Steps for roboflow Labeling

1. **Upload Images**
   - Go to your RoboFlow account.[RoboFlow](https://roboflow.com) account
   - Create a new project and select the object detection task.
   - Upload your images to the project.

2. **Label Images**
   - Use the RoboFlow annotation tool to draw bounding boxes around the objects in each image.
   - Assign the appropriate labels to each object.

3. **Export Labeled Data**
   - Once labeling is complete, export the dataset in your preferred format (e.g., COCO, YOLO, Pascal VOC).
   - Download the labeled dataset to use for training your object detection model.


## Model Details
### - EfficientNet
The project utilizes EfficientNet, specifically EfficientNet-B0, as the backbone model for image classification tasks. EfficientNet is a family of convolutional neural networks (CNNs) that have been designed to achieve state-of-the-art performance with significantly fewer parameters and FLOPs (Floating Point Operations Per Second) compared to other popular CNN architectures like ResNet and Inception.

**Key Features of EfficientNet**
- **Compound Scaling:** <br>
EfficientNet scales uniformly in depth, width, and resolution dimensions with a compound coefficient φ to balance network depth, width, and resolution.

- **Efficient Blocks:** <br>
EfficientNet uses a combination of mobile inverted bottleneck (MBConv) blocks and squeeze-and-excitation (SE) blocks to optimize model efficiency and accuracy.

- **Pretrained on ImageNet:** <br>
The pretrained EfficientNet-B0 model used in this project has been pretrained on the ImageNet dataset, providing a strong initial feature extractor for downstream tasks.

<img width="400" alt="결과2" src="https://github.com/SMWU-ossProj-jiseonglee/ossProject/assets/162777421/c201dddb-4436-4ec1-8d33-72a01c79bffd">
<br>95.8128078817734 의 정확도 생성

### - Training Process
The training process involves fine-tuning the pretrained EfficientNet-B0 model on a custom dataset for image classification.

Steps Involved in Training:
- **Dataset Preparation:**
The dataset is organized and prepared in the ./data directory.
Images are structured into classes, and transformations (e.g., resizing, padding, and normalization) are applied using PyTorch's transforms.Compose.

- **Model Definition:**
A custom PretrainModel class is defined to load the pretrained EfficientNet-B0 model and replace the classifier head with a new fully connected layer for the specific number of classes in the dataset.

- **Training Configuration:**
Hyperparameters such as batch size, learning rate, number of epochs, and optimizer settings (Adam optimizer with learning rate scheduling) are configured.

- **Loss Function and Metrics:**
Cross-entropy loss is used as the loss function for training, suitable for multi-class classification tasks.
Accuracy is monitored during training to evaluate model performance.

- **Training Loop:**
The model is trained using a combination of training and validation datasets.
During each epoch, the model's performance is evaluated on both datasets to monitor loss and accuracy.
Early stopping criteria are implemented to prevent overfitting and improve efficiency.

- **Model Evaluation:**
After training, the model's performance is evaluated on a separate test dataset to assess its generalization ability.
Test accuracy and other metrics are calculated to measure the model's effectiveness.

- **Visualization and Reporting:**
Training history (loss and accuracy curves) is plotted using matplotlib to visualize model performance throughout the training process.

### - Yolov5


To train a custom object detection model using YOLOv8, follow these steps:

1. **Prepare Custom Dataset**
   - Organize your custom dataset in YOLOv5 format (images and labels).
     
2. **Install module**
     ```sh
    !pip install --upgrade urllib3
    !pip install ultralytics
     ```

3. **Train the Model**
   - Use the following command to train your custom YOLOv8 model:
     ```sh
    !yolo task=detect mode=train model=yolov8n.pt data={dataset.location}/data.yaml  epochs= 100  plots=True
     ```

4. **Evaluate and Use**
   - After training, evaluate the model and use it for inference in your applications.

### Example

```python
import roboflow

# Load the project
rf = roboflow.Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("PROJECT_NAME")

# Download the dataset
dataset = project.version("VERSION_NUMBER").download("yolov8")

### - Web Service
This project generates a web page displaying a map with markers for barrier-free restaurants using data from a text file. The map is created using the Naver Maps API, and the data is dynamically injected into the HTML template through a Python script.

- **Python Script (generate_map.py):**
<br>Reads the restaurant data from locations.txt.<br>
Parses each line to extract latitude, longitude, name, and link.<br>
Inserts this data into map_template.html by replacing a placeholder.<br>
Writes the modified HTML content to map.html.<br>

- **HTML Template (map_template.html):**
<br>Sets up the basic HTML structure and styles.<br>
Loads the Naver Maps API.<br>
Displays a map centered on a predefined location.<br>
Adds markers and info windows for each restaurant based on the injected data.<br>

## Contributing
### How to Contribute
1. Fork the Repository<br>
2. Clone Your Fork<br>
3. Create a New Branch<br>
4. Make Your Changes<br>
5. Test Your Changes<br>
6. Commit Your Changes<br>
7. Push to Your Fork<br>
8. Create a Pull Request<br>

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
