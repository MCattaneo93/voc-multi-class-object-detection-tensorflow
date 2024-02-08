# VOC Multi-Class Object Detection with TensorFlow

This repository contains code for performing multi-class object detection on the VOC (Visual Object Classes) dataset using TensorFlow. The code implements a convolutional neural network (CNN) model for detecting and classifying objects in images from the VOC dataset.

## Dataset

The VOC dataset (Visual Object Classes) is a popular dataset for object detection and classification tasks. It contains images with annotated objects belonging to 20 different classes, including common objects such as cars, dogs, cats, and bicycles.

## Code Overview

The main script in this repository (`FocalLoss_CNN.py`) performs the following tasks:

1. **Data Preprocessing**: The script loads the VOC dataset using TensorFlow Datasets and preprocesses the images for training and testing. Preprocessing steps include resizing, normalization, and Gaussian noise reduction.

2. **Model Definition**: A CNN model is defined using TensorFlow's Keras API. The model consists of convolutional layers, max-pooling layers, and fully connected layers for classification.

3. **Model Training**: The model is compiled using the Sparse Categorical Focal Loss as the loss function and Adam optimization. It is then trained on the preprocessed training dataset.

4. **Model Evaluation**: After training, the model is evaluated on the preprocessed testing dataset to measure its performance in terms of accuracy.

## Acknowledgments
- This project is inspired by the VOC dataset and the TensorFlow framework.
- Special thanks to the authors of TensorFlow, TensorFlow Datasets, and TensorFlow Addons for their contributions to the deep learning community.

