
# Emotion Detection with CNN

This repository contains a convolutional neural network (CNN) model trained to detect human emotions from facial images using the FER dataset from Kaggle.

## Project Overview

The goal of this project is to develop a model capable of accurately detecting emotions from facial expressions. This model can be used for various applications, such as mental health monitoring, user experience improvement, and real-time emotion recognition in video streams.

## Dataset

The dataset used for training is the FER (Facial Expression Recognition) dataset from Kaggle. It consists of approximately 35,887 grayscale images, each 48x48 pixels, across seven emotion categories: angry, disgust, fear, happy, sad, surprise, and neutral.

The dataset underwent preprocessing steps, including normalization to ensure consistency and optimal training performance. Data augmentation techniques, such as random rotations, flips, and shifts, were also applied to augment the dataset and improve model robustness.

## Model Architecture

The CNN architecture consists of multiple convolutional and pooling layers followed by fully connected layers for classification. The model utilizes rectified linear unit (ReLU) activation functions and softmax activation in the output layer. Dropout regularization is applied to prevent overfitting.

## Training

The model was trained using the Adam optimizer with a categorical cross-entropy loss function. Training was performed for 50 epochs with early stopping based on validation accuracy and validation loss to prevent overfitting. The training process achieved a final training accuracy of approximately 90% and a validation accuracy of approximately 70%.

## Usage

To use the trained model for inference, simply load the model weights and pass input images through the network. An example notebook demonstrating how to load and use the model for emotion detection is provided in the repository.

## Repository Structure

- `data.md`: Contains the link of FER dataset used for training.
- `notebooks/`: Contains notebook for training.
- `models/`: Pre-trained model weights.

## Future Work

- Fine-tune the model architecture and hyperparameters for improved performance.
- Explore additional data augmentation techniques to enhance model robustness.
- Investigate transfer learning approaches using pretrained models for feature extraction.
