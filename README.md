# Lip Reading with Deep Neural Networks

This repository contains the code for a real-time lip reading system using a deep neural network. The system processes video frames of a speaker and generates text based on the lip movements detected.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Model Architecture](#model-architecture)
6. [Training and Testing](#training&testing)
7. [References](#references)

## Introduction

This project uses TensorFlow and OpenCV to build a deep neural network capable of reading lips from video sequences. The model predicts the sequence of characters being spoken by analyzing the speaker's lip movements.

## Features
- **Video Preprocessing:** Converts each video to grayscale and crops around the lips for efficient processing.
- **Deep Neural Network:** A 3D Convolutional Neural Network (CNN) followed by a Bidirectional Long Short-Term Memory (LSTM) network to handle temporal dependencies.
- **CTC Loss:** Connectionist Temporal Classification (CTC) loss is used to predict sequences of characters, making it robust to sequence length mismatches.
- **Character-to-Text Mapping:** Converts predicted character sequences into readable text.

## Installation

### 1. Install Dependencies

Make sure you have Python 3.x installed. Install the required libraries by running:

```bash
pip install opencv-python matplotlib imageio gdown tensorflow
```
### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/lip-reading.git
cd lip-reading
```
## Data Preparation

Download the dataset from Google Drive.
Extract the dataset:
```bash
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL', 'data.zip', quiet=False); gdown.extractall('data.zip')"
```

## Model Architecture

The lip-reading model uses a combination of:

- **3D Convolutional Layers**: For spatial and temporal feature extraction from video frames.
- **Bidirectional LSTMs**: To capture temporal dependencies from both past and future frames.
- **TimeDistributed Flattening**: To flatten the output before sending it to the LSTMs.
- **Dense Layers with Softmax**: For predicting characters from a predefined vocabulary.


## Training and Testing

The training script uses Adam optimizer and a Learning Rate Scheduler that adjusts the learning rate over epochs.
CTC Loss is applied to handle the sequence-to-sequence nature of the lip-reading task.
Testing and Evaluation
use checkpoints to train the model:

```bash
model.load_weights('models/checkpoint')
```
Test using the sample video 
The prediction will be displayed after processing, alongside the ground truth.

## References

## References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Documentation](https://opencv.org/)
