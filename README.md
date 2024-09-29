# Lip Reading with Deep Neural Networks

This repository contains the code for a real-time lip reading system using a deep neural network. The system processes video frames of a speaker and generates text based on the lip movements detected.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Model Architecture](#model-architecture)
6. [Training](#training)
7. [Testing and Evaluation](#testing-and-evaluation)
8. [Pre-trained Weights](#pre-trained-weights)
9. [References](#references)

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
