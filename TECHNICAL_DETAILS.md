# CNN 3D Visualization - Technical Documentation
**Version: v1.0.0**

## Overview
This project features an interactive 3D visualization of a Convolutional Neural Network (CNN) designed for handwritten digit recognition (MNIST style). It allows users to monitor the training process in real-time, inspect internal layer activations, and understand how information flows through the network.

## Core Architecture

### 1. Frontend: React + Three.js
- **React**: Manages the interface state, panels, and application logic.
- **React Three Fiber (@react-three/fiber)**: Bridges React with Three.js for rendering the 3D scene.
- **Tailwind CSS**: Provides a modern, premium look for the control panels.

### 2. Computing: TensorFlow.js + Web Workers
- **Inference & Training**: All heavy neural network computations are offloaded to a **Web Worker** (`inferenceWorker.ts`). This ensures the interface remains fluid (60 FPS) even during active training.
- **Backend**: Uses the `cpu` backend inside the worker for stability and compatibility across different browsers and hardware.

### 3. Intelligence: Google Gemini AI
- **AI Mentor**: Integrated `gemini-1.5-flash` model for real-time education.
- **Contextual Awareness**: The AI receives the current visualization state (active layer, training metrics, epoch) to provide relevant answers.
- **Communication**: An asynchronous client implemented in `utils/gemini.ts` communicates with the Google GenAI API.

## Neural Network Structure
The model is a sequential CNN with the following layers:
1.  **Input Layer**: 28x28 grayscale image.
2.  **Conv2D Layer**: 8 filters (3x3 kernel), ReLU activation, and Batch Normalization.
3.  **MaxPooling2D Layer**: 2x2 pooling, reduces spatial dimensions.
4.  **Fully Connected (FC) Layer**: 80 neurons with ReLU activation, Batch Normalization, and Dropout (25%).
5.  **Output Layer**: 10 neurons with Softmax activation (representing digits 0-9).

## Key Functionalities

### Real-Time Training Visualization
- **Activation Maps**: Convolutional and pooling layer outputs are converted to textures in real-time during training.
- **Neuron Glow**: FC and Output layer neurons change glow intensity based on their actual activation values ($0.0$ to $1.0$).
- **Dynamic Connections**: The thickness and color of connection lines between layers reflect the strength of information flow.

### Training Monitor
- **Loss Graph**: Tracks `categoricalCrossentropy` loss. A target value below **0.1** is recommended for stable predictions.
- **Accuracy (Acc) Graph**: Tracks the percentage of correct classifications on the current batch.
- **Epoch Tracking**: Visualizes progress through a cycle of 100 training epochs.

### Synthetic Data Generator
- **Font-to-Tensor**: Instead of static images, the system uses `OffscreenCanvas` to generate 1000 high-quality samples using system fonts (Arial/Sans-serif).
- **Augmentation**: Random rotation, translation, and noise are added to each sample to ensure the model generalizes well to different input styles.

### AI Mentor (Gemini Integration)
- **Problem-Solving**: Helps users understand high Loss values and low Accuracy (Acc).
- **Interactive Chat**: Allows direct questions about the model architecture (e.g., "What does the Conv layer do?").
- **Metrics Analysis**: The `analyzeTrainingState` function automatically interprets graphs and provides advice for hyperparameter optimization.

## Technical Implementation Details

### Weights Synchronization
Since the visualization requires data from intermediate layers (Conv, Pool), the worker uses a dual-model approach:
- **Training Model**: Optimized for speed, returns only the final prediction.
- **Visualization Model**: Compiled with multiple outputs for internal state extraction.
Weights are explicitly synchronized using `model.setWeights(trainModel.getWeights())` after each training step.

### Input Processing
When a user selects a digit or uploads an image:
1.  The image is resized to 28x28.
2.  **Grayscale conversion** is performed ($0.299R + 0.587G + 1.14B$).
3.  **Contrast Enhancement** is applied to ensure thin font lines are clearly visible to convolutional filters.

## Usage Guide
1.  **Initialization**: On load, the network starts with random weights.
2.  **Training**: Open the **Training Monitor** and click **Start**. Watch the red line (Loss) drop.
3.  **Testing**: Once Loss falls below 0.1, use the **MNIST Input** panel to select a digit. The 3D scene will update to show how the network classifies it.
4.  **Inspection**: Click on any 3D layer to focus the camera and view its specific parameters.

---
*Created by the Antigravity team for Advanced Agentic Coding.*
