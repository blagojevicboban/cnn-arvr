<div align="center">
  <img width="1200" alt="CNN 3D Visualization Hero" src="public/cnn-arvr.gif" />
  
  # 🧠 CNN 3D Visualization & Live Training
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)](https://reactjs.org/)
  [![Three.js](https://img.shields.io/badge/Three.js-000000?style=flat&logo=three.js&logoColor=white)](https://threejs.org/)
  [![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/js)
  
  **Interactive 3D neural network visualization with real-time in-browser training.**

  [View Live Demo](https://blagojevicboban.github.io/cnn-arvr/) 🚀
</div>

---

## 📖 About Project

This project is an advanced educational tool designed to provide a visual understanding of what happens "under the hood" of a modern Convolutional Neural Network (CNN). Focused on **MNIST** digit classification, users can trace the flow of information from the raw image to the network's final decision.

The primary goal is to demystify neural networks by showing how features are extracted through convolutions and pooling layers, and how the final dense layers converge towards a classification.

---

## ✨ Key Features

- **🌐 Live 3D Architecture**: Full 3D representation of layers (Input, Conv, Pool, FC, Output) updating in real-time.
- **⚡ Real-time Training**: True in-browser training using **Web Workers**, ensuring a fluid 60 FPS experience.
- **🧬 Dynamic Neurons & Connections**: Neurons and connections change glow intensity and thickness based on live weights and activations.
- **📊 Training Monitor**: Track **Loss** (Categorical Crossentropy) and **Accuracy** curves on interactive charts.
- **🛠️ Interactive HUD**: Adjust connection thickness, toggle activation maps, switch to AR/VR mode, and control camera rotation.
- **📸 Custom Data Collection**: Ability to upload your own images and assign labels for personalized training.

---

## 🛠️ Tech Stack

### Frontend & Rendering
- **React**: State management and UI componentization.
- **Three.js** (@react-three/fiber): Core 3D engine.
- **Drei**: Essential 3D helpers (Text, Line, OrbitControls).
- **Tailwind CSS**: Modern glassmorphism interface design.
- **Lucide React**: Premium icon set.

### Machine Learning & Processing
- **TensorFlow.js**: Training and inference engine.
- **Web Workers API**: Parallel processing of ML models off the main thread.
- **OffscreenCanvas**: Generating high-quality synthetic training samples.

---

## 🚀 Installation & Run Locally

**Prerequisites:** Node.js (v16+)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/blagojevicboban/cnn-arvr.git
   cd cnn-arvr
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

---

## 🎮 How to Use

### 1. Training
Open the **Training Monitor** (Activity icon) and click **Start**. Watch the Loss drop (ideally below 0.1). Wait for at least 10-20 epochs for optimal results.

### 2. Testing
In the **MNIST Input** panel, select one of the provided digits or upload your own image. Observe how the network "recognizes" the digit and which parts of the network activate.

### 3. Customization
Use the bottom panel to control visual elements:
- **Network**: Toggle connection lines and adjust their thickness with the slider.
- **Grid (Matrices)**: Show or hide internal activation maps.
- **AR/VR**: Switch to Augmented/Virtual Reality mode for a more immersive experience.

---

<div align="center">
  <p>Created as part of the <b>Advanced Agentic Coding</b> project.</p>
</div>

