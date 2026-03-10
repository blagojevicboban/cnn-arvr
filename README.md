<div align="center">
<img width="1200" alt="CNN 3D Visualization Hero" src="public/cnn-arvr.gif" />

# CNN 3D Visualization & LIVE Training (v1.1.0)
**CNN 3D Visualization** is an open-source interactive platform designed for education and research of convolutional neural networks. It allows users to monitor model training in real-time directly in the browser, visualize the flow of information through a 5-layer architecture, and experiment with their own datasets.

[![CNN Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=for-the-badge&logo=vercel)](https://blagojevicboban.github.io/cnn-arvr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
</div>

---

### 🌐 [Live Demo: blagojevicboban.github.io/cnn-arvr](https://blagojevicboban.github.io/cnn-arvr/)

This platform implements the key pillars of modern ML visualization:

### ✅ Result 1: Interactive 3D Ecosystem
- **Layer-by-Layer Inspection**: Each layer (Input, Conv, Pool, FC, Output) is displayed as a physical entity in 3D space.
- **Activation Maps**: Outputs of convolutional filters are rendered as dynamic textures that update in real-time during inference and training.
- **Neural Glow**: The light intensity of neurons in FC layers directly reflects their activation value ($0.0$ to $1.0$).
- **Dynamic Connections**: The thickness and color of lines between layers visualize the strength and direction of information flow.

### ✅ Result 2: In-Browser Training (TF.js)
- **Client-Side Computing**: Complete training and inference are executed within the user's browser using TensorFlow.js.
- **Web Worker Parallelization**: All heavy ML computations are offloaded to a separate worker thread, allowing for a fluid 60 FPS for 3D visualization even during intensive training.
- **Dual-Model Synchronization**: The system uses two models - one optimized for training speed and another for extracting internal activations for visualization.

### ✅ Result 3: 8x8 FC Matrix Representation
- **Structural Alignment**: The Fully Connected (FC) layer is represented as a structured 8x8 matrix (64 neurons) for better spatial organization.
- **Full Connectivity Visualization**: Optimized sampling algorithms ensure that every single neuron in the 8x8 matrix shows visual data flow from the pooling layer, eliminating "dead zones".

### ✅ Result 4: Multilingual & Visual Contrast
- **EN/RS Toggle**: Instant switching between English (default) and Serbian languages for all UI elements and the AI Mentor.
- **Visual Contrast Mode**: High-contrast toggle to enhance the visibility of active neurons and connections, making the learning process more apparent.

### ✅ Result 5: Dynamic Data Collection
- **Dataset Builder**: Users can create their own training sets by uploading images or using built-in MNIST samples.
- **Interactive Labeling**: A simple interface for assigning labels (0-9) and instant conversion to tensor formats.
- **Real-Time Augmentation**: The system automatically performs grayscale conversion, resizing (28x28), and contrast enhancement for optimal results.

### ✅ Result 6: Visual Performance Monitor
- **Real-time Recharts**: Integrated charts track Loss and Accuracy through epochs.
- **Checkpoints**: Automatic saving of best models to the browser's `localStorage`, allowing training to resume after a page refresh.
- **Status Console**: Detailed insight into Web Worker state and training progress.

### ✅ Result 7: Gemini AI Mentor
- **Context-Aware Assistance**: Chat with an AI that knows your current training metrics and active layer.
- **Interactive Explanations**: Ask technical questions like "What does a convolution layer do?" and get instant expert answers.
- **Optimization Tips**: Get real-time advice on how to improve your model's accuracy and reduce loss.

---

## 🚀 Key Features
- **3D Rendering**: Powered by **React Three Fiber** and **Three.js** for top-tier performance.
- **Synthetic Generator**: Generating thousands of samples using system fonts and `OffscreenCanvas`.
- **Responsive UI**: A modern interface with a glassmorphism effect built using **Tailwind CSS**.
- **Weight Initialization**: Visual confirmation of transformation from random noise into recognizable filters.
- **Gemini AI Integration**: Capability to use Google GenAI for result analysis and explaining neural network concepts.

## 🛠 Tech Stack
- **Frontend**: React 19, Three.js, React Three Fiber, React Three Drei
- **ML Engine**: TensorFlow.js (CPU/Core backend in worker)
- **Styling**: Tailwind CSS 4.0
- **Charts**: Recharts
- **Build Tool**: Vite 6.0
- **Icons**: Lucide React

## 💻 Local Setup

### 1. Prerequisites
- **Node.js** (v18+)
- **NPM** or **Yarn**

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/blagojevicboban/cnn-arvr.git
cd cnn-arvr
npm install
```

### 3. Configuration
Set `GEMINI_API_KEY` in your `.env` file if you plan to use Google GenAI features:
```env
VITE_GEMINI_API_KEY=your_api_key
```

### 4. Running the Development Server
```bash
npm run dev
```
The application will be available at `http://localhost:3000`.

---

## 🌍 Deployment

The project is optimized for **GitHub Pages**. To deploy your own version:
1. Configure `base` in `vite.config.ts`.
2. Run:
   ```bash
   npm run deploy
   ```

## 🐛 Troubleshooting
- **Black Screen**: Check the browser console. Most often, the cause is a failed initialization of `WebGPU` or `WebGL` context. The application primarily uses the `CPU` backend inside the worker for stability.
- **Gemini Error**: Check if your API key is valid and if you have CORS configured if accessing from unauthorized domains.
- **Performance Lag**: Close other tabs that use the GPU. Although the worker handles heavy computations, Three.js still requires resources for 60 FPS rendering.

## 🤝 Contributing
Contributions are always welcome! If you have an idea for improving the visualization or adding new layers, feel free to open an **Issue** or submit a **Pull Request**.

## 📄 License
This project is licensed under the **MIT** license - see the [LICENSE](LICENSE) file for details.
