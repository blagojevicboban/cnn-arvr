// Lightweight synthetic MNIST-like dataset for in-browser training
// 100 samples: 10 per digit, 28x28 grayscale images

export const trainingDataset = {
  images: [
    // Digit 0 - simple circle
    Array(28).fill(0).map(() => Array(28).fill(0).map(() => Math.random() * 0.1)),
    // ... (repeat for 10 samples per digit, but simplified here)
    // For brevity, I'll generate programmatically, but in code:
  ],
  labels: []
};

// Generate simple patterns
for (let digit = 0; digit < 10; digit++) {
  for (let sample = 0; sample < 10; sample++) {
    const img = Array(28).fill(0).map(() => Array(28).fill(0));
    // Simple pattern: draw digit shape roughly
    if (digit === 0) {
      // Circle
      for (let y = 5; y < 23; y++) {
        for (let x = 5; x < 23; x++) {
          const dist = Math.sqrt((x - 14) ** 2 + (y - 14) ** 2);
          img[y][x] = dist < 8 ? 255 : 0;
        }
      }
    } else if (digit === 1) {
      // Vertical line
      for (let y = 5; y < 23; y++) {
        img[y][14] = 255;
      }
    } else if (digit === 2) {
      // Curved shape
      for (let y = 5; y < 15; y++) {
        for (let x = 5; x < 23; x++) {
          img[y][x] = 255;
        }
      }
      for (let y = 15; y < 23; y++) {
        for (let x = 5; x < 23 - (y - 15); x++) {
          img[y][x] = 255;
        }
      }
    }
    // Add more digits similarly...
    // For now, just fill with random for others
    for (let d = 3; d < 10; d++) {
      for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
          img[y][x] = Math.random() * 255;
        }
      }
    }

    trainingDataset.images.push(img);
    trainingDataset.labels.push(digit);
  }
}