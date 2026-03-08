// High-quality synthetic MNIST-like dataset generator
// 800 samples: 80 per digit, with rotation, scaling and varying thickness

export const trainingDataset: { images: number[][][]; labels: number[] } = {
  images: [],
  labels: []
};

for (let digit = 0; digit < 10; digit++) {
  for (let sample = 0; sample < 80; sample++) {
    const img = Array(28).fill(0).map(() => Array(28).fill(0));
    
    // Random style parameters
    const offsetX = (Math.random() * 4) - 2;
    const offsetY = (Math.random() * 4) - 2;
    const angle = (Math.random() - 0.5) * 0.4; // Random rotation up to 12 degrees
    const scale = 0.9 + Math.random() * 0.2; // Slight scaling
    
    const drawPoint = (x: number, y: number) => {
      // Rotate and scale point
      const rx = (x - 14) * Math.cos(angle) - (y - 14) * Math.sin(angle);
      const ry = (x - 14) * Math.sin(angle) + (y - 14) * Math.cos(angle);
      
      const fx = Math.floor(14 + rx * scale + offsetX);
      const fy = Math.floor(14 + ry * scale + offsetY);
      
      if (fx >= 0 && fx < 28 && fy >= 0 && fy < 28) {
        img[fy][fx] = 255;
        // Blur/Thickness
        if (fx + 1 < 28) img[fy][fx+1] = Math.max(img[fy][fx+1], 150);
        if (fy + 1 < 28) img[fy+1][fx] = Math.max(img[fy+1][fx], 150);
      }
    };

    const drawLine = (x1: number, y1: number, x2: number, y2: number) => {
      const dist = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
      for (let i = 0; i <= dist; i++) {
        const t = i / dist;
        drawPoint(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t);
      }
    };

    // Draw procedural shapes
    if (digit === 0) {
      for (let a = 0; a < Math.PI * 2; a += 0.1) drawPoint(14 + Math.cos(a) * 8, 14 + Math.sin(a) * 8);
    } else if (digit === 1) {
      drawLine(14, 5, 14, 23);
      drawLine(11, 8, 14, 5);
    } else if (digit === 2) {
      drawLine(8, 8, 20, 8); drawLine(20, 8, 8, 22); drawLine(8, 22, 20, 22);
    } else if (digit === 3) {
      drawLine(8, 8, 20, 8); drawLine(20, 8, 14, 15); drawLine(14, 15, 20, 22); drawLine(20, 22, 8, 22);
    } else if (digit === 4) {
      drawLine(8, 8, 8, 15); drawLine(8, 15, 22, 15); drawLine(17, 8, 17, 24);
    } else if (digit === 5) {
      drawLine(20, 8, 8, 8); drawLine(8, 8, 8, 15); drawLine(8, 15, 20, 15); drawLine(20, 15, 20, 22); drawLine(20, 22, 8, 22);
    } else if (digit === 6) {
      drawLine(20, 8, 8, 15);
      for (let a = 0; a < Math.PI * 2; a += 0.2) drawPoint(14 + Math.cos(a) * 6, 19 + Math.sin(a) * 6);
    } else if (digit === 7) {
      drawLine(8, 8, 20, 8); drawLine(20, 8, 10, 24);
    } else if (digit === 8) {
      for (let a = 0; a < Math.PI * 2; a += 0.2) {
        drawPoint(14 + Math.cos(a) * 5, 10 + Math.sin(a) * 5);
        drawPoint(14 + Math.cos(a) * 6, 19 + Math.sin(a) * 6);
      }
    } else if (digit === 9) {
      for (let a = 0; a < Math.PI * 2; a += 0.2) drawPoint(14 + Math.cos(a) * 6, 10 + Math.sin(a) * 6);
      drawLine(20, 10, 14, 24);
    }

    trainingDataset.images.push(img);
    trainingDataset.labels.push(digit);
  }
}