
// Simple image processing utilities to simulate CNN layers

export async function processImage(imageUrl: string) {
  // 1. Load Image
  const img = new Image();
  img.crossOrigin = "Anonymous";
  img.src = imageUrl;
  await new Promise((resolve) => { img.onload = resolve; });

  // 2. Draw to Canvas to get data
  const canvas = document.createElement('canvas');
  const size = 28; // Standardize to 28x28 for processing
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0, size, size);
  const inputData = ctx.getImageData(0, 0, size, size);

  // 3. Define Kernels (Edge detection, etc.)
  const kernels = [
    [0, -1, 0, -1, 5, -1, 0, -1, 0], // Sharpen
    [-1, -1, -1, -1, 8, -1, -1, -1, -1], // Edge detection
    [1, 0, -1, 2, 0, -2, 1, 0, -1], // Sobel Horizontal
    [1, 2, 1, 0, 0, 0, -1, -2, -1], // Sobel Vertical
    [1, 1, 1, 1, 1, 1, 1, 1, 1], // Box Blur (normalized later)
    [-2, -1, 0, -1, 1, 1, 0, 1, 2] // Emboss
  ];

  // 4. Generate Conv1 Maps (24x24 output from 28x28 input with 5x5 kernel is valid, 
  // but let's stick to 3x3 kernels for simplicity -> 26x26, or 'same' padding 28x28.
  // The scene config says Conv1 is 24x24. 28 - 5 + 1 = 24. So 5x5 kernel.
  // Let's just resize/resample for visual simplicity to match the target size 24x24.
  
  const convMaps: string[] = [];
  const poolMaps: string[] = [];

  for (let k = 0; k < 6; k++) {
    // Simulate Conv: Apply kernel and resize to 24x24
    const convCanvas = document.createElement('canvas');
    convCanvas.width = 24;
    convCanvas.height = 24;
    const convCtx = convCanvas.getContext('2d')!;
    
    // Apply "filter" (just some composite operations or pixel manipulation)
    // For visualization, we can just tint/contrast the original image
    convCtx.drawImage(canvas, 0, 0, 24, 24);
    const cData = convCtx.getImageData(0, 0, 24, 24);
    
    // Simple pixel manipulation to simulate different filters
    for (let i = 0; i < cData.data.length; i += 4) {
        const r = cData.data[i];
        const g = cData.data[i+1];
        const b = cData.data[i+2];
        const avg = (r + g + b) / 3;
        
        if (k === 0) { // Invert
             cData.data[i] = 255 - r;
             cData.data[i+1] = 255 - g;
             cData.data[i+2] = 255 - b;
        } else if (k === 1) { // High contrast / Threshold
             const v = avg > 128 ? 255 : 0;
             cData.data[i] = v; cData.data[i+1] = v; cData.data[i+2] = v;
        } else if (k === 2) { // Red channel / Heatmap-ish
             cData.data[i] = r; cData.data[i+1] = 0; cData.data[i+2] = 0;
        } else if (k === 3) { // Green channel
             cData.data[i] = 0; cData.data[i+1] = g; cData.data[i+2] = 0;
        } else if (k === 4) { // Blue channel
             cData.data[i] = 0; cData.data[i+1] = 0; cData.data[i+2] = b;
        } else { // Grayscale
             cData.data[i] = avg; cData.data[i+1] = avg; cData.data[i+2] = avg;
        }
    }
    convCtx.putImageData(cData, 0, 0);
    convMaps.push(convCanvas.toDataURL());

    // Simulate Pool: Downsample to 12x12
    const poolCanvas = document.createElement('canvas');
    poolCanvas.width = 12;
    poolCanvas.height = 12;
    const poolCtx = poolCanvas.getContext('2d')!;
    poolCtx.drawImage(convCanvas, 0, 0, 12, 12); // Linear interpolation acts as pooling visually
    poolMaps.push(poolCanvas.toDataURL());
  }

  // 5. Generate Output Probabilities
  // Deterministic "fake" probabilities based on the image URL hash or similar
  // so it's consistent for the same image.
  const hash = imageUrl.split('').reduce((a,b)=>a+b.charCodeAt(0),0);
  const outputProbs = new Array(10).fill(0).map((_, i) => {
      const val = Math.sin(hash + i) * 0.5 + 0.5;
      return val;
  });
  // Normalize
  const sum = outputProbs.reduce((a,b) => a+b, 0);
  const normalizedProbs = outputProbs.map(p => p / sum);

  return {
    convMaps,
    poolMaps,
    outputProbs
  };
}
