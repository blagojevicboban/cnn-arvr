
// Utilities for running a small TensorFlow.js CNN on a WebWorker.
// The worker handles all computations (convolution, pooling, inference) off the
// main thread and returns intermediate maps along with final probabilities.

let inferenceWorker: Worker | null = null;

function getWorker() {
  if (!inferenceWorker) {
    inferenceWorker = new Worker(new URL('../workers/inferenceWorker.ts', import.meta.url), { type: 'module' });
  }
  return inferenceWorker;
}

function tensorDataToMaps(data: Float32Array, shape: [number, number, number, number]): string[] {
  const [, h, w, c] = shape;
  const maps: string[] = [];
  for (let k = 0; k < c; k++) {
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d')!;
    const imgData = ctx.createImageData(w, h);

    let min = Infinity, max = -Infinity;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * c + k;
        const v = data[idx];
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    const range = max - min || 1;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * c + k;
        let v = (data[idx] - min) / range;
        const p = Math.floor(v * 255);
        const offset = (y * w + x) * 4;
        imgData.data[offset] = p;
        imgData.data[offset + 1] = p;
        imgData.data[offset + 2] = p;
        imgData.data[offset + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);
    maps.push(canvas.toDataURL());
  }
  return maps;
}

export async function processImage(imageUrl: string) {
  const worker = getWorker();
  return new Promise<{
    convMaps: string[];
    poolMaps: string[];
    outputProbs: number[];
  }>((resolve) => {
    const listener = (e: MessageEvent<any>) => {
      worker.removeEventListener('message', listener);
      const { convData, convShape, poolData, poolShape, outputProbs } = e.data;
      const convMaps = tensorDataToMaps(convData, convShape);
      const poolMaps = tensorDataToMaps(poolData, poolShape);
      resolve({ convMaps, poolMaps, outputProbs: Array.from(outputProbs) });
    };
    worker.addEventListener('message', listener);
    worker.postMessage({ type: 'inference', imageUrl });
  });
}

export async function startTraining(dataset: { images: number[][][]; labels: number[] }, epochs: number = 10, batchSize: number = 32) {
  const worker = getWorker();
  worker.postMessage({ type: 'startTraining', dataset, epochs, batchSize });
}

export async function pauseTraining() {
  const worker = getWorker();
  worker.postMessage({ type: 'pauseTraining' });
}

export async function resumeTraining() {
  const worker = getWorker();
  worker.postMessage({ type: 'resumeTraining' });
}

export async function saveCheckpoint() {
  const worker = getWorker();
  return new Promise<string>((resolve) => {
    const listener = (e: MessageEvent<any>) => {
      if (e.data.type === 'checkpointSaved') {
        worker.removeEventListener('message', listener);
        resolve(e.data.checkpoint);
      }
    };
    worker.addEventListener('message', listener);
    worker.postMessage({ type: 'saveCheckpoint' });
  });
}

export async function loadCheckpoint(checkpoint: string) {
  const worker = getWorker();
  worker.postMessage({ type: 'loadCheckpoint', checkpoint });
}

export async function exportTrainingHistory() {
  const worker = getWorker();
  return new Promise<{ step: number; loss: number; accuracy: number }[]>((resolve) => {
    const listener = (e: MessageEvent<any>) => {
      if (e.data.type === 'historyExported') {
        worker.removeEventListener('message', listener);
        resolve(e.data.history);
      }
    };
    worker.addEventListener('message', listener);
    worker.postMessage({ type: 'exportHistory' });
  });
}
