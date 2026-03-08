import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-cpu';

// force CPU backend immediately since WebGL isn't available inside worker
(async () => {
  try {
    await tf.setBackend('cpu');
    await tf.ready();
    // console.log('TensorFlow backend in worker:', tf.getBackend());
  } catch {}
})();

interface WorkerMessage {
  type: 'inference' | 'startTraining' | 'pauseTraining' | 'resumeTraining' | 'saveCheckpoint' | 'loadCheckpoint' | 'exportHistory';
  imageUrl?: string;
  dataset?: { images: number[][][]; labels: number[] }; // small dataset for training
  epochs?: number;
  batchSize?: number;
}

interface WorkerResult {
  type: 'inference' | 'trainingUpdate' | 'checkpointSaved' | 'checkpointLoaded' | 'historyExported';
  convData?: Float32Array;
  convShape?: [number, number, number, number];
  poolData?: Float32Array;
  poolShape?: [number, number, number, number];
  outputProbs?: Float32Array;
  history?: { step: number; loss: number; accuracy: number }[];
  checkpoint?: string; // JSON string of model weights
}


let model: any = null; // Use any to avoid type issues with lazy loading
let isTraining = false;
let trainingHistory: { step: number; loss: number; accuracy: number }[] = [];
let optimizer: any = null;
let dataset: { images: any; labels: any } | null = null;

async function createModel() {
  // ensure worker-friendly backend (WebGL often unavailable inside workers)
  if (tf.getBackend() !== 'cpu') {
    await tf.setBackend('cpu');
    await tf.ready();
  }

  // TensorFlow is available via the static import above
  const input = tf.input({ shape: [28, 28, 1] });
  const conv1 = tf.layers
    .conv2d({ filters: 6, kernelSize: 5, strides: 1, activation: 'relu', padding: 'valid' })
    .apply(input) as any;
  const pool1 = tf.layers
    .maxPooling2d({ poolSize: [2, 2], strides: [2, 2], padding: 'valid' })
    .apply(conv1) as any;
  const flatten = tf.layers.flatten().apply(pool1) as any;
  const fc = tf.layers.dense({ units: 120, activation: 'relu' }).apply(flatten) as any;
  const output = tf.layers.dense({ units: 10, activation: 'softmax' }).apply(fc) as any;

  model = tf.model({ inputs: input, outputs: [conv1, pool1, output] });
  optimizer = tf.train.adam(0.001);
  model.compile({ optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
}

async function prepareDataset(rawDataset: { images: any; labels: any }) {
  // TensorFlow already imported
  if (!rawDataset || !Array.isArray(rawDataset.images) || rawDataset.images.length === 0) {
    console.error('prepareDataset received bad dataset', rawDataset);
    throw new Error('Invalid training dataset supplied to worker');
  }

  // enforce each sample is a 2D array of numbers
  const cleanedImages: number[][][] = rawDataset.images.map((img: any, idx: number) => {
    if (!Array.isArray(img)) {
      console.warn(`sample ${idx} is not an array`, img);
      return [];
    }
    return img.map((row: any, rIdx: number) => {
      if (!Array.isArray(row)) {
        console.warn(`sample ${idx} row ${rIdx} not array`, row);
        return [];
      }
      return row.map((v: any) => (typeof v === 'number' ? v : 0));
    });
  });

  // validate dimensions
  const dimsOk = cleanedImages.every((img) => img.length === 28 && img.every((row) => row.length === 28));
  if (!dimsOk) {
    console.error('cleanedImages has wrong dimensions', cleanedImages.map(i=>i.length));
    throw new Error('Training images must be 28x28 arrays');
  }

  // ensure shape [N,28,28,1]
  const imagesTensor = tf.tensor3d(cleanedImages as any);
  const imagesReshaped = imagesTensor.reshape([cleanedImages.length, 28, 28, 1]).div(255); // normalize
  imagesTensor.dispose();
  const labels = tf.oneHot(tf.tensor1d(rawDataset.labels.map((v: any) => (typeof v === 'number' ? v : 0)) as any, 'int32'), 10);
  dataset = { images: imagesReshaped, labels };
}

async function trainStep(batchSize: number) {
  // TensorFlow already imported
  if (!dataset || !model || !isTraining) return;

  const { images, labels } = dataset;
  const numSamples = images.shape[0];
  const indices = tf.util.createShuffledIndices(numSamples);
  const shuffledImages = tf.gather(images, indices);
  const shuffledLabels = tf.gather(labels, indices);

  for (let i = 0; i < numSamples; i += batchSize) {
    if (!isTraining) break; // allow pausing
    const batchImages = shuffledImages.slice(i, batchSize);
    const batchLabels = shuffledLabels.slice(i, batchSize);

    const history = await model.fit(batchImages, batchLabels, { epochs: 1, verbose: 0 });
    const loss = history.history.loss[0] as number;
    const accuracy = history.history.acc[0] as number;
    trainingHistory.push({ step: trainingHistory.length + 1, loss, accuracy });

    // Send update to main thread
    postMessage({ type: 'trainingUpdate', history: trainingHistory.slice(-1) });

    tf.dispose([batchImages, batchLabels]);
  }

  tf.dispose([shuffledImages, shuffledLabels]);
}

async function imageToTensor(img: HTMLImageElement | ImageBitmap) {
  // TensorFlow already imported
  const off = new OffscreenCanvas(28, 28);
  const ctx = off.getContext('2d')!;
  // both HTMLImageElement and ImageBitmap are drawable
  ctx.drawImage(img as any, 0, 0, 28, 28);
  const imageData = ctx.getImageData(0, 0, 28, 28);
  let tensor = tf.browser.fromPixels(imageData, 1).toFloat().div(255).expandDims(0);
  return tensor;
}

async function handleMessage(e: MessageEvent<WorkerMessage>) {
  // TensorFlow already imported
  const { type } = e.data;

  if (type === 'inference') {
    const { imageUrl } = e.data;
    if (!model) await createModel();

    // `Image` is not available in web workers; use fetch + ImageBitmap
    const resp = await fetch(imageUrl);
    const blob = await resp.blob();
    const bitmap = await createImageBitmap(blob);
    const inputTensor = await imageToTensor(bitmap);
    const [conv1, pool1, output] = model!.predict(inputTensor) as any[];

    const convData = conv1.dataSync() as Float32Array;
    const poolData = pool1.dataSync() as Float32Array;
    const outputProbs = output.dataSync() as Float32Array;

    const result: WorkerResult = {
      type: 'inference',
      convData,
      convShape: conv1.shape as [number, number, number, number],
      poolData,
      poolShape: pool1.shape as [number, number, number, number],
      outputProbs,
    };

    (postMessage as any)(result, [convData.buffer, poolData.buffer, outputProbs.buffer]);
    tf.dispose([inputTensor, conv1, pool1, output]);

  } else if (type === 'startTraining') {
    const { dataset: rawDataset, epochs = 10, batchSize = 32 } = e.data;
    if (!model) await createModel();
    await prepareDataset(rawDataset!);
    isTraining = true;
    trainingHistory = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      if (!isTraining) break;
      await trainStep(batchSize);
    }

  } else if (type === 'pauseTraining') {
    isTraining = false;

  } else if (type === 'resumeTraining') {
    if (dataset && model) {
      isTraining = true;
      await trainStep(32); // resume with default batch size
    }

  } else if (type === 'saveCheckpoint') {
    if (model) {
      const weights = model.getWeights();
      const checkpoint = JSON.stringify(weights.map((w: any) => w.dataSync()));
      postMessage({ type: 'checkpointSaved', checkpoint });
    }

  } else if (type === 'loadCheckpoint') {
    const { checkpoint } = e.data;
    if (model && checkpoint) {
      const parsedWeights = JSON.parse(checkpoint);
      const weights = model.getWeights();
      weights.forEach((w: any, i: number) => {
        w.assign(tf.tensor(parsedWeights[i]));
      });
      postMessage({ type: 'checkpointLoaded' });
    }

  } else if (type === 'exportHistory') {
    postMessage({ type: 'historyExported', history: trainingHistory });
  }
}

onmessage = handleMessage;
