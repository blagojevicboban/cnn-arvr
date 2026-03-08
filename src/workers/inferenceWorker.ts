// @ts-ignore - types will be available once @tensorflow/tfjs is installed
// import * as tf from '@tensorflow/tfjs';

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

let tf: any = null;

async function loadTensorFlow() {
  if (!tf) {
    tf = await import('@tensorflow/tfjs');
  }
  return tf;
}

let model: tf.LayersModel | null = null;
let isTraining = false;
let trainingHistory: { step: number; loss: number; accuracy: number }[] = [];
let optimizer: tf.Optimizer | null = null;
let dataset: { images: tf.Tensor4D; labels: tf.Tensor2D } | null = null;

async function createModel() {
  const tf = await loadTensorFlow();
  
  const input = tf.input({ shape: [28, 28, 1] });
  const conv1 = tf.layers
    .conv2d({ filters: 6, kernelSize: 5, strides: 1, activation: 'relu', padding: 'valid' })
    .apply(input) as tf.SymbolicTensor;
  const pool1 = tf.layers
    .maxPooling2d({ poolSize: [2, 2], strides: [2, 2], padding: 'valid' })
    .apply(conv1) as tf.SymbolicTensor;
  const flatten = tf.layers.flatten().apply(pool1) as tf.SymbolicTensor;
  const fc = tf.layers.dense({ units: 120, activation: 'relu' }).apply(flatten) as tf.SymbolicTensor;
  const output = tf.layers.dense({ units: 10, activation: 'softmax' }).apply(fc) as tf.SymbolicTensor;

  model = tf.model({ inputs: input, outputs: [conv1, pool1, output] });
  optimizer = tf.train.adam(0.001);
  model.compile({ optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
}

async function prepareDataset(rawDataset: { images: number[][][]; labels: number[] }) {
  const tf = await loadTensorFlow();
  
  const images = tf.tensor4d(rawDataset.images).div(255); // normalize
  const labels = tf.oneHot(tf.tensor1d(rawDataset.labels, 'int32'), 10);
  dataset = { images, labels };
}

async function trainStep(batchSize: number) {
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

function imageToTensor(img: HTMLImageElement) {
  const off = new OffscreenCanvas(28, 28);
  const ctx = off.getContext('2d')!;
  ctx.drawImage(img, 0, 0, 28, 28);
  const imageData = ctx.getImageData(0, 0, 28, 28);
  let tensor = tf.browser.fromPixels(imageData, 1).toFloat().div(255).expandDims(0);
  return tensor;
}

async function handleMessage(e: MessageEvent<WorkerMessage>) {
  const { type } = e.data;

  if (type === 'inference') {
    const { imageUrl } = e.data;
    if (!model) await createModel();

    const img = new Image();
    img.crossOrigin = 'Anonymous';
    img.src = imageUrl;
    await new Promise<void>((res) => { img.onload = () => res(); });

    const inputTensor = imageToTensor(img);
    const [conv1, pool1, output] = model!.predict(inputTensor) as tf.Tensor[];

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
    prepareDataset(rawDataset);
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
      const checkpoint = JSON.stringify(weights.map(w => w.dataSync()));
      postMessage({ type: 'checkpointSaved', checkpoint });

    } else if (type === 'loadCheckpoint') {
      const { checkpoint } = e.data;
      if (model && checkpoint) {
        const parsedWeights = JSON.parse(checkpoint);
        const weights = model.getWeights();
        weights.forEach((w, i) => {
          w.assign(tf.tensor(parsedWeights[i]));
        });
        postMessage({ type: 'checkpointLoaded' });
      }

    } else if (type === 'exportHistory') {
      postMessage({ type: 'historyExported', history: trainingHistory });
    }
  }
}

onmessage = handleMessage;
