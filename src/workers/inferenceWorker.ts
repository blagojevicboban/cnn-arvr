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
  checkpoint?: string;
}

interface WorkerResult {
  type: 'inference' | 'trainingUpdate' | 'checkpointSaved' | 'checkpointLoaded' | 'historyExported';
  convData?: Float32Array;
  convShape?: [number, number, number, number];
  poolData?: Float32Array;
  poolShape?: [number, number, number, number];
  fcData?: number[];
  outputProbs?: Float32Array;
  history?: { step: number; loss: number; accuracy: number }[];
  checkpoint?: string; // JSON string of model weights
}


let model: any = null; 
let trainModel: any = null; 
let isTraining = false;
let trainingHistory: { step: number; loss: number; accuracy: number }[] = [];
let optimizer: any = null;
let dataset: { images: any; labels: any } | null = null;
let currentEpoch = 0;

async function createModel() {
  // ensure worker-friendly backend (WebGL often unavailable inside workers)
  if (tf.getBackend() !== 'cpu') {
    await tf.setBackend('cpu');
    await tf.ready();
  }

  // TensorFlow is available via the static import above
  const input = tf.input({ shape: [28, 28, 1] });
  const conv1 = tf.layers
    .conv2d({ 
      filters: 8, 
      kernelSize: 3, 
      strides: 1, 
      activation: 'relu', 
      padding: 'same',
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
    })
    .apply(input) as any;
  const bn1 = tf.layers.batchNormalization().apply(conv1) as any;
  const pool1 = tf.layers
    .maxPooling2d({ poolSize: [2, 2], strides: [2, 2] })
    .apply(bn1) as any;
    
  const flatten = tf.layers.flatten().apply(pool1) as any;
  const fc = tf.layers.dense({ 
    units: 64, 
    activation: 'relu',
    kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
  }).apply(flatten) as any;
  const bn2 = tf.layers.batchNormalization().apply(fc) as any;
  const dropout = tf.layers.dropout({ rate: 0.2 }).apply(bn2) as any;
  const output = tf.layers.dense({ units: 10, activation: 'softmax' }).apply(dropout) as any;

  model = tf.model({ inputs: input, outputs: [conv1, pool1, fc, output] });
  trainModel = tf.model({ inputs: input, outputs: output });
  optimizer = tf.train.adam(0.0005); 
  trainModel.compile({ optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
}

async function prepareDataset(rawDataset: { images: any; labels: any }) {
  if (dataset) {
    dataset.images.dispose();
    dataset.labels.dispose();
  }

  // GENERATE 1000 HIGH QUALITY SAMPLES USING CANVAS
  const numSamples = 1000;
  const flattened = new Float32Array(numSamples * 784);
  const labelsArr = new Int32Array(numSamples);
  
  const canvas = new OffscreenCanvas(28, 28);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;

  for (let i = 0; i < numSamples; i++) {
    const digit = i % 10;
    labelsArr[i] = digit;
    
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, 28, 28);
    
    // Add tiny bit of background noise to prevent overfitting
    if (Math.random() > 0.5) {
      for(let n=0; n<5; n++) {
        ctx.fillStyle = `rgba(255,255,255,${Math.random()*0.1})`;
        ctx.fillRect(Math.random()*28, Math.random()*28, 2, 2);
      }
    }

    ctx.fillStyle = 'white';
    ctx.font = `bold ${17 + Math.random() * 5}px Arial, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    ctx.save();
    ctx.translate(14 + (Math.random()-0.5)*3, 14 + (Math.random()-0.5)*3);
    ctx.rotate((Math.random() - 0.5) * 0.3);
    ctx.fillText(digit.toString(), 0, 0);
    ctx.restore();
    
    const imgData = ctx.getImageData(0, 0, 28, 28).data;
    for (let j = 0; j < 784; j++) {
      flattened[i * 784 + j] = imgData[j * 4] / 255;
    }
  }

  const imagesReshaped = tf.tensor4d(flattened, [numSamples, 28, 28, 1]);
  const labels = tf.oneHot(tf.tensor1d(labelsArr, 'int32'), 10);
  dataset = { images: imagesReshaped, labels };
}

async function trainStep(batchSize: number, currentEpoch: number) {
  // TensorFlow already imported
  if (!dataset || !model || !isTraining) return;

  const { images, labels } = dataset;
  const numSamples = images.shape[0];
  const indicesArray = tf.util.createShuffledIndices(numSamples);
  const indices = tf.tensor1d(Array.from(indicesArray), 'int32');
  const shuffledImages = tf.gather(images, indices);
  const shuffledLabels = tf.gather(labels, indices);
  indices.dispose();

  for (let i = 0; i < numSamples; i += batchSize) {
    if (!isTraining) break; 
    const actualBatchSize = Math.min(batchSize, numSamples - i);
    const batchImages = tf.slice(shuffledImages, [i, 0, 0, 0], [actualBatchSize, 28, 28, 1]);
    const batchLabels = tf.slice(shuffledLabels, [i, 0], [actualBatchSize, 10]);

    const historyResponse = await trainModel.fit(batchImages, batchLabels, { epochs: 1, verbose: 0 });
    
    // Explicitly sync weights from trainModel to visualization model
    model.setWeights(trainModel.getWeights());
    
    const loss = historyResponse.history.loss[0] as number;
    const accuracy = (historyResponse.history.acc ? historyResponse.history.acc[0] : (historyResponse.history.accuracy ? historyResponse.history.accuracy[0] : 0)) as number;
    
    trainingHistory.push({ step: trainingHistory.length + 1, loss, accuracy });

    // --- Visualization during training ---
    // Take the first image of this batch to show it in the 3D Scene
    const singleImage = tf.slice(batchImages, [0, 0, 0, 0], [1, 28, 28, 1]);
    const [conv1, pool1, fc, output] = model!.predict(singleImage) as any[];
    
    const convData = conv1.dataSync() as Float32Array;
    const poolData = pool1.dataSync() as Float32Array;
    const fcData = fc.dataSync() as Float32Array;
    const outputProbs = output.dataSync() as Float32Array;
    const currentInputImage = singleImage.dataSync() as Float32Array;

    // Send update to main thread
    postMessage({ 
      type: 'trainingUpdate', 
      history: trainingHistory.slice(-1),
      epoch: currentEpoch + 1,
      // Visualization data for this specific step
      visualization: {
        convData,
        convShape: conv1.shape as [number, number, number, number],
        poolData,
        poolShape: pool1.shape as [number, number, number, number],
        fcData: Array.from(fcData),
        outputProbs: Array.from(outputProbs),
        inputData: Array.from(currentInputImage)
      }
    });

    tf.dispose([batchImages, batchLabels, singleImage, conv1, pool1, fc, output]);
  }

  tf.dispose([shuffledImages, shuffledLabels]);
}

async function imageToTensor(img: ImageBitmap) {
  const off = new OffscreenCanvas(28, 28);
  const ctx = off.getContext('2d', { willReadFrequently: true })!;
  
  // Fill black background just in case of transparency
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, 28, 28);
  ctx.drawImage(img, 0, 0, 28, 28);
  
  const imageData = ctx.getImageData(0, 0, 28, 28);
  const data = new Float32Array(28 * 28);
  
  for (let i = 0; i < 28 * 28; i++) {
    const r = imageData.data[i * 4];
    const g = imageData.data[i * 4 + 1];
    const b = imageData.data[i * 4 + 2];
    // Grayscale + Contrast Boost
    let v = (r * 0.299 + g * 0.587 + b * 0.114) / 255;
    // Push values towards 0 or 1 to help the model with font-based input
    data[i] = v > 0.1 ? Math.min(1.0, v * 1.5) : 0;
  }
  
  return tf.tensor4d(data, [1, 28, 28, 1]);
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
    const [conv1, pool1, fc, output] = model!.predict(inputTensor) as any[];

    const convData = conv1.dataSync() as Float32Array;
    const poolData = pool1.dataSync() as Float32Array;
    const fcData = fc.dataSync() as Float32Array;
    const outputProbs = output.dataSync() as Float32Array;

    const result: WorkerResult = {
      type: 'inference',
      convData,
      convShape: Array.from(conv1.shape) as [number, number, number, number],
      poolData,
      poolShape: Array.from(pool1.shape) as [number, number, number, number],
      fcData: Array.from(fcData),
      outputProbs,
    };

    (postMessage as any)(result, [convData.buffer, poolData.buffer, outputProbs.buffer]);
    tf.dispose([inputTensor, conv1, pool1, fc, output]);

  } else if (type === 'startTraining') {
    const { dataset: rawDataset, epochs = 10, batchSize = 32 } = e.data;
    // Always recreate model on start to reset weights/biases
    await createModel();
    await prepareDataset(rawDataset!);
    isTraining = true;
    trainingHistory = [];
    currentEpoch = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      if (!isTraining) break;
      currentEpoch = epoch;
      await trainStep(batchSize, currentEpoch);
    }

  } else if (type === 'pauseTraining') {
    isTraining = false;

  } else if (type === 'resumeTraining') {
    if (dataset && model) {
      isTraining = true;
      await trainStep(32, currentEpoch); // resume with default batch size and current epoch
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
