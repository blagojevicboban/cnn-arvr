import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars, Environment } from '@react-three/drei';
import { Layer } from './Layer';
import { Connection } from './Connection';
import { DenseConnections } from './DenseConnections';
import { useState, useEffect, useRef } from 'react';
import { Play, Pause, SkipForward, SkipBack, Upload, Settings, X, Activity, ChevronDown, ChevronUp, Eye, EyeOff, Grid3x3, Network, LayoutDashboard, HelpCircle, Info, Box, Zap, Target, Cpu, Layers, Contrast } from 'lucide-react';
import * as THREE from 'three';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { processImage, startTraining, pauseTraining, resumeTraining, saveCheckpoint, loadCheckpoint, exportTrainingHistory, getWorker, tensorDataToMaps } from '../utils/imageProcessor';
import { trainingDataset } from '../utils/trainingDataset';
import { AIMentor } from './AIMentor';
import { Sparkles } from 'lucide-react';
// Sample images (using placeholder services for demo)
const SAMPLE_IMAGES = [
  { id: '0', url: 'https://placehold.co/28x28/000000/FFFFFF/png?text=0', label: '0', type: 'MNIST' },
  { id: '1', url: 'https://placehold.co/28x28/000000/FFFFFF/png?text=1', label: '1', type: 'MNIST' },
  { id: '2', url: 'https://placehold.co/28x28/000000/FFFFFF/png?text=2', label: '2', type: 'MNIST' },
  { id: '3', url: 'https://placehold.co/28x28/000000/FFFFFF/png?text=3', label: '3', type: 'MNIST' },
  { id: '4', url: 'https://placehold.co/28x28/000000/FFFFFF/png?text=4', label: '4', type: 'MNIST' },
  { id: '5', url: 'https://placehold.co/28x28/000000/FFFFFF/png?text=5', label: '5', type: 'MNIST' },
  { id: '6', url: 'https://placehold.co/28x28/000000/FFFFFF/png?text=6', label: '6', type: 'MNIST' },
  { id: '7', url: 'https://placehold.co/28x28/000000/FFFFFF/png?text=7', label: '7', type: 'MNIST' },
  { id: '8', url: 'https://placehold.co/28x28/000000/FFFFFF/png?text=8', label: '8', type: 'MNIST' },
  { id: '9', url: 'https://placehold.co/28x28/000000/FFFFFF/png?text=9', label: '9', type: 'MNIST' },
];

interface LayerConfig {
  id: number;
  type: 'input' | 'conv' | 'pool' | 'fc' | 'output';
  pos: [number, number, number];
  size: [number, number];
  depth: number;
  label: string;
  kernelSize?: number;
  stride?: number;
  filters?: number;
}

export function Scene() {
  const [activeLayer, setActiveLayer] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [selectedImage, setSelectedImage] = useState<string | undefined>(SAMPLE_IMAGES[0].url);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const [layers, setLayers] = useState<LayerConfig[]>([
    { id: 0, type: 'input', pos: [-6, 0, 0], size: [28, 28], depth: 1, label: "Input (28x28)" },
    { id: 1, type: 'conv', pos: [-3, 0, 0], size: [24, 24], depth: 6, label: "Conv1 (6x24x24)", kernelSize: 5, stride: 1, filters: 6 },
    { id: 2, type: 'pool', pos: [0, 0, 0], size: [12, 12], depth: 6, label: "Pool1 (6x12x12)", kernelSize: 2, stride: 2 },
    { id: 3, type: 'fc', pos: [3, 0, 0], size: [80, 1], depth: 1, label: "FC (80)" },
    { id: 4, type: 'output', pos: [6, 0, 0], size: [10, 1], depth: 1, label: "Output (10)" },
  ]);

  const [selectedLayerId, setSelectedLayerId] = useState<number | null>(null);
  
  // Processed Data State
  const [processedData, setProcessedData] = useState<{
    convMaps: string[];
    poolMaps: string[];
    fcActivations: number[];
    outputProbs: number[];
  } | null>(null);

  // UI State
  const [showInputPanel, setShowInputPanel] = useState(false);
  const [showTrainingPanel, setShowTrainingPanel] = useState(false);
  const [showDataPanel, setShowDataPanel] = useState(false);
  const [showHelp, setShowHelp] = useState(true);
  const [lang, setLang] = useState<'sr' | 'en'>('sr');
  const [showStars, setShowStars] = useState(false);
  const [showGrid, setShowGrid] = useState(false);
  const [showConnections, setShowConnections] = useState(true);
  const [showARMode, setShowARMode] = useState(false);
  const [showMatrices, setShowMatrices] = useState(true);
  const [connectionThickness, setConnectionThickness] = useState(2.0);
  const [showAIMentor, setShowAIMentor] = useState(false);
  const [visualContrast, setVisualContrast] = useState(1.0);

  // Training Data Collection
  const [collectedData, setCollectedData] = useState<{ images: number[][][]; labels: number[] }>(trainingDataset);
  const [isDataCollectionMode, setIsDataCollectionMode] = useState(false);
  const [currentLabel, setCurrentLabel] = useState(0);

  // Training State
  const [isTraining, setIsTraining] = useState(false);
  const [trainingHistory, setTrainingHistory] = useState<{ step: number; loss: number; accuracy: number }[]>([]);
  const [trainingStep, setTrainingStep] = useState(0);
  const [epoch, setEpoch] = useState(0);

  useEffect(() => {
    const worker = getWorker();
    const handleMessage = (e: MessageEvent<any>) => {
      if (e.data.type === 'trainingUpdate') {
        const { history, epoch: currentEpoch, visualization } = e.data;
        if (history && history.length > 0) {
          setTrainingHistory(prev => [...prev, ...history]);
          setTrainingStep(prev => prev + history.length);
          if (currentEpoch !== undefined) setEpoch(currentEpoch);

          // Handle real-time visualization update
          if (visualization) {
            const { convData, convShape, poolData, poolShape, fcData, outputProbs, inputData } = visualization;
            
            // Fast convert back to maps
            const convMaps = tensorDataToMaps(new Float32Array(convData), convShape);
            const poolMaps = tensorDataToMaps(new Float32Array(poolData), poolShape);
            const inputMaps = tensorDataToMaps(new Float32Array(inputData), [1, 28, 28, 1]);

            setProcessedData({
              convMaps,
              poolMaps,
              fcActivations: fcData,
              outputProbs: Array.from(outputProbs)
            });
            
            if (inputMaps.length > 0) {
              setSelectedImage(inputMaps[0]);
            }
          }
        }
      } else if (e.data.type === 'error') {
        console.error('Worker Error:', e.data.message);
        alert('Training Error: ' + e.data.message);
        setIsTraining(false);
      }
    };
    worker.addEventListener('message', handleMessage);
    return () => worker.removeEventListener('message', handleMessage);
  }, []);

  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    if (selectedImage) {
      setIsProcessing(true);
      processImage(selectedImage)
        .then(data => {
          setProcessedData(data);
        })
        .finally(() => setIsProcessing(false));
    }
  }, [selectedImage]);

  // Simulate data flow (inference and training flow)
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying || isTraining) {
      interval = setInterval(() => {
        setActiveLayer((prev) => (prev + 1) % 5);
      }, isTraining ? 400 : 1500); // Faster pulse during training to show activity
    }
    return () => clearInterval(interval);
  }, [isPlaying, isTraining]);

  // Simulate training process (now real training)
  useEffect(() => {
    if (isTraining) {
      if (collectedData.images.length > 0) {
        startTraining(collectedData, 100, 32);
      } else {
        alert('Please collect some training data first!');
        setIsTraining(false);
      }
    } else {
      pauseTraining();
    }
  }, [isTraining, collectedData]);

  const toggleTraining = () => {
    setIsTraining(!isTraining);
    setIsPlaying(false); // Stop inference animation when training starts
  };

  const handlePauseResume = () => {
    if (isTraining) {
      pauseTraining();
      setIsTraining(false);
    } else {
      resumeTraining();
      setIsTraining(true);
    }
  };

  const handleSaveCheckpoint = async () => {
    const checkpoint = await saveCheckpoint();
    localStorage.setItem('cnnCheckpoint', checkpoint);
    alert('Checkpoint saved!');
  };

  const handleLoadCheckpoint = async () => {
    const checkpoint = localStorage.getItem('cnnCheckpoint');
    if (checkpoint) {
      await loadCheckpoint(checkpoint);
      alert('Checkpoint loaded!');
    } else {
      alert('No checkpoint found!');
    }
  };

  const handleExportHistory = async () => {
    const history = await exportTrainingHistory();
    const dataStr = JSON.stringify(history, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'training_history.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const result = e.target?.result as string;
        setSelectedImage(result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAddToTrainingData = async (imageUrl: string, label: number) => {
    const img = new Image();
    img.crossOrigin = 'Anonymous';
    img.src = imageUrl;
    await new Promise<void>((res) => { img.onload = () => res(); });

    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(img, 0, 0, 28, 28);
    const imageData = ctx.getImageData(0, 0, 28, 28);

    // Convert to grayscale 28x28 array
    const pixels: number[][] = [];
    for (let y = 0; y < 28; y++) {
      const row: number[] = [];
      for (let x = 0; x < 28; x++) {
        const idx = (y * 28 + x) * 4;
        const r = imageData.data[idx];
        const g = imageData.data[idx + 1];
        const b = imageData.data[idx + 2];
        const gray = Math.round((r + g + b) / 3);
        row.push(gray);
      }
      pixels.push(row);
    }

    setCollectedData(prev => ({
      images: [...prev.images, pixels],
      labels: [...prev.labels, label]
    }));
  };

  const handleNext = () => setActiveLayer((prev) => (prev + 1) % layers.length);
  const handlePrev = () => setActiveLayer((prev) => (prev - 1 + layers.length) % layers.length);
  
  const updateLayerConfig = (id: number, key: keyof LayerConfig, value: number) => {
    setLayers(prev => prev.map(l => l.id === id ? { ...l, [key]: value } : l));
  };

  const selectedLayerConfig = layers.find(l => l.id === selectedLayerId);

  return (
    <>
      <AIMentor 
        isOpen={showAIMentor} 
        onClose={() => setShowAIMentor(false)} 
        context={{
            activeLayer: layers[activeLayer]?.label,
            trainingHistory: trainingHistory,
            currentMetrics: trainingHistory.length > 0 ? trainingHistory[trainingHistory.length - 1] : undefined,
            epoch: epoch,
            step: trainingStep
        }}
      />
      {isProcessing && (
        <div className="absolute inset-0 flex items-center justify-center z-20 pointer-events-none">
          <div className="bg-black/60 text-white px-4 py-2 rounded-lg">
            Processing MNIST digit...
          </div>
        </div>
      )}

      {/* AR/VR HUD Overlay */}
      {showARMode && (
        <div className="absolute inset-0 z-50 pointer-events-none overflow-hidden border-[10px] border-blue-500/10 animate-pulse">
          <div className="absolute top-10 left-10 text-blue-400 font-mono text-[10px] space-y-1">
            <div className="flex items-center gap-2"><div className="w-2 h-2 bg-blue-500 animate-ping"></div> SYSTEM READY</div>
            <div>NET_SCAN: ACTIVE</div>
            <div>LATENCY: 12ms</div>
            <div>MODE: IMMERSIVE_3D</div>
          </div>
          <div className="absolute bottom-10 right-10 text-blue-400/50 font-mono text-[10px]">
            [ HOLO_MATRIX_v1.0.0 ]
          </div>
          {/* Subtle scanline effect */}
          <div className="absolute inset-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.1)_50%),linear-gradient(90deg,rgba(255,0,0,0.03),rgba(0,255,0,0.01),rgba(0,0,255,0.03))] bg-[length:100%_2px,3px_100%] pointer-events-none opacity-20"></div>
        </div>
      )}

      {/* Help Modal */}
      {showHelp && (
        <div className="fixed inset-0 z-[200] bg-black/80 backdrop-blur-lg overflow-y-auto pointer-events-auto flex justify-center items-start p-4 py-10 sm:py-20">
          <div className="bg-[#151515] border border-white/10 p-5 sm:p-10 rounded-3xl max-w-2xl w-full shadow-2xl animate-in zoom-in-95 duration-300 relative flex flex-col gap-6">
            <div className="flex justify-between items-start">
              <div className="flex items-center gap-3">
                <div className="bg-blue-500/20 p-1.5 sm:p-2 rounded-xl">
                  <HelpCircle size={22} className="text-blue-400 sm:w-[28px] sm:h-[28px]" />
                </div>
                <div className="text-left">
                  <h2 className="text-lg sm:text-2xl font-bold text-white tracking-tight leading-none sm:leading-normal">{lang === 'sr' ? 'MNIST CNN Vodič' : 'MNIST CNN Guide'}</h2>
                  <p className="text-gray-400 text-[10px] sm:text-sm">{lang === 'sr' ? 'Kako koristiti 3D simulaciju' : 'How to use simulation'}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex bg-white/5 p-1 rounded-lg border border-white/10">
                  <button 
                    onClick={() => setLang('sr')}
                    className={`px-2 py-1 text-[10px] rounded-md transition-all ${lang === 'sr' ? 'bg-blue-500 text-white' : 'text-gray-400 hover:text-white'}`}
                  >SR</button>
                  <button 
                    onClick={() => setLang('en')}
                    className={`px-2 py-1 text-[10px] rounded-md transition-all ${lang === 'en' ? 'bg-blue-500 text-white' : 'text-gray-400 hover:text-white'}`}
                  >EN</button>
                </div>
                <button 
                  onClick={() => setShowHelp(false)}
                  className="bg-white/5 hover:bg-white/10 p-2 rounded-full transition-colors"
                >
                  <X size={20} className="text-gray-400" />
                </button>
              </div>
            </div>

            <div className="flex flex-col gap-8 max-h-[60vh] overflow-y-auto pr-2 custom-scrollbar text-left">
              {/* Sekcija 1: Brzi Vodič / Quick Guide */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white/5 p-4 rounded-2xl border border-white/5">
                  <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                    <Activity size={16} className="text-blue-400" />
                    {lang === 'sr' ? '1. TRENIRANJE' : '1. TRAINING'}
                  </h4>
                  <p className="text-xs text-gray-400">
                    {lang === 'sr' ? 
                      'Kliknite na "Start" u Monitoru. Mreža uči kroz epohe dok Loss (greška) ne padne ispod 0.1.' : 
                      'Click "Start" in the Monitor. The network learns through epochs until Loss (error) drops below 0.1.'}
                  </p>
                </div>
                <div className="bg-white/5 p-4 rounded-2xl border border-white/5">
                  <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                    <Target size={16} className="text-green-400" />
                    {lang === 'sr' ? '2. TESTIRANJE' : '2. TESTING'}
                  </h4>
                  <p className="text-xs text-gray-400">
                    {lang === 'sr' ? 
                      'Izaberite cifru u "MNIST Input" panelu i posmatrajte kako se aktivacije prenose kroz mrežu.' : 
                      'Select a digit in the "MNIST Input" panel and watch activations flow through the network.'}
                  </p>
                </div>
              </div>

              {/* Sekcija 2: Detalji Mreže / Network Details */}
              <div className="space-y-6">
                <div className="border-l-2 border-blue-500 pl-4 py-1">
                  <h3 className="text-white font-bold text-lg mb-4 flex items-center gap-2">
                    <Layers size={20} className="text-blue-400" />
                    {lang === 'sr' ? 'Struktura i Uloga Slojeva' : 'Structure and Layer Roles'}
                  </h3>
                  <div className="grid grid-cols-1 gap-3">
                    <div className="bg-white/5 p-3 rounded-xl">
                      <p className="text-xs"><strong className="text-blue-400">Conv (Convolution):</strong> {lang === 'sr' ? 'Ovaj sloj koristi filtere za skeniranje slike i detekciju ivica, uglova i specifičnih oblika cifara.' : 'This layer uses filters to scan the image and detect edges, corners, and specific digit patterns.'}</p>
                    </div>
                    <div className="bg-white/5 p-3 rounded-xl">
                      <p className="text-xs"><strong className="text-blue-400">Pool (MaxPooling):</strong> {lang === 'sr' ? 'Smanjuje rezoluciju mapa aktivacija, zadržavajući samo najjače signale (najbitnije karakteristike).' : 'Reduces the resolution of activation maps, keeping only the strongest signals (most important features).'}</p>
                    </div>
                    <div className="bg-white/5 p-3 rounded-xl">
                      <p className="text-xs"><strong className="text-blue-400">FC (Fully Connected):</strong> {lang === 'sr' ? 'Gledajući sve detektovane oblike, ovaj sloj vrši finalnu klasifikaciju i donosi logički zaključak.' : 'By looking at all detected patterns, this layer performs the final classification and makes a logical decision.'}</p>
                    </div>
                  </div>
                </div>

                <div className="border-l-2 border-green-500 pl-4 py-1">
                  <h3 className="text-white font-bold text-lg mb-4 flex items-center gap-2">
                    <Zap size={20} className="text-green-400" />
                    {lang === 'sr' ? 'Kako Mreža Uči?' : 'How Does the Network Learn?'}
                  </h3>
                  <p className="text-xs text-gray-400 leading-relaxed mb-4">
                    {lang === 'sr' ? 
                      'Tokom treninga, mreža dobija slike sa tačnim odgovorima. Ako pogreši, koriguju se "težine" (weights) između neurona koristeći matematički proces zvan backpropagation (povratno širenje greške). Što je Loss manji, mreža je sigurnija u svoje predviđanje.' : 
                      'During training, the network is given images along with correct answers. If it makes a mistake, it adjusts the "weights" between neurons using a math process called backpropagation. The lower the Loss, the more confident the network is.'}
                  </p>
                  <div className="bg-green-500/10 p-4 rounded-2xl border border-green-500/20 flex items-start gap-3">
                    <Cpu size={24} className="text-green-400 shrink-0" />
                    <div>
                      <h5 className="text-green-400 font-bold text-xs uppercase mb-1">{lang === 'sr' ? 'Donošenje Odluke' : 'Decision Making'}</h5>
                      <p className="text-[11px] text-gray-300">
                        {lang === 'sr' ? 
                          'Izlazni sloj ima 10 neurona (za cifre 0-9). Neuron koji najjače zasvetli (ima najveću vrednost) predstavlja odluku mreže.' : 
                          'The output layer has 10 neurons (for digits 0-9). The neuron that glows the brightest (has the highest value) represents the network prediction.'}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <button 
              onClick={() => setShowHelp(false)}
              className="w-full bg-blue-600 hover:bg-blue-500 text-white py-2.5 sm:py-3 rounded-xl sm:rounded-2xl text-xs sm:text-base font-bold transition-all shadow-lg shadow-blue-600/20 active:scale-95 mt-4"
            >
              {lang === 'sr' ? 'Razumem, pokreni!' : 'Got it, start!'}
            </button>
          </div>
        </div>
      )}

      {/* Dataset Selection Panel */}
      {showInputPanel && (
        <div className="absolute top-4 right-4 z-10 flex flex-col gap-2 bg-black/60 backdrop-blur-xl p-4 rounded-xl border border-white/20 w-64 shadow-2xl max-h-[80vh] overflow-y-auto pointer-events-auto animate-in fade-in slide-in-from-right-4 duration-300">
          <div className="flex justify-between items-center mb-2 border-b border-white/10 pb-2">
              <h3 className="text-white font-bold text-sm tracking-wide flex items-center gap-2">
                <Settings size={16} className="text-blue-400" />
                {lang === 'sr' ? 'MNIST ULAZ' : 'MNIST INPUT'}
              </h3>
          </div>
          
          <div className="opacity-100">
            <div className="grid grid-cols-3 gap-2 mb-4">
            {SAMPLE_IMAGES.map((img) => (
                <button
                key={img.id}
                onClick={() => setSelectedImage(img.url)}
                className={`relative aspect-square rounded-lg overflow-hidden border-2 transition-all group ${
                    selectedImage === img.url ? 'border-blue-500' : 'border-transparent hover:border-white/30'
                }`}
                title={`${img.label} (${img.type})`}
                >
                <img src={img.url} alt={img.label} className="w-full h-full object-cover" referrerPolicy="no-referrer" />
                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                    <span className="text-[10px] text-white font-medium px-1 text-center">{img.label}</span>
                </div>
                </button>
            ))}
            </div>

            <div className="flex flex-col gap-2">
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept="image/*"
                className="hidden"
            />
            <button
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center justify-center gap-2 bg-blue-500/20 hover:bg-blue-500/40 text-blue-100 py-2 rounded-lg text-sm transition-colors border border-blue-500/30"
            >
                <Upload size={16} />
                Upload Image
            </button>
            </div>
          </div>
        </div>
      )}

      {/* Layer Configuration Panel */}
      {selectedLayerConfig && (selectedLayerConfig.type === 'conv' || selectedLayerConfig.type === 'pool') && (
        <div className="absolute top-4 right-72 z-10 flex flex-col gap-4 bg-black/80 backdrop-blur-md p-4 rounded-xl border border-white/10 w-64">
          <div className="flex justify-between items-center border-b border-white/10 pb-2">
            <h3 className="text-white font-semibold text-sm">Layer Config: {selectedLayerConfig.label.split(' ')[0]}</h3>
            <button onClick={() => setSelectedLayerId(null)} className="text-gray-400 hover:text-white">
              <X size={16} />
            </button>
          </div>
          
          <div className="flex flex-col gap-3">
            {selectedLayerConfig.kernelSize !== undefined && (
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400 flex justify-between">
                  Kernel Size <span>{selectedLayerConfig.kernelSize}x{selectedLayerConfig.kernelSize}</span>
                </label>
                <input 
                  type="range" 
                  min="1" 
                  max="7" 
                  step="1"
                  value={selectedLayerConfig.kernelSize}
                  onChange={(e) => updateLayerConfig(selectedLayerConfig.id, 'kernelSize', parseInt(e.target.value))}
                  className="w-full accent-blue-500 h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            )}

            {selectedLayerConfig.stride !== undefined && (
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400 flex justify-between">
                  Stride <span>{selectedLayerConfig.stride}</span>
                </label>
                <input 
                  type="range" 
                  min="1" 
                  max="4" 
                  step="1"
                  value={selectedLayerConfig.stride}
                  onChange={(e) => updateLayerConfig(selectedLayerConfig.id, 'stride', parseInt(e.target.value))}
                  className="w-full accent-blue-500 h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            )}

            {selectedLayerConfig.filters !== undefined && (
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400 flex justify-between">
                  Filters <span>{selectedLayerConfig.filters}</span>
                </label>
                <input 
                  type="range" 
                  min="1" 
                  max="16" 
                  step="1"
                  value={selectedLayerConfig.filters}
                  onChange={(e) => updateLayerConfig(selectedLayerConfig.id, 'filters', parseInt(e.target.value))}
                  className="w-full accent-blue-500 h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            )}

          </div>
        </div>
      )}

      {/* Training Panel */}
      {showTrainingPanel && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 flex flex-col gap-1.5 bg-black/70 backdrop-blur-xl p-3 rounded-xl border border-white/10 w-80 shadow-2xl pointer-events-auto animate-in fade-in slide-in-from-top-4 duration-300">
          <div className="flex justify-between items-center mb-1.5 border-b border-white/10 pb-1.5">
              <h3 className="text-white font-bold text-[11px] flex items-center gap-2 tracking-wider">
                  <Activity size={14} className="text-blue-400" />
                  {lang === 'sr' ? 'TRENING MONITOR' : 'TRAINING MONITOR'}
              </h3>
              <div className="flex gap-1.5">
                  <button 
                      onClick={toggleTraining}
                      className={`px-2.5 py-0.5 rounded-md text-[11px] font-bold transition-all shadow-lg ${
                          isTraining ? 'bg-red-500 hover:bg-red-600 shadow-red-500/20 text-white' : 'bg-green-500 hover:bg-green-600 shadow-green-500/20 text-white'
                      }`}
                  >
                      {isTraining ? 'Stop' : 'Start'}
                  </button>
                  <button 
                      onClick={handlePauseResume}
                      className="px-2.5 py-0.5 rounded-md text-[11px] font-bold bg-yellow-500 hover:bg-yellow-600 shadow-yellow-500/20 text-white transition-all"
                  >
                      {isTraining ? 'Pause' : 'Resume'}
                  </button>
              </div>
          </div>

          <div className="opacity-100 flex flex-col gap-1.5">
            <div className="grid grid-cols-2 gap-1.5">
                <div className="bg-white/5 p-1.5 rounded-lg flex flex-col items-center">
                    <div className="text-[10px] text-gray-500">Epoch</div>
                    <div className="text-sm font-mono font-bold text-white">{epoch}</div>
                </div>
                <div className="bg-white/5 p-1.5 rounded-lg flex flex-col items-center">
                    <div className="text-[10px] text-gray-500">Step</div>
                    <div className="text-sm font-mono font-bold text-white">{trainingStep}</div>
                </div>
            </div>

            <div className="h-20 w-full" style={{ minHeight: '80px' }}>
                <ResponsiveContainer width="100%" height={80}>
                    <LineChart data={trainingHistory}>
                        <CartesianGrid strokeDasharray="2 2" stroke="#222" />
                        <XAxis dataKey="step" hide />
                        <YAxis domain={[0, 3]} width={0} />
                        <Tooltip 
                            contentStyle={{ backgroundColor: '#000', border: '1px solid #333', fontSize: '10px' }}
                            itemStyle={{ padding: 0 }}
                        />
                        <Line type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="accuracy" stroke="#22c55e" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
            <div className="flex justify-between text-[10px] px-1 font-mono">
                <span className="text-red-400">Loss: {trainingHistory[trainingHistory.length - 1]?.loss.toFixed(3) || '0.000'}</span>
                <span className="text-green-400">Acc: {(trainingHistory[trainingHistory.length - 1]?.accuracy * 100).toFixed(1) || '0.0'}%</span>
            </div>
            <div className="flex gap-1">
                <button onClick={handleSaveCheckpoint} className="flex-1 px-1 py-1 rounded text-[9px] bg-blue-600/30 hover:bg-blue-600/50 text-blue-100 border border-blue-500/20 transition-colors">Save</button>
                <button onClick={handleLoadCheckpoint} className="flex-1 px-1 py-1 rounded text-[9px] bg-purple-600/30 hover:bg-purple-600/50 text-purple-100 border border-purple-500/20 transition-colors">Load</button>
                <button onClick={handleExportHistory} className="flex-1 px-1 py-1 rounded text-[9px] bg-gray-600/30 hover:bg-gray-600/50 text-gray-100 border border-gray-500/20 transition-colors">Export</button>
            </div>
          </div>
        </div>
      )}

      {/* Data Collection Panel */}
      {showDataPanel && (
        <div className="absolute top-4 left-4 z-10 flex flex-col gap-2 bg-black/60 backdrop-blur-xl p-4 rounded-xl border border-white/20 w-80 shadow-2xl pointer-events-auto animate-in fade-in slide-in-from-left-4 duration-300">
          <div className="flex justify-between items-center mb-2 border-b border-white/10 pb-2">
            <h3 className="text-white font-bold text-sm flex items-center gap-2 tracking-wide">
              <Upload size={16} className="text-green-400" />
              {lang === 'sr' ? 'KOLEKCIJA PODATAKA' : 'DATA COLLECTION'}
            </h3>
          </div>

          <div className="opacity-100">
          <div className="mb-4">
            <label className="text-xs text-gray-400 block mb-2">Select Label (0-9)</label>
            <div className="grid grid-cols-5 gap-1">
              {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((digit) => (
                <button
                  key={digit}
                  onClick={() => setCurrentLabel(digit)}
                  className={`p-2 rounded text-sm font-bold transition-colors ${
                    currentLabel === digit ? 'bg-blue-500 text-white' : 'bg-white/10 hover:bg-white/20 text-white'
                  }`}
                >
                  {digit}
                </button>
              ))}
            </div>
          </div>

          <div className="mb-4">
            <div className="text-xs text-gray-400 mb-2">Collected Data: {collectedData.images.length} samples</div>
            <div className="flex gap-2">
              <button
                onClick={() => {
                  if (selectedImage) {
                    handleAddToTrainingData(selectedImage, currentLabel);
                  }
                }}
                className="flex-1 bg-green-500 hover:bg-green-600 text-white py-2 rounded text-sm transition-colors"
              >
                Add Current Image
              </button>
              <button
                onClick={() => setCollectedData({ images: [], labels: [] })}
                className="bg-red-500 hover:bg-red-600 text-white py-2 px-3 rounded text-sm transition-colors"
              >
                Clear
              </button>
            </div>
          </div>

          <div className="text-xs text-gray-500 italic mt-auto">
            <p>Upload images to label and add to training set. Images are auto-resized to 28x28.</p>
          </div>
        </div>
      </div>
      )}

      <Canvas camera={{ position: [0, 5, 12], fov: 50 }}>
        <color attach="background" args={['#050505']} />
          <fog attach="fog" args={['#050505', 10, 30]} />
          
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} intensity={1} />
          <spotLight position={[-10, 10, -10]} angle={0.3} penumbra={1} intensity={1} castShadow />
          
          {showStars && <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />}
          <Environment preset="city" />

          {showGrid && <gridHelper args={[30, 30, 0x444444, 0x222222]} position={[0, -3, 0]} />}

          <group position={[0, 0, 0]}>
            {layers.map((layer, index) => (
              <group key={layer.id}>
                <Layer 
                  position={layer.pos as [number, number, number]} 
                  type={layer.type} 
                  size={layer.size as [number, number]} 
                  depth={layer.depth} 
                  label={layer.label}
                  active={activeLayer === index}
                  textureUrl={layer.type === 'input' ? selectedImage : undefined}
                  featureMaps={
                    layer.type === 'conv' ? processedData?.convMaps : 
                    layer.type === 'pool' ? processedData?.poolMaps : undefined
                  }
                  activations={
                    layer.type === 'output' ? processedData?.outputProbs : 
                    layer.type === 'fc' ? processedData?.fcActivations : undefined
                  }
                  outputLabels={
                    layer.type === 'output' ? ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] : undefined
                  }
                  onClick={() => setSelectedLayerId(layer.id)}
                  showMaps={showMatrices}
                  visualContrast={visualContrast}
                />
                {showConnections && index < layers.length - 1 && (
                  <>
                    {/* Use DenseConnections for all detailed layer connections */}
                    {(layer.type === 'pool' && layers[index + 1].type === 'fc') || 
                     (layer.type === 'fc' && layers[index + 1].type === 'output') ||
                     (layer.type === 'input' && layers[index + 1].type === 'conv') ||
                     (layer.type === 'conv' && layers[index + 1].type === 'pool') ||
                     (layer.type === 'input' && layers[index + 1].type === 'pool') ? (
                      <DenseConnections
                        layer1={layer}
                        layer2={layers[index + 1]}
                        active={activeLayer === index}
                        trainingStep={trainingStep}
                        weight={0.8 * connectionThickness}
                        activations1={
                          layer.type === 'fc' ? processedData?.fcActivations : undefined
                        }
                        activations2={
                          layers[index + 1].type === 'fc' ? processedData?.fcActivations :
                          layers[index + 1].type === 'output' ? processedData?.outputProbs : undefined
                        }
                        visualContrast={visualContrast}
                      />
                    ) : (
                      <Connection 
                        start={layer.pos as [number, number, number]} 
                        end={layers[index + 1].pos as [number, number, number]} 
                        active={activeLayer === index}
                        weight={0.6 * connectionThickness}
                        visualContrast={visualContrast}
                      />
                    )}
                  </>
                )}
              </group>
            ))}
          </group>

          <OrbitControls makeDefault />
      </Canvas>
      
      <div className={`absolute bottom-6 left-1/2 -translate-x-1/2 flex flex-col items-center gap-3 w-full max-w-2xl px-4 transition-all duration-500 pointer-events-none z-[60]`}>
        <div className={`p-3 rounded-2xl border transition-all duration-500 text-center w-full pointer-events-auto shadow-2xl ${
          showARMode 
            ? 'bg-blue-950/20 backdrop-blur-md border-blue-500/30 shadow-blue-500/10' 
            : 'bg-black/60 backdrop-blur-lg border-white/10'
        }`}>
          <h1 className="text-sm font-bold mb-2 uppercase tracking-widest text-blue-400">MNIST CNN Simulation</h1>
          
          <div className="flex flex-wrap justify-center items-center gap-2 mb-3">
            <button 
              onClick={handlePrev}
              className="p-1.5 rounded-full bg-white/5 hover:bg-white/20 transition-colors"
            >
              <SkipBack size={18} />
            </button>
            <button 
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-2.5 rounded-full bg-blue-500 hover:bg-blue-600 transition-colors shadow-lg shadow-blue-500/20"
            >
              {isPlaying ? <Pause size={20} /> : <Play size={20} />}
            </button>
            <button 
              onClick={handleNext}
              className="p-1.5 rounded-full bg-white/5 hover:bg-white/20 transition-colors"
            >
              <SkipForward size={18} />
            </button>
            <div className="w-px h-5 bg-white/10"></div>
            <button 
              onClick={() => setShowStars(!showStars)}
              className="p-1.5 rounded-full bg-white/5 hover:bg-white/20 transition-colors"
              title={showStars ? "Hide stars" : "Show stars"}
            >
              {showStars ? <Eye size={18} className="text-blue-400" /> : <EyeOff size={18} className="text-gray-400" />}
            </button>
            <button 
              onClick={() => setShowGrid(!showGrid)}
              className="p-1.5 rounded-full bg-white/5 hover:bg-white/20 transition-colors"
              title={showGrid ? "Hide grid" : "Show grid"}
            >
              <Grid3x3 size={18} className={showGrid ? "text-blue-400" : "text-gray-400"} />
            </button>
            <button 
              onClick={() => setShowConnections(!showConnections)}
              className="p-1.5 rounded-full bg-white/5 hover:bg-white/20 transition-colors"
              title={showConnections ? "Hide connections" : "Show connections"}
            >
              <Network size={18} className={showConnections ? "text-blue-400" : "text-gray-400"} />
            </button>

            {/* Connection Thickness Slider */}
            {showConnections && (
              <div className="flex items-center px-2 bg-white/5 rounded-full border border-white/5 h-8">
                <input 
                  type="range" 
                  min="0.2" 
                  max="5.0" 
                  step="0.1"
                  value={connectionThickness}
                  onChange={(e) => setConnectionThickness(parseFloat(e.target.value))}
                  className="w-16 h-1 accent-blue-500 bg-white/10 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            )}
            <button 
              onClick={() => setShowMatrices(!showMatrices)}
              className="p-1.5 rounded-full bg-white/5 hover:bg-white/20 transition-colors shadow-lg"
              title={lang === 'sr' ? 'Prikaži/Sakrij matrice' : 'Show/Hide Matrices'}
            >
              <Grid3x3 size={18} className={showMatrices ? "text-blue-400" : "text-gray-400"} />
            </button>
            <button 
              onClick={() => setShowDataPanel(!showDataPanel)}
              className="p-1.5 rounded-full bg-white/5 hover:bg-white/20 transition-colors"
              title={showDataPanel ? "Hide Data Collection" : "Show Data Collection"}
            >
              <Upload size={18} className={showDataPanel ? "text-green-400" : "text-gray-400"} />
            </button>
            <button 
              onClick={() => setShowTrainingPanel(!showTrainingPanel)}
              className="p-1.5 rounded-full bg-white/5 hover:bg-white/20 transition-colors"
              title={showTrainingPanel ? "Hide Training Monitor" : "Show Training Monitor"}
            >
              <Activity size={18} className={showTrainingPanel ? "text-blue-400" : "text-gray-400"} />
            </button>
            <button 
              onClick={() => setShowInputPanel(!showInputPanel)}
              className="p-1.5 rounded-full bg-white/5 hover:bg-white/20 transition-colors"
              title={showInputPanel ? "Hide Input Panel" : "Show Input Panel"}
            >
              <Settings size={18} className={showInputPanel ? "text-blue-400" : "text-gray-400"} />
            </button>
            <div className="w-[1px] h-4 bg-white/10 mx-1" />
            <button 
              onClick={() => setShowARMode(!showARMode)}
              className="p-1.5 rounded-full bg-white/5 hover:bg-white/20 transition-colors flex items-center gap-1"
              title="AR/VR Mode"
            >
              <Box size={18} className={showARMode ? "text-blue-400" : "text-gray-400"} />
              <span className="text-[10px] text-gray-400 font-bold pr-1">AR/VR</span>
            </button>
            <button 
              onClick={() => setShowAIMentor(!showAIMentor)}
              className={`p-1.5 rounded-full transition-all flex items-center gap-1.5 ${
                showAIMentor 
                  ? 'bg-blue-600 shadow-lg shadow-blue-600/30' 
                  : 'bg-blue-500/20 hover:bg-blue-500/40'
              }`}
              title="AI Mentor (Gemini)"
            >
              <Sparkles size={18} className={showAIMentor ? "text-white" : "text-blue-400"} />
              <span className={`text-[10px] font-bold pr-1 ${showAIMentor ? 'text-white' : 'text-blue-400'}`}>AI MENTOR</span>
            </button>
            <button 
              onClick={() => setShowHelp(!showHelp)}
              className="p-1.5 rounded-full bg-blue-500/20 hover:bg-blue-500/40 transition-colors"
              title="Help & Guide"
            >
              <HelpCircle size={18} className="text-blue-400" />
            </button>
            <button 
              onClick={() => setVisualContrast(prev => prev === 1.0 ? 3.0 : 1.0)}
              className={`p-1.5 rounded-full transition-all flex items-center ${
                visualContrast > 1.0 
                  ? 'bg-orange-500 shadow-lg shadow-orange-500/30' 
                  : 'bg-white/5 hover:bg-white/20'
              }`}
              title={lang === 'sr' ? 'Kontrast Vizuelizacije' : 'Visual Contrast'}
            >
              <Contrast size={18} className={visualContrast > 1.0 ? "text-white" : "text-gray-400"} />
            </button>
          </div>

          <div className="flex justify-center gap-2 text-[10px] sm:text-xs overflow-x-auto pb-1 w-full">
            {layers.map((layer, index) => (
              <div 
                key={layer.id}
                onClick={() => {
                  setActiveLayer(index);
                  setIsPlaying(false);
                }}
                className={`cursor-pointer px-2.5 py-1.5 rounded transition-all whitespace-nowrap border ${
                  activeLayer === index 
                    ? 'bg-blue-500 border-blue-400 text-white scale-105 shadow-md shadow-blue-500/20' 
                    : 'bg-white/5 border-transparent hover:bg-white/10 text-gray-400'
                }`}
              >
                {layer.label.split(' ')[0]}
              </div>
            ))}
          </div>
          
          <div className="absolute bottom-2 right-4 text-[10px] text-gray-600 pointer-events-none opacity-50">
            v1.0.0
          </div>
        </div>
      </div>
    </>
  );
}
