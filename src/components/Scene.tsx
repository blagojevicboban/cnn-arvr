import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars, Environment } from '@react-three/drei';
import { XR, createXRStore } from '@react-three/xr';
import { Layer } from './Layer';
import { Connection } from './Connection';
import { DenseConnections } from './DenseConnections';
import { useState, useEffect, useRef } from 'react';
import { Play, Pause, SkipForward, SkipBack, Upload, Settings, X, Activity, ChevronDown, ChevronUp } from 'lucide-react';
import * as THREE from 'three';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { processImage } from '../utils/imageProcessor';

const store = createXRStore();

// Sample images (using placeholder services for demo)
const SAMPLE_IMAGES = [
  { id: '0', url: 'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/25.png', label: 'Pikachu', type: 'Sprite' },
  { id: '1', url: 'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/1.png', label: 'Bulbasaur', type: 'Sprite' },
  { id: '2', url: 'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/4.png', label: 'Charmander', type: 'Sprite' },
  { id: '3', url: 'https://placehold.co/28x28/000000/FFFFFF/png?text=7', label: 'MNIST Digit', type: 'Grayscale' },
  { id: '4', url: 'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=200', label: 'Cat', type: 'Photo' },
  { id: '5', url: 'https://images.unsplash.com/photo-1523875194681-bedd468c58bf?w=200', label: 'Dice', type: 'Object' },
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
    { id: 3, type: 'fc', pos: [3, 0, 0], size: [10, 1], depth: 1, label: "FC (120)" },
    { id: 4, type: 'output', pos: [6, 0, 0], size: [10, 1], depth: 1, label: "Output (10)" },
  ]);

  const [selectedLayerId, setSelectedLayerId] = useState<number | null>(null);
  
  // Processed Data State
  const [processedData, setProcessedData] = useState<{
    convMaps: string[];
    poolMaps: string[];
    outputProbs: number[];
  } | null>(null);

  // UI State
  const [isInputPanelOpen, setIsInputPanelOpen] = useState(true);
  const [isTrainingPanelOpen, setIsTrainingPanelOpen] = useState(true);

  // Training State
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStep, setTrainingStep] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const [trainingHistory, setTrainingHistory] = useState<{ step: number; loss: number; accuracy: number }[]>([
    { step: 0, loss: 2.5, accuracy: 0.1 }
  ]);

  useEffect(() => {
    if (selectedImage) {
      processImage(selectedImage).then(data => {
        setProcessedData(data);
      });
    }
  }, [selectedImage]);

  // Simulate data flow (inference)
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying && !isTraining) {
      interval = setInterval(() => {
        setActiveLayer((prev) => (prev + 1) % 5);
      }, 1500);
    }
    return () => clearInterval(interval);
  }, [isPlaying, isTraining]);

  // Simulate training process
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isTraining) {
      interval = setInterval(() => {
        setTrainingStep((prev) => prev + 1);
        
        // Simulate loss decreasing and accuracy increasing
        setTrainingHistory((prev) => {
          const step = prev.length + 1;
          const progress = Math.min(step / 100, 1); // 0 to 1 over 100 steps
          
          // Noisy loss curve
          const baseLoss = 2.5 * Math.exp(-3 * progress);
          const loss = Math.max(0, baseLoss + (Math.random() - 0.5) * 0.2);
          
          // Noisy accuracy curve
          const baseAcc = 0.1 + 0.85 * (1 - Math.exp(-2.5 * progress));
          const accuracy = Math.min(1, Math.max(0, baseAcc + (Math.random() - 0.5) * 0.05));

          const newHistory = [...prev, { step, loss, accuracy }];
          // if (newHistory.length > 50) newHistory.shift(); // Keep last 50 points -> REMOVED to show full history
          return newHistory;
        });

        if (trainingStep % 20 === 0) {
            setEpoch(e => e + 1);
        }

      }, 100);
    }
    return () => clearInterval(interval);
  }, [isTraining, trainingStep]);

  const toggleTraining = () => {
    setIsTraining(!isTraining);
    setIsPlaying(false); // Stop inference animation when training starts
    if (!isTraining) {
        // Reset if starting fresh? No, let's continue training
        if (trainingHistory.length === 0) {
             setTrainingHistory([{ step: 0, loss: 2.5, accuracy: 0.1 }]);
        }
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setSelectedImage(url);
    }
  };

  const updateLayerConfig = (id: number, key: keyof LayerConfig, value: number) => {
    setLayers(prev => prev.map(layer => {
      if (layer.id === id) {
        return { ...layer, [key]: value };
      }
      return layer;
    }));
  };

  const handleNext = () => setActiveLayer((prev) => (prev + 1) % layers.length);
  const handlePrev = () => setActiveLayer((prev) => (prev - 1 + layers.length) % layers.length);

  const selectedLayerConfig = layers.find(l => l.id === selectedLayerId);

  return (
    <>
      <div className="absolute top-4 left-4 z-10 flex gap-2">
        <button 
          onClick={() => store.enterAR()}
          className="bg-white/10 backdrop-blur-md border border-white/20 text-white px-4 py-2 rounded-lg hover:bg-white/20 transition-colors"
        >
          Enter AR
        </button>
        <button 
          onClick={() => store.enterVR()}
          className="bg-white/10 backdrop-blur-md border border-white/20 text-white px-4 py-2 rounded-lg hover:bg-white/20 transition-colors"
        >
          Enter VR
        </button>
      </div>

      {/* Dataset Selection Panel */}
      <div className={`absolute top-4 right-4 z-10 flex flex-col gap-2 bg-black/50 backdrop-blur-md p-4 rounded-xl border border-white/10 w-64 transition-all duration-300 ${isInputPanelOpen ? 'max-h-[80vh]' : 'max-h-[60px] overflow-hidden'}`}>
        <div className="flex justify-between items-center mb-2">
            <h3 className="text-white font-semibold text-sm">Input Data</h3>
            <button onClick={() => setIsInputPanelOpen(!isInputPanelOpen)} className="text-gray-400 hover:text-white">
                {isInputPanelOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>
        </div>
        
        <div className={`transition-opacity duration-300 ${isInputPanelOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
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
                className="flex items-center justify-center gap-2 bg-white/10 hover:bg-white/20 text-white py-2 rounded-lg text-sm transition-colors border border-white/10"
            >
                <Upload size={16} />
                Upload Image
            </button>
            </div>
        </div>
      </div>

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
      <div className={`absolute top-4 left-48 z-10 flex flex-col gap-2 bg-black/50 backdrop-blur-md p-4 rounded-xl border border-white/10 w-80 transition-all duration-300 ${isTrainingPanelOpen ? 'max-h-[400px]' : 'max-h-[60px] overflow-hidden'}`}>
        <div className="flex justify-between items-center mb-2">
            <h3 className="text-white font-semibold text-sm flex items-center gap-2">
                <Activity size={16} className="text-blue-400" />
                Training Monitor
            </h3>
            <div className="flex gap-2">
                <button 
                    onClick={toggleTraining}
                    className={`px-3 py-1 rounded-md text-xs font-bold transition-colors ${
                        isTraining ? 'bg-red-500 hover:bg-red-600 text-white' : 'bg-green-500 hover:bg-green-600 text-white'
                    }`}
                >
                    {isTraining ? 'Stop' : 'Start'}
                </button>
                <button onClick={() => setIsTrainingPanelOpen(!isTrainingPanelOpen)} className="text-gray-400 hover:text-white">
                    {isTrainingPanelOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </button>
            </div>
        </div>

        <div className={`transition-opacity duration-300 ${isTrainingPanelOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
            <div className="grid grid-cols-2 gap-2 mb-2">
                <div className="bg-white/5 p-2 rounded-lg">
                    <div className="text-xs text-gray-400">Epoch</div>
                    <div className="text-lg font-mono text-white">{epoch}</div>
                </div>
                <div className="bg-white/5 p-2 rounded-lg">
                    <div className="text-xs text-gray-400">Step</div>
                    <div className="text-lg font-mono text-white">{trainingStep}</div>
                </div>
            </div>

            <div className="h-32 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trainingHistory}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                        <XAxis dataKey="step" hide />
                        <YAxis domain={[0, 3]} width={0} />
                        <Tooltip 
                            contentStyle={{ backgroundColor: '#000', border: '1px solid #333', fontSize: '12px' }}
                            itemStyle={{ padding: 0 }}
                        />
                        <Line type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="accuracy" stroke="#22c55e" strokeWidth={2} dot={false} isAnimationActive={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
            <div className="flex justify-between text-xs px-1">
                <span className="text-red-400">Loss: {trainingHistory[trainingHistory.length - 1]?.loss.toFixed(3) || '0.000'}</span>
                <span className="text-green-400">Acc: {(trainingHistory[trainingHistory.length - 1]?.accuracy * 100).toFixed(1) || '0.0'}%</span>
            </div>
        </div>
      </div>

      <Canvas camera={{ position: [0, 5, 12], fov: 50 }}>
        <XR store={store}>
          <color attach="background" args={['#050505']} />
          <fog attach="fog" args={['#050505', 10, 30]} />
          
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} intensity={1} />
          <spotLight position={[-10, 10, -10]} angle={0.3} penumbra={1} intensity={1} castShadow />
          
          <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
          <Environment preset="city" />

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
                    layer.type === 'output' ? processedData?.outputProbs : undefined
                  }
                  outputLabels={
                    layer.type === 'output' ? ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] : undefined
                  }
                  onClick={() => setSelectedLayerId(layer.id)}
                />
                {index < layers.length - 1 && (
                  <>
                    {/* Use DenseConnections for Pool->FC and FC->Output for detailed visualization */}
                    {(layer.type === 'pool' && layers[index + 1].type === 'fc') || 
                     (layer.type === 'fc' && layers[index + 1].type === 'output') ? (
                      <DenseConnections
                        layer1={layer}
                        layer2={layers[index + 1]}
                        active={activeLayer === index || isTraining}
                        trainingStep={trainingStep}
                      />
                    ) : (
                      <Connection 
                        start={layer.pos as [number, number, number]} 
                        end={layers[index + 1].pos as [number, number, number]} 
                        active={activeLayer === index || isTraining}
                      />
                    )}
                  </>
                )}
              </group>
            ))}
          </group>

          <OrbitControls makeDefault />
          <gridHelper args={[30, 30, 0x444444, 0x222222]} position={[0, -3, 0]} />
        </XR>
      </Canvas>
      
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-4 w-full max-w-2xl px-4 pointer-events-none">
        <div className="bg-black/50 backdrop-blur-md p-4 rounded-xl border border-white/10 text-center w-full pointer-events-auto">
          <h1 className="text-xl font-bold mb-4">CNN Architecture Visualization</h1>
          
          <div className="flex justify-center items-center gap-4 mb-4">
            <button 
              onClick={handlePrev}
              className="p-2 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
            >
              <SkipBack size={20} />
            </button>
            <button 
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-3 rounded-full bg-blue-500 hover:bg-blue-600 transition-colors"
            >
              {isPlaying ? <Pause size={24} /> : <Play size={24} />}
            </button>
            <button 
              onClick={handleNext}
              className="p-2 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
            >
              <SkipForward size={20} />
            </button>
          </div>

          <div className="flex justify-between gap-2 text-xs sm:text-sm overflow-x-auto pb-2 w-full">
            {layers.map((layer, index) => (
              <div 
                key={layer.id}
                onClick={() => {
                  setActiveLayer(index);
                  setIsPlaying(false);
                }}
                className={`cursor-pointer px-3 py-2 rounded transition-all whitespace-nowrap ${
                  activeLayer === index 
                    ? 'bg-blue-500 text-white scale-105 shadow-lg shadow-blue-500/20' 
                    : 'bg-white/5 hover:bg-white/10 text-gray-400'
                }`}
              >
                {layer.label.split(' ')[0]}
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}
