import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface LayerData {
  pos: [number, number, number];
  size: [number, number]; // [neurons, 1] for FC, [rows, cols] for Conv/Pool
  depth: number;
  type: 'input' | 'conv' | 'pool' | 'fc' | 'output';
}

interface DenseConnectionsProps {
  layer1: LayerData;
  layer2: LayerData;
  active: boolean;
  trainingStep?: number;
  weight?: number; 
  activations1?: number[];
  activations2?: number[];
}

export function DenseConnections({ layer1, layer2, active, trainingStep = 0, weight = 1.0, activations1, activations2 }: DenseConnectionsProps) {
  const linesRef = useRef<THREE.LineSegments>(null);
  
  const { positions, colors } = useMemo(() => {
    const points: number[] = [];
    const colorArray: number[] = [];
    
    // Helper to get world position of a neuron in a layer
    const getNeuronPos = (layer: LayerData, index: number, total: number) => {
      const layerPos = new THREE.Vector3(...layer.pos);
      
      if (layer.type === 'fc' || layer.type === 'output') {
        const gridWidth = layer.type === 'fc' ? 10 : 1;
        const gridHeight = Math.ceil(total / gridWidth);
        const spacing = layer.type === 'output' ? 0.6 : 0.35;
        
        const startZ = -(gridWidth - 1) * spacing / 2;
        const startY = (gridHeight - 1) * spacing / 2;
        
        const row = Math.floor(index / gridWidth);
        const col = index % gridWidth;
        const y = startY - row * spacing;
        const z = startZ + col * spacing;
        
        return layerPos.clone().add(new THREE.Vector3(0, y, z));
      } else if (layer.type === 'conv' || layer.type === 'pool') {
        // Handle neuron within map if total is large, otherwise map center
        const neuronGridSize = layer.type === 'conv' ? 6 : 4;
        const neuronsPerMap = neuronGridSize * neuronGridSize;
        const mapSize = 0.8;
        const gap = 0.1;
        const cols = Math.ceil(Math.sqrt(layer.depth));
        const rows = Math.ceil(layer.depth / cols);
        
        const mapIdx = Math.floor(index / neuronsPerMap);
        const neuronIdx = index % neuronsPerMap;
        
        const mRow = Math.floor(mapIdx / cols);
        const mCol = mapIdx % cols;
        
        const mY = -(mRow - (rows - 1) / 2) * (mapSize + gap);
        const mZ = (mCol - (cols - 1) / 2) * (mapSize + gap);
        
        // Position within the map matrix (r is vertical row, c is horizontal col)
        const r = Math.floor(neuronIdx / neuronGridSize);
        const c = neuronIdx % neuronGridSize;
        const spacing = mapSize / (neuronGridSize + 1);
        
        const y = mY - mapSize/2 + (neuronGridSize - 1 - r + 1) * spacing;
        const z = mZ - mapSize/2 + (c + 1) * spacing;
        
        return layerPos.clone().add(new THREE.Vector3(0, y, z));
      } else {
        // Fallback for input layer: 14x14 grid
        const gridSide = 14;
        const spacing = 0.15;
        const start = -(gridSide - 1) * spacing / 2;
        const r = Math.floor(index / gridSide);
        const c = index % gridSide;
        // Y = Vertical (Row), Z = Horizontal (Column)
        const y = start + (13 - r) * spacing;
        const z = start + c * spacing;
        return layerPos.clone().add(new THREE.Vector3(0, y, z));
      }
    };

    // Generate connections
    // Case 1: FC -> Output (Fully connected)
    if ((layer1.type === 'fc' || layer1.type === 'output') && 
        (layer2.type === 'fc' || layer2.type === 'output')) {
      
      const count1 = layer1.size[0];
      const count2 = layer2.size[0];
      
      for (let i = 0; i < count1; i++) {
        for (let j = 0; j < count2; j++) {
          // Cull connections if too many (120x10 = 1200 is manageable, but sample if more)
          if (count1 > 200 && Math.random() > 0.3) continue;
          
          const start = getNeuronPos(layer1, i, count1);
          const end = getNeuronPos(layer2, j, count2);
          
          points.push(start.x, start.y, start.z);
          points.push(end.x, end.y, end.z);
          
          const act1 = activations1 ? activations1[i] : 0;
          const act2 = activations2 ? activations2[j] : 0;
          const strength = active ? (activations1 && activations2 ? act1 * act2 : 0.5) : 0;

          let color: THREE.Color;
          if (active) {
            const seed = i * count2 + j;
            const hue = 0.35 + ((Math.sin(seed + trainingStep * 0.5) + 1) / 2) * 0.15; // Cyan to Green
            const light = 0.4 + strength * 0.6;
            color = new THREE.Color().setHSL(hue, 0.8 + strength * 0.2, light);
          } else {
            color = new THREE.Color(0x333333); 
          }
          
          colorArray.push(color.r, color.g, color.b);
          colorArray.push(color.r, color.g, color.b);
        }
      }
    } 
    // Case 2: Pool -> FC (Flattening)
    else if (layer1.type === 'pool' && layer2.type === 'fc') {
        const neuronGridSize = 4; // Pool layer neuron grid (visual)
        const neuronsPerMap = neuronGridSize * neuronGridSize;
        const totalPoolNeurons = layer1.depth * neuronsPerMap;
        const totalFCNeurons = layer2.size[0];
        
        // Visual connection: connect every visual pool neuron to a selection of FC neurons
        for (let i = 0; i < totalPoolNeurons; i++) {
            const start = getNeuronPos(layer1, i, totalPoolNeurons);
            
            // Targeted connections to ensure full feel
            for (let j = 0; j < totalFCNeurons; j += 4) { // Connect to every 4th FC neuron
                const end = getNeuronPos(layer2, j, totalFCNeurons);
                const act2 = activations2 ? activations2[j] : 0;
                const strength = active ? (activations2 ? act2 : 0.5) : 0.1;
                
                points.push(start.x, start.y, start.z);
                points.push(end.x, end.y, end.z);
                
                let color: THREE.Color;
                if (active) {
                    const seed = i + j;
                    const hue = 0.55 + ((Math.sin(seed + trainingStep * 0.5) + 1) / 2) * 0.1;
                    const light = 0.3 + strength * 0.6;
                    color = new THREE.Color().setHSL(hue, 0.8 + strength * 0.2, light);
                } else {
                    color = new THREE.Color(0x333333); 
                }
                
                colorArray.push(color.r, color.g, color.b, color.r, color.g, color.b);
            }
        }
    }
    // Case 3: Conv -> Pool or Input -> Conv
    else if (layer1.type === 'input' && layer2.type === 'conv') {
        const inputGridSize = 14;
        const neuronGridSize = 6;
        const neuronsPerMap = neuronGridSize * neuronGridSize;
        
        for (let mapIdx = 0; mapIdx < layer2.depth; mapIdx++) {
            for (let r = 0; r < neuronGridSize; r++) {
                for (let c = 0; c < neuronGridSize; c++) {
                    const targetIdx = mapIdx * neuronsPerMap + (r * neuronGridSize + c);
                    const targetPos = getNeuronPos(layer2, targetIdx, layer2.depth * neuronsPerMap);
                    
                    // Structured mapping: find corresponding input area
                    const srcR = Math.floor((r / (neuronGridSize - 1)) * (inputGridSize - 1));
                    const srcC = Math.floor((c / (neuronGridSize - 1)) * (inputGridSize - 1));
                    const srcIdx = srcR * inputGridSize + srcC;
                    const srcPos = getNeuronPos(layer1, srcIdx, inputGridSize * inputGridSize);
                    
                    points.push(srcPos.x, srcPos.y, srcPos.z);
                    points.push(targetPos.x, targetPos.y, targetPos.z);
                    
                    let color = new THREE.Color(active ? 0x60a5fa : 0x333333);
                    colorArray.push(color.r, color.g, color.b, color.r, color.g, color.b);
                }
            }
        }
    } else if (layer1.type === 'conv' && layer2.type === 'pool') {
        const neuronGrid1 = 6;
        const neuronsPerMap1 = neuronGrid1 * neuronGrid1;
        const neuronGrid2 = 4;
        const neuronsPerMap2 = neuronGrid2 * neuronGrid2;
        
        for (let mapIdx = 0; mapIdx < layer1.depth; mapIdx++) {
            for (let r = 0; r < neuronGrid1; r++) {
                for (let c = 0; c < neuronGrid1; c++) {
                    // Precision 1:1 mapping for Conv -> Pool
                    const srcIdx = mapIdx * neuronsPerMap1 + (r * neuronGrid1 + c);
                    const srcPos = getNeuronPos(layer1, srcIdx, layer1.depth * neuronsPerMap1);
                    
                    const pr = Math.min(neuronGrid2 - 1, Math.floor(r / 1.5));
                    const pc = Math.min(neuronGrid2 - 1, Math.floor(c / 1.5));
                    const targetIdx = mapIdx * neuronsPerMap2 + (pr * neuronGrid2 + pc);
                    const targetPos = getNeuronPos(layer2, targetIdx, layer2.depth * neuronsPerMap2);

                    points.push(srcPos.x, srcPos.y, srcPos.z);
                    points.push(targetPos.x, targetPos.y, targetPos.z);
                    
                    let color = new THREE.Color(active ? 0x4ade80 : 0x333333);
                    colorArray.push(color.r, color.g, color.b, color.r, color.g, color.b);
                }
            }
        }
    }

    return {
      positions: new Float32Array(points),
      colors: new Float32Array(colorArray)
    };
  }, [layer1, layer2, trainingStep, activations1, activations2]);

  useFrame((state) => {
    if (linesRef.current && active) {
        // Animate opacity or color if needed
        // linesRef.current.material.opacity = 0.1 + Math.sin(state.clock.elapsedTime) * 0.05;
    }
  });

  return (
    <lineSegments ref={linesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={positions.length / 3}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={colors.length / 3}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <lineBasicMaterial 
        vertexColors 
        transparent 
        opacity={(active ? 0.6 : 0.4) * weight} 
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        linewidth={10}
      />
    </lineSegments>
  );
}
