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
        
        // FC/Output neurons are at local X=0
        return layerPos.add(new THREE.Vector3(0, y, z));
      } else {
        // For conv/pool feature map PLANE (used for map centers)
        const mapSize = 0.8;
        const gap = 0.1;
        const cols = Math.ceil(Math.sqrt(layer.depth));
        const rows = Math.ceil(layer.depth / cols);
        
        const mapIdx = index % layer.depth;
        const row = Math.floor(mapIdx / cols);
        const col = mapIdx % cols;
        
        const y = (col - (cols - 1) / 2) * (mapSize + gap);
        const z = -(row - (rows - 1) / 2) * (mapSize + gap);
        
        // Feature map meshes are at local X=-0.8, Neurons are at local X=0.1
        // Usually connections target the neuron spheres
        return layerPos.add(new THREE.Vector3(0.1, y, z));
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
        // Sample connections from feature maps to FC neurons
        for (let i = 0; i < layer1.depth; i++) {
            const start = getNeuronPos(layer1, i, layer1.depth);
            
            for (let j = 0; j < layer2.size[0]; j += 5) {
                const end = getNeuronPos(layer2, j, layer2.size[0]);
                const act2 = activations2 ? activations2[j] : 0;
                const strength = active ? (activations2 ? act2 : 0.5) : 0;
                
                points.push(start.x, start.y, start.z);
                points.push(end.x, end.y, end.z);
                
                let color: THREE.Color;
                if (active) {
                  const seed = i * layer2.size[0] + j;
                  const hue = 0.55 + ((Math.sin(seed + trainingStep * 0.5) + 1) / 2) * 0.1; // Blue to cyan
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
    // Case 3: Conv -> Pool or Input -> Conv (Feature map to Feature map)
    else {
        // For Input -> Conv: Sample connections from input neurons to conv feature maps
        // For Conv -> Pool: Sample connections from conv feature maps to pool feature maps
        
        if (layer1.type === 'input' && layer2.type === 'conv') {
            // Input (10x10 neurons) -> Conv1 (6 feature maps)
            // Sample connections from input grid to each conv feature map
            const inputGridSize = 14;
            const sampleRate = 0.3; // Sample 30% of input neurons
            
            for (let mapIdx = 0; mapIdx < layer2.depth; mapIdx++) {
                // Connections from Input target the feature map PLANE at local X=-0.8
                const mapIdxCenter = getNeuronPos(layer2, mapIdx, layer2.depth);
                const mapPos = new THREE.Vector3(layer2.pos[0] - 0.8, mapIdxCenter.y, mapIdxCenter.z);
                
                for (let i = 0; i < inputGridSize; i++) {
                    if (Math.random() > sampleRate) continue;
                    
                    for (let j = 0; j < inputGridSize; j++) {
                        if (Math.random() > sampleRate) continue;
                        
                        // Calculate position of this input neuron in the input layer
                        const inputNeuronIdx = i * inputGridSize + j;
                        const inputNeuronSpacing = 0.15;
                        const totalInputWidth = (inputGridSize - 1) * inputNeuronSpacing;
                        const inputStartY = (inputGridSize - 1) * inputNeuronSpacing / 2;
                        const inputStartZ = -totalInputWidth / 2;
                        
                        const inputY = inputStartY - j * inputNeuronSpacing;
                        const inputZ = inputStartZ + i * inputNeuronSpacing;
                        const inputPos = new THREE.Vector3(...layer1.pos).add(new THREE.Vector3(0, inputY, inputZ));
                        
                        points.push(inputPos.x, inputPos.y, inputPos.z);
                        points.push(mapPos.x, mapPos.y, mapPos.z);
                        
                        // Orange color for input->conv connections
                        let color: THREE.Color;
                        if (active) {
                            const seed = inputNeuronIdx * layer2.depth + mapIdx;
                            const hue = 0.08 + ((Math.sin(seed + trainingStep * 0.5) + 1) / 2) * 0.1; // Orange-ish
                            color = new THREE.Color().setHSL(hue, 0.9, 0.7);
                        } else {
                            color = new THREE.Color(0x666666); // Lighter gray
                        }
                        
                        colorArray.push(color.r, color.g, color.b);
                        colorArray.push(color.r, color.g, color.b);
                    }
                }
            }
        } else if (layer1.type === 'conv' && layer2.type === 'pool') {
            // Conv -> Pool: Connect between downsampled neurons (6x6 -> 4x4)
            const mapCols = Math.ceil(Math.sqrt(layer1.depth));
            const mapSize = 0.8;
            const gap = 0.1;
            
            const neuronGrid1 = 6;
            const neuronGrid2 = 4;
            const spacing1 = mapSize / (neuronGrid1 + 1);
            const spacing2 = mapSize / (neuronGrid2 + 1);
            
            for (let mapIdx = 0; mapIdx < layer1.depth; mapIdx++) {
                const mRow = Math.floor(mapIdx / mapCols);
                const mCol = mapIdx % mapCols;
                const mY = (mCol - (mapCols - 1) / 2) * (mapSize + gap);
                const mZ = -(mRow - (mapCols - 1) / 2) * (mapSize + gap);

                for (let r = 0; r < neuronGrid1; r++) {
                    for (let c = 0; c < neuronGrid1; c++) {
                        if (Math.random() > 0.4) continue; // Sample
                        
                        const y1 = mY - mapSize/2 + (c + 1) * spacing1;
                        const z1 = mZ - mapSize/2 + (r + 1) * spacing1;
                        
                        // Target corresponding pool neuron (simple 1.5 ratio)
                        const pr = Math.min(neuronGrid2 - 1, Math.floor(r / 1.5));
                        const pc = Math.min(neuronGrid2 - 1, Math.floor(c / 1.5));
                        
                        const y2 = mY - mapSize/2 + (pc + 1) * spacing2;
                        const z2 = mZ - mapSize/2 + (pr + 1) * spacing2;

                        points.push(layer1.pos[0] + 0.1, y1, z1);
                        points.push(layer2.pos[0] + 0.1, y2, z2);
                        
                        let color = new THREE.Color(active ? 0x4ade80 : 0x666666);
                        colorArray.push(color.r, color.g, color.b, color.r, color.g, color.b);
                    }
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
