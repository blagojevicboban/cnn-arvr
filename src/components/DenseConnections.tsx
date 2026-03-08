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
  weight?: number; // Connection weight (0-1), affects line thickness
}

export function DenseConnections({ layer1, layer2, active, trainingStep = 0, weight = 0.5 }: DenseConnectionsProps) {
  const linesRef = useRef<THREE.LineSegments>(null);
  
  const { positions, colors } = useMemo(() => {
    const points: number[] = [];
    const colorArray: number[] = [];
    
    // Helper to get world position of a neuron in a layer
    const getNeuronPos = (layer: LayerData, index: number, total: number) => {
      const layerPos = new THREE.Vector3(...layer.pos);
      
      if (layer.type === 'fc' || layer.type === 'output') {
        // Vertical line of neurons
        const height = (layer.size[0] - 1) * 0.5;
        const y = (index - (layer.size[0] - 1) / 2) * 0.5;
        return layerPos.add(new THREE.Vector3(0, y, 0));
      } else {
        // For conv/pool, we'll just sample center points of feature maps for now
        // or maybe a few random points within the volume to represent "neurons"
        // Let's simplify and just use the layer center for non-FC layers to avoid visual chaos
        // OR, let's try to map to the feature maps locations
        
        const mapSize = 0.8;
        const gap = 0.1;
        const cols = Math.ceil(Math.sqrt(layer.depth));
        const rows = Math.ceil(layer.depth / cols);
        
        // If we treat index as flat index across all pixels... that's too many.
        // Let's treat index as feature map index
        const mapIdx = index % layer.depth;
        const row = Math.floor(mapIdx / cols);
        const col = mapIdx % cols;
        
        const x = (col - (cols - 1) / 2) * (mapSize + gap);
        const y = -(row - (rows - 1) / 2) * (mapSize + gap);
        
        return layerPos.add(new THREE.Vector3(x, y, 0));
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
          // Randomly cull connections to reduce visual noise if needed, 
          // but for 120x10 (1200) it's fine.
          
          const start = getNeuronPos(layer1, i, count1);
          const end = getNeuronPos(layer2, j, count2);
          
          points.push(start.x, start.y, start.z);
          points.push(end.x, end.y, end.z);
          
          // Random weight strength for color/opacity
          // Use trainingStep to shift the "random" weights slightly to simulate learning
          const seed = i * count2 + j;
          const weight = (Math.sin(seed + trainingStep * 0.5) + 1) / 2;
          const color = new THREE.Color().setHSL(0.3 + weight * 0.2, 1, 0.5); // Green-ish
          
          colorArray.push(color.r, color.g, color.b);
          colorArray.push(color.r, color.g, color.b);
        }
      }
    } 
    // Case 2: Pool -> FC (Flattening)
    else if (layer1.type === 'pool' && layer2.type === 'fc') {
        // Sample connections from feature maps to FC neurons
        // Connect each feature map center to a subset of FC neurons
        for (let i = 0; i < layer1.depth; i++) {
            const start = getNeuronPos(layer1, i, layer1.depth);
            
            // Connect to 20% of FC neurons randomly
            for (let j = 0; j < layer2.size[0]; j += 5) {
                const end = getNeuronPos(layer2, j, layer2.size[0]);
                
                points.push(start.x, start.y, start.z);
                points.push(end.x, end.y, end.z);
                
                const seed = i * layer2.size[0] + j;
                const weight = (Math.sin(seed + trainingStep * 0.5) + 1) / 2;
                const color = new THREE.Color().setHSL(0.6, 1, 0.5); // Blue-ish
                
                colorArray.push(color.r, color.g, color.b);
                colorArray.push(color.r, color.g, color.b);
            }
        }
    }
    // Case 3: Conv -> Pool or Input -> Conv (Feature map to Feature map)
    else {
        // Just draw lines between corresponding feature maps or centers
        // This is a simplification
        const count = Math.max(layer1.depth, layer2.depth);
        for (let i = 0; i < count; i++) {
             const start = getNeuronPos(layer1, i % layer1.depth, layer1.depth);
             const end = getNeuronPos(layer2, i % layer2.depth, layer2.depth);
             
             points.push(start.x, start.y, start.z);
             points.push(end.x, end.y, end.z);
             
             const color = new THREE.Color(0x444444);
             colorArray.push(color.r, color.g, color.b);
             colorArray.push(color.r, color.g, color.b);
        }
    }

    return {
      positions: new Float32Array(points),
      colors: new Float32Array(colorArray)
    };
  }, [layer1, layer2, trainingStep]);

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
        opacity={(active ? 0.3 : 0.05) * weight} 
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </lineSegments>
  );
}
