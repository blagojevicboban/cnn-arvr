import { useRef, useState, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text, Box, Line } from '@react-three/drei';
import * as THREE from 'three';

interface LayerProps {
  position: [number, number, number];
  type: 'input' | 'conv' | 'pool' | 'fc' | 'output';
  size: [number, number]; // Grid size [rows, cols] or [neurons]
  depth: number; // Number of feature maps
  label: string;
  active?: boolean;
  textureUrl?: string;
  featureMaps?: string[]; // Array of data URLs for feature maps
  activations?: number[]; // Array of activation values for FC/Output
  outputLabels?: string[]; // Labels for output neurons
  onClick?: () => void;
}

export function Layer({ position, type, size, depth, label, active, textureUrl, featureMaps, activations, outputLabels, onClick }: LayerProps) {
  const groupRef = useRef<THREE.Group>(null);
  const [texture, setTexture] = useState<THREE.Texture | null>(null);
  const [mapTextures, setMapTextures] = useState<THREE.Texture[]>([]);

  useEffect(() => {
    if (textureUrl) {
      new THREE.TextureLoader().load(textureUrl, (loadedTexture) => {
        loadedTexture.colorSpace = THREE.SRGBColorSpace;
        setTexture(loadedTexture);
      });
    } else {
      setTexture(null);
    }
  }, [textureUrl]);

  useEffect(() => {
    if (featureMaps && featureMaps.length > 0) {
        const loader = new THREE.TextureLoader();
        Promise.all(featureMaps.map(url => new Promise<THREE.Texture>((resolve) => {
            loader.load(url, (tex) => {
                tex.colorSpace = THREE.SRGBColorSpace;
                tex.minFilter = THREE.NearestFilter;
                tex.magFilter = THREE.NearestFilter;
                resolve(tex);
            });
        }))).then(textures => {
            setMapTextures(textures);
        });
    } else {
        setMapTextures([]);
    }
  }, [featureMaps]);
  
  // Animation logic for "activation"
  useFrame((state) => {
    if (groupRef.current && active) {
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
    }
  });

  const renderFeatureMaps = () => {
    const maps = [];
    const mapSize = 0.8;
    const gap = 0.1;
    
    // Calculate grid layout for depth (feature maps) - YZ plane parallel to X axis
    const cols = Math.ceil(Math.sqrt(depth));
    const rows = Math.ceil(depth / cols);
    
    for (let i = 0; i < depth; i++) {
      const row = Math.floor(i / cols);
      const col = i % cols;
      
      const y = (col - (cols - 1) / 2) * (mapSize + gap);
      const z = -(row - (rows - 1) / 2) * (mapSize + gap);
      
      // Use specific feature map texture if available, otherwise fallback to main texture or color
      const mapTex = mapTextures[i] || (i === 0 ? texture : null);

      maps.push(
        <group key={i} position={[-0.8, y, z]}>
          <Box args={[0.05, mapSize, mapSize]}>
            <meshStandardMaterial 
              color={mapTex ? "white" : (active ? "#4ade80" : "#60a5fa")} 
              transparent 
              opacity={0.9} 
              emissive={active ? "#4ade80" : "#000"}
              emissiveIntensity={active ? 0.2 : 0}
              map={mapTex}
              side={THREE.DoubleSide}
            />
          </Box>
          {/* Grid lines on the feature map to represent pixels */}
          {type !== 'fc' && type !== 'output' && (() => {
            const lines = [];
            const segments = Math.min(size[0], 10); // Limit grid to 10x10 for performance
            const step = mapSize / segments;
            const halfSize = mapSize / 2;
            
            // Vertical lines (along Y axis)
            for (let i = 0; i <= segments; i++) {
              const y = -halfSize + i * step;
              lines.push(
                <Line
                  key={`v${i}`}
                  points={[[0.02, y, -halfSize], [0.02, y, halfSize]]}
                  color={0xFFFFFF}
                  lineWidth={1.5}
                  opacity={0.8}
                />
              );
            }
            
            // Horizontal lines (along Z axis)
            for (let i = 0; i <= segments; i++) {
              const z = -halfSize + i * step;
              lines.push(
                <Line
                  key={`h${i}`}
                  points={[[0.02, -halfSize, z], [0.02, halfSize, z]]}  
                  color={0xFFFFFF}
                  lineWidth={1.5}
                  opacity={0.8}
                />
              );
            }
            
            return lines;
          })()}
        </group>
      );
    }
    return maps;
  };

  const renderInputDisplay = () => {
    // Display main input image/grid (separate from neurons) - YZ plane like feature maps
    const mapSize = 2;
    
    return (
      <group key="input-display" position={[-0.8, 0, 0]}>
        <Box args={[0.05, mapSize, mapSize]}>
          <meshStandardMaterial 
            color={texture ? "white" : (active ? "#fbbf24" : "#6b7280")} 
            transparent 
            opacity={0.9} 
            emissive={active ? "#fbbf24" : "#000"}
            emissiveIntensity={active ? 0.3 : 0}
            map={texture}
            side={THREE.DoubleSide}
          />
        </Box>
        
        {/* Grid lines to show 28x28 pixel structure - in YZ plane like neurons */}
        {(() => {
          const lines = [];
          const segments = 28;
          const step = mapSize / segments;
          const halfSize = mapSize / 2;
          
          // Vertical lines (along Y axis)
          for (let i = 0; i <= segments; i += 4) {
            const y = -halfSize + i * step;
            lines.push(
              <Line
                key={`v${i}`}
                points={[[0.02, y, -halfSize], [0.02, y, halfSize]]}
                color={0xFFFFFF}
                lineWidth={1.2}
                opacity={0.6}
              />
            );
          }
          
          // Horizontal lines (along Z axis)
          for (let i = 0; i <= segments; i += 4) {
            const z = -halfSize + i * step;
            lines.push(
              <Line
                key={`h${i}`}
                points={[[0.02, -halfSize, z], [0.02, halfSize, z]]}
                color={0xFFFFFF}
                lineWidth={1.2}
                opacity={0.6}
              />
            );
          }
          
          return lines;
        })()}
      </group>
    );
  };

  const renderFeatureMapsInput = () => {
    // For input layer visual representation as matrix
    return renderInputDisplay();
  };

  const renderNeurons = () => {
    const neurons = [];
    
    if (type === 'input') {
      // Input layer: 28x28 neurons (downsampled to 14x14 for performance)
      const gridSide = 14;
      const spacing = 0.15;
      const radius = 0.05;
      const start = -(gridSide - 1) * spacing / 2;
      
      for (let r = 0; r < gridSide; r++) {
        for (let c = 0; c < gridSide; c++) {
          neurons.push(
            <mesh key={`in-${r}-${c}`} position={[0, start + c * spacing, start + r * spacing]}>
              <sphereGeometry args={[radius, 8, 8]} />
              <meshStandardMaterial color={active ? "#fbbf24" : "#444"} />
            </mesh>
          );
        }
      }
    } else if (type === 'conv' || type === 'pool') {
      // Conv/Pool layers: Render neurons as grids behind each feature map
      const mapCols = Math.ceil(Math.sqrt(depth));
      const mapRows = Math.ceil(depth / mapCols);
      const mapSize = 0.8;
      const gap = 0.1;
      
      // Downsample neurons for performance (e.g., 6x6 for Conv, 4x4 for Pool)
      const neuronGridSize = type === 'conv' ? 6 : 4;
      const neuronSpacing = mapSize / (neuronGridSize + 1);
      const radius = type === 'conv' ? 0.04 : 0.06;

      for (let i = 0; i < depth; i++) {
        const mRow = Math.floor(i / mapCols);
        const mCol = i % mapCols;
        const mY = (mCol - (mapCols - 1) / 2) * (mapSize + gap);
        const mZ = -(mRow - (mapRows - 1) / 2) * (mapSize + gap);

        for (let r = 0; r < neuronGridSize; r++) {
          for (let c = 0; c < neuronGridSize; c++) {
            const y = mY - mapSize/2 + (c + 1) * neuronSpacing;
            const z = mZ - mapSize/2 + (r + 1) * neuronSpacing;
            neurons.push(
              <mesh key={`map-${i}-n-${r}-${c}`} position={[0.1, y, z]}>
                <sphereGeometry args={[radius, 8, 8]} />
                <meshStandardMaterial color={active ? "#4ade80" : "#444"} />
              </mesh>
            );
          }
        }
      }
    } else if (type === 'fc' || type === 'output') {
      // FC: 10x12 vertical-first grid, Output: 1x10 vertical stack
      const totalNeurons = size[0];
      const gridWidth = type === 'fc' ? 10 : 1; // Width along Z
      const gridHeight = Math.ceil(totalNeurons / gridWidth); // Height along Y
      
      const spacing = type === 'output' ? 0.6 : 0.35;
      const radius = type === 'output' ? 0.22 : 0.15;
      const startZ = -(gridWidth - 1) * spacing / 2;
      const startY = (gridHeight - 1) * spacing / 2;
      
      for (let i = 0; i < totalNeurons; i++) {
        const row = Math.floor(i / gridWidth); // Vertical level (0 is top)
        const col = i % gridWidth;             // Horizontal position in row
        const y = startY - row * spacing;
        const z = startZ + col * spacing;
        
        // Visualize activation intensity if available
        const activation = activations ? activations[i] : 0;
        const baseIntensity = active ? 0.2 : 0;
        const intensity = activations ? Math.min(1.2, baseIntensity + activation * 0.8) : (active ? 0.6 : 0);
        
        // For Output layer, show probability bars
        const isOutput = type === 'output';
        const isFC = type === 'fc';
        const color = isOutput ? (activation > 0.5 ? "#22c55e" : "#facc15") : (isFC ? "#fbbf24" : "#f87171");
        const outputLabel = outputLabels ? outputLabels[i] : `${i}`;
        
        neurons.push(
          <group key={i} position={[0, y, z]}>
            <mesh>
              <sphereGeometry args={[radius, 16, 16]} />
              <meshStandardMaterial 
                color={active ? color : "#444"} 
                emissive={active ? color : "#000"}
                emissiveIntensity={intensity}
                side={THREE.DoubleSide}
              />
            </mesh>
            {isOutput && (
              <group position={[0.4, 0, 0]}>
                {/* Background Scale Bar (Horizontal along X) */}
                <Box args={[1.5, 0.08, 0.02]} position={[0.75, 0, 0]}>
                   <meshStandardMaterial color="#222" transparent opacity={0.5} />
                </Box>

                {/* Active Scale Bar */}
                {activation > 0.01 && (
                  <Box args={[1.5 * activation, 0.1, 0.03]} position={[(1.5 * activation) / 2, 0, 0.01]}>
                    <meshStandardMaterial 
                      color={activation > 0.5 ? "#22c55e" : (active ? "#facc15" : "#6b7280")} 
                      emissive={activation > 0.5 ? "#22c55e" : "#000"}
                      emissiveIntensity={0.5}
                    />
                  </Box>
                )}

                {/* Percentage Text */}
                <Text 
                  position={[1.6, 0, 0]} 
                  fontSize={0.12} 
                  color={activation > 0.1 ? "#22c55e" : "#666"} 
                  anchorX="left" 
                  anchorY="middle"
                >
                  {(activation * 100).toFixed(0)}%
                </Text>

                {/* Large Label (Digit) - Moved to the right of % */}
                <Text 
                  position={[2.0, 0, 0]} 
                  fontSize={0.45} 
                  fontWeight="bold"
                  color={activation > 0.5 ? "#22c55e" : "white"} 
                  anchorX="left" 
                  anchorY="middle"
                >
                  {outputLabel}
                </Text>
              </group>
            )}
          </group>
        );
      }
    }
    return neurons;
  };

  return (
    <group ref={groupRef} position={position} onClick={(e) => {
      e.stopPropagation();
      onClick?.();
    }}>
      <Text
        position={[0.5, (() => {
          if (type === 'fc' || type === 'output') {
            const gridWidth = type === 'fc' ? 10 : 1;
            const gridHeight = Math.ceil(size[0] / gridWidth);
            const spacing = type === 'output' ? 0.6 : 0.35;
            const totalHeight = (gridHeight - 1) * spacing;
            return totalHeight / 2 + 1.2;
          } else {
            return 2.5;
          }
        })(), 0]}
        fontSize={0.3}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {label}
      </Text>
      
      {type === 'fc' || type === 'output' ? 
        renderNeurons() : 
        (type === 'conv' || type === 'pool' || type === 'input') ? 
          <>
            {/* Feature maps/Grid in front */}
            <group position={[0, 0, 0.5]}>
              {type === 'input' ? renderFeatureMapsInput() : renderFeatureMaps()}
            </group>
            {/* Neurons behind */}
            <group position={[0, 0, -0.5]}>
              {renderNeurons()}
            </group>
          </> : 
          renderFeatureMaps()
      }
      
      {/* Base platform for the layer */}
      <mesh position={[0, -3, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[2, 32]} />
        <meshStandardMaterial color="#333" transparent opacity={0.5} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}
