import { useRef, useState, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text, Box } from '@react-three/drei';
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
    
    // Calculate grid layout for depth (feature maps)
    const cols = Math.ceil(Math.sqrt(depth));
    const rows = Math.ceil(depth / cols);
    
    for (let i = 0; i < depth; i++) {
      const row = Math.floor(i / cols);
      const col = i % cols;
      
      const x = (col - (cols - 1) / 2) * (mapSize + gap);
      const y = -(row - (rows - 1) / 2) * (mapSize + gap);
      
      // Use specific feature map texture if available, otherwise fallback to main texture or color
      const mapTex = mapTextures[i] || (i === 0 ? texture : null);

      maps.push(
        <group key={i} position={[x, y, 0]}>
          <Box args={[mapSize, mapSize, 0.05]}>
            <meshStandardMaterial 
              color={mapTex ? "white" : (active ? "#4ade80" : "#60a5fa")} 
              transparent 
              opacity={0.9} 
              emissive={active ? "#4ade80" : "#000"}
              emissiveIntensity={active ? 0.2 : 0}
              map={mapTex}
            />
          </Box>
          {/* Grid lines on the feature map to represent pixels */}
          {type !== 'fc' && type !== 'output' && (
             <gridHelper 
               args={[mapSize, size[0], 0xffffff, 0xffffff]} 
               rotation={[Math.PI / 2, 0, 0]} 
               position={[0, 0, 0.026]} 
             />
          )}
        </group>
      );
    }
    return maps;
  };

  const renderNeurons = () => {
    const neurons = [];
    const height = (size[0] - 1) * 0.5;
    
    for (let i = 0; i < size[0]; i++) {
      const y = (i - (size[0] - 1) / 2) * 0.5;
      
      // Visualize activation intensity if available
      const activation = activations ? activations[i] : 0;
      const intensity = activations ? activation : (active ? 0.8 : 0);
      
      // For Output layer, show probability bars
      const isOutput = type === 'output';
      const outputLabel = outputLabels ? outputLabels[i] : `${i}`;
      
      neurons.push(
        <group key={i} position={[0, y, 0]}>
            <mesh>
            <sphereGeometry args={[0.15, 16, 16]} />
            <meshStandardMaterial 
                color={active ? "#facc15" : "#f87171"} 
                emissive={active ? "#facc15" : "#000"}
                emissiveIntensity={intensity}
            />
            </mesh>
            {isOutput && activations ? (
                <group position={[0.3, 0, 0]}>
                    <Box args={[activation * 2, 0.1, 0.1]} position={[activation, 0, 0]}>
                        <meshStandardMaterial color="#22c55e" />
                    </Box>
                    <Text position={[activation * 2 + 0.2, 0, 0]} fontSize={0.15} color="white" anchorX="left">
                        {(activation * 100).toFixed(0)}%
                    </Text>
                    <Text position={[activation * 2 + 0.8, 0, 0]} fontSize={0.15} color="white" anchorX="left">
                        {outputLabel}
                    </Text>
                </group>
            ) : (
                isOutput && (
                    <Text position={[0.4, 0, 0]} fontSize={0.15} color="white" anchorX="left" anchorY="middle">
                        {outputLabel}
                    </Text>
                )
            )}
        </group>
      );
    }
    return neurons;
  };

  return (
    <group ref={groupRef} position={position} onClick={(e) => {
      e.stopPropagation();
      onClick?.();
    }}>
      <Text
        position={[0, type === 'fc' || type === 'output' ? size[0] * 0.25 + 1 : 2.5, 0]}
        fontSize={0.3}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {label}
      </Text>
      
      {type === 'fc' || type === 'output' ? renderNeurons() : renderFeatureMaps()}
      
      {/* Base platform for the layer */}
      <mesh position={[0, -3, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[2, 32]} />
        <meshStandardMaterial color="#333" transparent opacity={0.5} />
      </mesh>
    </group>
  );
}
