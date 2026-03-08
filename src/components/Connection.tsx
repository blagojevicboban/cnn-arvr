import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Line, Sphere } from '@react-three/drei';
import * as THREE from 'three';

interface ConnectionProps {
  start: [number, number, number];
  end: [number, number, number];
  active?: boolean;
  weight?: number; // Connection weight (0-1), affects line thickness
}

export function Connection({ start, end, active, weight = 0.5 }: ConnectionProps) {
  const particleRef = useRef<THREE.Mesh>(null);
  const lineRef = useRef<any>(null);
  
  const points = useMemo(() => {
    const startVec = new THREE.Vector3(...start);
    const endVec = new THREE.Vector3(...end);
    const midVec = new THREE.Vector3().addVectors(startVec, endVec).multiplyScalar(0.5);
    // Add some arch
    midVec.y += 2;
    
    const curve = new THREE.QuadraticBezierCurve3(startVec, midVec, endVec);
    return curve.getPoints(20);
  }, [start, end]);

  useFrame((state, delta) => {
    const time = state.clock.elapsedTime;
    
    if (particleRef.current && active) {
      const t = (time * 1.5) % 1;
      const curve = new THREE.QuadraticBezierCurve3(
        new THREE.Vector3(...start),
        new THREE.Vector3(...start).add(new THREE.Vector3(...end)).multiplyScalar(0.5).add(new THREE.Vector3(0, 2, 0)),
        new THREE.Vector3(...end)
      );
      const pos = curve.getPoint(t);
      particleRef.current.position.copy(pos);
      
      // Pulse effect for particle
      const scale = 1 + Math.sin(time * 10) * 0.3;
      particleRef.current.scale.setScalar(scale);
    }

    if (lineRef.current) {
        if (active) {
            // Flow animation
            lineRef.current.material.dashOffset -= delta * 2;
            
            // Pulsing opacity
            lineRef.current.material.opacity = 0.6 + Math.sin(time * 5) * 0.2;
            
            // Color shift (Green to Cyan pulse)
            const hue = 0.35 + Math.sin(time * 2) * 0.1; 
            lineRef.current.material.color.setHSL(hue, 1, 0.5);
        } else {
            lineRef.current.material.opacity = 0.1;
            lineRef.current.material.color.set("#333");
        }
    }
  });

  return (
    <group>
      <Line 
        ref={lineRef}
        points={points} 
        color={active ? "#4ade80" : "#333"} 
        lineWidth={active ? 20 : 2}
        transparent 
        opacity={active ? 0.8 : 0.2} 
        dashed={true}
        dashScale={5}
        dashSize={active ? 0.5 : 100}
        gapSize={active ? 0.5 : 0}
      />
      {active && (
        <Sphere ref={particleRef} args={[0.15, 16, 16]}>
          <meshBasicMaterial color="#fff" toneMapped={false} />
          <pointLight distance={1} intensity={2} color="#4ade80" />
        </Sphere>
      )}
    </group>
  );
}
