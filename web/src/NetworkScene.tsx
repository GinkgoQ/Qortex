import { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Text, Line } from "@react-three/drei";
import type { ConnectomeData } from "./types";

/** Deterministic even-spacing layout — not a fabricated data value, just node placement. */
function fibonacciSphere(n: number, radius: number): [number, number, number][] {
  const points: [number, number, number][] = [];
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / Math.max(n - 1, 1)) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = goldenAngle * i;
    points.push([Math.cos(theta) * r * radius, y * radius, Math.sin(theta) * r * radius]);
  }
  return points;
}

function Nodes({ data, positions }: { data: ConnectomeData; positions: [number, number, number][] }) {
  const maxDegree = Math.max(1, ...data.nodes.map((n) => n.degree));
  return (
    <>
      {data.nodes.map((node, i) => {
        const size = 0.12 + 0.14 * (node.degree / maxDegree);
        return (
          <group key={node.label} position={positions[i]}>
            <mesh>
              <sphereGeometry args={[size, 24, 24]} />
              <meshStandardMaterial color="#4f46e5" roughness={0.35} metalness={0.1} />
            </mesh>
            <Text
              position={[0, size + 0.18, 0]}
              fontSize={0.16}
              color="#e5e7eb"
              anchorX="center"
              anchorY="bottom"
            >
              {node.label}
            </Text>
          </group>
        );
      })}
    </>
  );
}

function Edges({ data, positions }: { data: ConnectomeData; positions: [number, number, number][] }) {
  const maxAbs = Math.max(1e-6, ...data.edges.map((e) => Math.abs(e.weight)));
  return (
    <>
      {data.edges.map((edge, i) => {
        const a = positions[edge.source];
        const b = positions[edge.target];
        const color = edge.weight >= 0 ? "#dc2626" : "#2563eb";
        const opacity = 0.25 + 0.65 * (Math.abs(edge.weight) / maxAbs);
        return (
          <Line
            key={i}
            points={[a, b]}
            color={color}
            lineWidth={1 + 2.5 * (Math.abs(edge.weight) / maxAbs)}
            transparent
            opacity={opacity}
          />
        );
      })}
    </>
  );
}

export default function NetworkScene({ data }: { data: ConnectomeData }) {
  const positions = useMemo(() => fibonacciSphere(data.nodes.length, 2.4), [data.nodes.length]);

  return (
    <Canvas camera={{ position: [0, 0, 7], fov: 50 }}>
      <color attach="background" args={["#0b0b0f"]} />
      <ambientLight intensity={0.6} />
      <pointLight position={[5, 5, 5]} intensity={1.2} />
      <pointLight position={[-5, -3, -5]} intensity={0.4} />
      <Nodes data={data} positions={positions} />
      <Edges data={data} positions={positions} />
      <OrbitControls enableDamping dampingFactor={0.08} rotateSpeed={0.6} />
    </Canvas>
  );
}
