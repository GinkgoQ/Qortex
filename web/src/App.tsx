import { useEffect, useState } from "react";
import NetworkScene from "./NetworkScene";
import type { ConnectomeData } from "./types";

const HEADER_STYLE: React.CSSProperties = {
  position: "absolute",
  top: 0,
  left: 0,
  right: 0,
  padding: "14px 20px",
  fontFamily: "Lato, Roboto, 'Segoe UI', sans-serif",
  color: "#e5e7eb",
  pointerEvents: "none",
  zIndex: 1,
};

export default function App() {
  const [data, setData] = useState<ConnectomeData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("./assets/connectome.json")
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        return res.json();
      })
      .then(setData)
      .catch((err) => setError(String(err)));
  }, []);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <div style={HEADER_STYLE}>
        <div style={{ fontSize: 18, fontWeight: 700 }}>Connectome — 3D network</div>
        {data && (
          <div style={{ fontSize: 12, color: "#9ca3af", marginTop: 2 }}>
            {data.metric} · {data.node_definition} · {data.nodes.length} nodes · {data.edges.length} edges
          </div>
        )}
        {error && <div style={{ fontSize: 12, color: "#f87171", marginTop: 2 }}>Failed to load data: {error}</div>}
      </div>
      {data && <NetworkScene data={data} />}
    </div>
  );
}
