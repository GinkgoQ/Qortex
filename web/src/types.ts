export interface ConnectomeNode {
  label: string;
  degree: number;
}

export interface ConnectomeEdge {
  source: number;
  target: number;
  weight: number;
}

export interface ConnectomeData {
  metric: string;
  node_definition: string;
  nodes: ConnectomeNode[];
  edges: ConnectomeEdge[];
}
