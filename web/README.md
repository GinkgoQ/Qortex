# qortex-web

A minimal, real Three.js/React-Three-Fiber scene rendering a Qortex
connectivity export as an interactive 3D network graph — the scoped
starting point for a JS/WebGL frontend, separate from the Python package's
static-figure generation in `src/qortex/visualize/`.

This is deliberately small: one scene, one data shape, no routing, no state
library. Grow it only when a second real view is needed.

## Run it

```bash
cd web
npm install
npm run dev       # http://localhost:5173
```

## Data

`public/assets/connectome.json` is a static export produced by the Python
side — regenerate it with:

```bash
python3 web/scripts/export_connectome.py
```

The script builds a `ConnectivityMatrix` via
`qortex.neuroclassic.connectivity` (the same real computation
`qortex.visualize.connectivity_figure` renders as a static PNG) and writes
node degree + edge weights to JSON. No values are fabricated in the
frontend — node layout (`fibonacciSphere` in `src/NetworkScene.tsx`) is a
deterministic placement algorithm, not data.

## Stack

- Vite + React + TypeScript
- `@react-three/fiber` + `@react-three/drei` over raw Three.js, for
  declarative scene composition
- No routing/state library — add one only when a second page needs it

## Relationship to `src/qortex/visualize/`

The Python package remains the source of truth for report-quality static
figures (PNG/HTML) and is unaffected by this app. This app is for
interactive 3D exploration only, reading Python-exported JSON — it does not
replace `connectivity_figure()`, `niivue_html()`, or any other Python
renderer.
