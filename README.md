---
title: NeuroSignal India
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - traffic-optimization
  - computer-vision
  - llm-agent
  - metadrive
  - yolov8
  - gemini
---

# NeuroSignal India — Dynamic AI Traffic Flow Optimizer & Emergency Grid

![System Architecture](<shapes at 26-03-22 13.30.11.png>)

An intelligent traffic management system that uses real-time computer vision to dynamically adjust signal timings based on live traffic density, with an AI-powered green corridor feature to prioritize routes for emergency vehicles such as ambulances and fire services.


## Problem

Urban Indian intersections run on fixed-timer signals that ignore actual traffic load, lane-less mixed-vehicle flow, and emergency vehicle priority. Congestion wastes fuel, delays ambulances, and causes preventable fatalities.

## Solution Overview

A software-only MVP composed of three layers working in a closed loop:

1. **Virtual CCTV (Simulation).** MetaDrive `IntersectionEnv` renders a top-down orthographic view of a 4-way junction at ~750 FPS. This frame is the camera input — no hardware required.
2. **Real-Time Computer Vision.** YOLOv8n detects vehicles on every frame. `supervision.PolygonZone` partitions the frame into N/S/E/W approach zones. India-standard **Passenger Car Equivalent (PCE)** weights convert raw counts into per-direction traffic load scores.
3. **LLM Signal Agent.** Gemini 2.5 Flash receives the structured PCE state plus the annotated top-down frame in a single multimodal call. An India-specific system prompt encodes lane-less rules, starvation prevention, and emergency protocol. `instructor` + Pydantic guarantee a validated `SignalDecision` that is applied to MetaDrive signals via `engine.traffic_manager`.

On ambulance detection, Dijkstra over a road graph (`networkx`, upgradeable to a Delhi OSM graph via `osmnx`) computes the optimal corridor. The route is injected as emergency context into the LLM prompt, which then reasons about green-wave priority across the path.

## Core Components

| Layer | Component | Role |
|---|---|---|
| Simulation | MetaDrive 0.4.3 `IntersectionEnv` | Virtual CCTV + signal state manager |
| Computer Vision | YOLOv8n | Vehicle detection (car, bus, truck, motorcycle) |
| Computer Vision | `supervision.PolygonZone` | N/S/E/W zone segmentation |
| Computer Vision | PCE Calculator | IRC-standard weighted load per zone |
| LLM Agent | Gemini 2.5 Flash (`instructor` + Pydantic) | Multimodal signal decision with chain-of-thought |
| LLM Agent | Safety Enforcer | Hard constraints on every LLM output (duration clamp, starvation override, emergency floor) |
| LLM Agent | Rule-Based Controller | Fallback on LLM timeout + benchmark baseline |
| Routing | `networkx` Dijkstra | Ambulance corridor computation |
| Backend | FastAPI + `python-socketio` | REST API + real-time event stream |
| Logging | SQLite (`sqlite-utils`) | Decision log for KPI aggregation |
| Frontend | React + Vite + Recharts | Judge-facing dashboard (camera, reasoning, KPI, zones) |

## Key Formulas

- **Green time:** `green_s = clamp(15 + 1.5 * zone_PCE, 15, 60)`
- **Pressure score (rule-based baseline):** `pressure = PCE * log(1 + wait_s)`
- **PCE weights:** motorcycle = 0.5, auto-rickshaw = 1.2, car = 1.0, minibus = 1.5, bus/truck = 2.5

## FastAPI Endpoints

- `GET /health`
- `GET /api/kpi`
- `GET /api/decisions`
- `POST /api/control`
- `POST /api/emergency`
- WebSocket events: `frame`, `decision`, `kpi`, `zones`, `emergency`

## Directory Layout

```
Project_meta_traffic/
  agent/            # LLM agent, instructor wrapper, safety enforcer
  api/              # FastAPI app + Socket.IO handlers
  benchmark/        # KPI comparison runner (LLM vs rule-based vs fixed-timer)
  cv/               # YOLO detector, PolygonZone, PCE calculator
  data/             # decisions.db, benchmark results
  emergency/        # Ambulance detector + corridor trigger
  envs/             # MetaDrive IntersectionEnv wrapper
  routing/          # Dijkstra over road graph
  server/           # ASGI entrypoint
  Dockerfile
  openenv.yaml
  requirements.txt
```

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the Gemini API key:

```bash
set GEMINI_API_KEY=your_key_here
```

Start the server:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Docker Run

```bash
docker build -t neurosignal-india:latest .
docker run --rm -p 8000:8000 -e GEMINI_API_KEY=$GEMINI_API_KEY neurosignal-india:latest
```

## Benchmarking

Three controllers are logged side-by-side to `data/decisions.db` with a `controller_type` column (`llm`, `rule_based`, `fixed_timer`). Run the comparison:

```bash
python benchmark/run_all.py
```

Reported KPIs: average wait time, ambulance corridor clearance time, throughput, decision latency.

## Environment

- OS: Windows 11 / Linux
- Python: 3.12
- MetaDrive: 0.4.3 (~748 FPS headless verified)
- LLM: Gemini 2.5 Flash via `instructor.from_provider("google/gemini-2.5-flash")`

## Roadmap (Phase 2)

- **SUMO TraCI** for real-intersection signal control (`traci.trafficlight.setPhase()`).
- **TrafficDojo** bridge to sync SUMO signals with MetaDrive 3D visuals.
- **CoLLMLight** multi-agent coordination for city-wide green waves.
- **osmnx** Delhi OSM graph for realistic ambulance routing.
- Fine-tuned YOLO ambulance class (replacing HSV red-mask heuristic).

## References

- LLMLight (KDD 2025, Geneva Gold Medal) — LLM-as-agent for traffic signal control: https://github.com/usail-hkust/LLMTSCS
- CoLLMLight — cooperative multi-junction LLM agents: https://github.com/usail-hkust/CoLLMLight
- MetaDrive: https://github.com/metadriverse/metadrive
