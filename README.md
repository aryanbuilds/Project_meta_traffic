# NeuroSignal India 🚦 🚑

**AI-Powered Dynamic Traffic Management System**

An intelligent traffic signal control system combining **computer vision**, **large language models**, and **emergency routing** to optimize vehicle flow and prioritize emergency vehicles in real-world traffic scenarios.

**Project**: India Innovates 2026 Competition | **Status**: MVP Phase (Core Features Complete)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Features & Capabilities](#features--capabilities)
3. [Architecture Overview](#architecture-overview)
4. [System Requirements](#system-requirements)
5. [Quick Start](#quick-start)
6. [Installation & Setup](#installation--setup)
7. [Configuration](#configuration)
8. [Usage](#usage)
9. [API Documentation](#api-documentation)
10. [Module Architecture](#module-architecture)
11. [Computer Vision Pipeline](#computer-vision-pipeline)
12. [LLM Agent Details](#llm-agent-details)
13. [Emergency Corridor System](#emergency-corridor-system)
14. [Testing](#testing)
15. [Project Status](#project-status)
16. [Troubleshooting](#troubleshooting)
17. [Contributing](#contributing)
18. [License & Attribution](#license--attribution)

---

## Project Overview

**NeuroSignal India** is a software-only AI traffic management system designed to solve urban congestion in Indian road networks. The system runs in the **MetaDrive simulator** and combines three core technologies:

- **Real-Time Computer Vision**: YOLOv8n-based vehicle detection and classification on simulated CCTV frames
- **Multimodal LLM Reasoning**: Google Gemini processes both structured traffic metrics and raw frame images to make intelligent signal decisions
- **Emergency Routing**: Dijkstra-based route optimization on real Delhi road networks to enable green corridors for ambulances

### Problem Statement

Urban traffic congestion in India causes:
- 2-5% GDP loss annually
- Emergency response delays (ambulances blocked by congestion)
- Inefficient signal timing based on fixed schedules
- Lack of real-time emergency vehicle prioritization

**NeuroSignal** addresses these through:
1. **Adaptive signal timing** based on real-time vehicle density and vehicle type (using IRC PCE weights)
2. **Intelligent emergency corridors** that automatically grant green lights for ambulances along optimal routes
3. **Multimodal LLM decision-making** that reasons over both numerical metrics and visual frame data

### Competition Context

- **Competition**: India Innovates 2026
- **Evaluation**: Judges, traffic engineers, policy makers, competition organizers
- **Deliverables**: MVP simulation, API backend, decision logging, real-world feasibility analysis

---

## Features & Capabilities

### 🎯 Core Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Vehicle Detection** | YOLOv8n detects motorcycles, cars, buses, trucks in 4 zones | ✅ Complete |
| **PCE-Weighted Traffic Analysis** | IRC/MORTH standard vehicle equivalence (motorcycle=0.5, car=1.0, bus=2.5) | ✅ Complete |
| **Adaptive Signal Control** | Intelligent green time allocation (15–60 sec) based on zone PCE density | ✅ Complete |
| **LLM Decision Engine** | Google Gemini multimodal reasoning (text + image) for signal decisions | ✅ Complete |
| **Emergency Green Corridor** | Automatic ambulance detection and route-based green grant | ✅ Complete |
| **REST/WebSocket API** | FastAPI backend with real-time event streaming for dashboards | ✅ Complete |
| **Decision Logging** | SQLite persistence of all signal decisions and KPI metrics | ✅ Complete |

### 🔧 Advanced Capabilities

- **Ambulance Detection**: HSV-based red siren color detection + manual override API
- **Route Optimization**: Real-time Dijkstra routing on OpenStreetMap Delhi graph
- **Dual-Mode Reasoning**: Falls back to rule-based decisions if LLM unavailable
- **Multi-Agent Support**: MetaDrive multi-agent environment (configurable)
- **Starvation Prevention**: Phase rotation ensures green time for all directions
- **FPS-Capped Streaming**: Throttled WebSocket events (configurable 1–30 FPS)

---

## Architecture Overview

### High-Level Data Flow

```
MetaDrive Simulation Environment
    ↓
    ├─→ Frame Capture (top-down 800×800px)
    ├─→ Vehicle Detection (YOLOv8n)
    └─→ Zone Classification (4 polygon zones: N/S/E/W)
    ↓
Computer Vision Pipeline
    ├─→ Per-Zone PCE Calculation
    └─→ State Builder (traffic metrics JSON)
    ↓
Decision Engine
    ├─→ LLM Agent (Gemini multimodal) OR Rule-Based Fallback
    └─→ Safety Validator
    ↓
Signal Controller
    ├─→ Apply Decision to MetaDrive Traffic Lights
    └─→ Log to Database
    ↓
Simulation Broadcaster
    ├─→ WebSocket Events (frame, decision, zones, kpi, emergency)
    └─→ REST API Endpoints (health, control, emergency, kpi, decisions)
```

### System Components

| Component | Module | Responsibility |
|-----------|--------|-----------------|
| **Simulation** | `envs/` | MetaDrive environment, traffic light control, top-down rendering |
| **Detection** | `cv/` | YOLOv8n inference, zone polygon assignment, PCE weighting, frame annotation |
| **Decision Making** | `agent/` | LLM prompting, state building, rule-based fallback, safety validation |
| **Control** | `agent/signal_controller.py` | Apply decisions to traffic lights |
| **Emergency Handling** | `emergency/` | Ambulance detection, corridor lifecycle, route computation |
| **Routing** | `routing/` | Delhi OSM graph, Dijkstra pathfinding |
| **Backend API** | `api/` | FastAPI server, Socket.IO events, control endpoints |
| **Persistence** | `data/` | SQLite logging, KPI aggregation, decision history |

---

## System Requirements

### Minimum Specifications

- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.11 or higher
- **RAM**: 8 GB (16 GB+ recommended for smooth YOLOv8n inference)
- **VRAM**: 2 GB (NVIDIA/CUDA capable; CPU fallback available)
- **Storage**: 5 GB (for MetaDrive assets and YOLOv8n model weights)
- **Network**: Internet for first-run downloads (MetaDrive, models, OSM data)

### Required Credentials

- **Gemini API Key** (free tier at https://aistudio.google.com) — 1,500 requests/day limit
- No credit card required; sufficient for all demo and competition scenarios

### Python Dependencies

All managed in [requirements.txt](requirements.txt); major packages:
- `metadrive-simulator==0.4.3` — Traffic simulation engine
- `ultralytics` — YOLOv8 vehicle detection
- `google-genai` — Gemini API client
- `instructor` — Pydantic output validation from LLM
- `fastapi` + `python-socketio` — REST/WebSocket backend
- `osmnx`, `networkx` — Route computation
- `sqlite-utils` — Decision logging

---

## Quick Start

### 1. Clone & Setup (5 minutes)

```powershell
# Windows PowerShell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r .\requirements.txt

# Linux/macOS
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get Gemini API Key (2 minutes)

1. Visit https://aistudio.google.com
2. Sign in with Google account (free, no credit card needed)
3. Create a new API key
4. Copy key and add to `.env`:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Verify Installation (2 minutes)

```powershell
python -c "import cv2, fastapi, socketio, uvicorn; print('✓ imports-ok')"
python -c "from api.app import socket_app; print('✓ app-import-ok')"
```

### 4. Start Backend (1 minute)

```powershell
uvicorn api.app:socket_app --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### 5. Test API (1 minute)

In a new terminal:

```powershell
# Check health
curl http://localhost:8000/health

# Start simulation
curl -X POST http://localhost:8000/api/control `
  -H "Content-Type: application/json" `
  -d '{"action":"start"}'
```

---

## Installation & Setup

### Detailed Windows Setup

#### Prerequisites

- **Python 3.11** (download from https://python.org)
- **PowerShell** (built-in on Windows 10+)
- Internet connection

#### Step 1: Create Virtual Environment

```powershell
cd D:\Project_meta_traffic
py -3.11 -m venv .venv
```

If you get "execution policy" errors:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

#### Step 2: Activate Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
# Prompt should now show (.venv) at the start
```

#### Step 3: Install Dependencies

```powershell
# Option A: Using uv (faster, recommended)
uv pip install -r .\requirements.txt

# Option B: Using pip
pip install -r requirements.txt
```

First run will download:
- MetaDrive simulator assets (~2 GB)
- YOLOv8n weights (~180 MB)
- OpenStreetMap data for Delhi (cached)

#### Step 4: Configure Environment Variables

Create `.env` file in project root:

```env
# Gemini API Key (REQUIRED)
GEMINI_API_KEY=your_api_key_here

# Simulation Settings
AGENT_POLICY_MODE=manual
SIM_ENV_MODE=single
YOLO_INFERENCE_RATE=3
LLM_DECISION_INTERVAL=10
EMERGENCY_HOLD_S=30
BROADCAST_FPS_CAP=10

# Frame Rendering
FRAME_SIZE=800
TOPDOWN_AUTO_FIT=1
TOPDOWN_MAX_SCREEN_RATIO=0.65
```

#### Step 5: Verify Imports

```powershell
python -c "import cv2, fastapi, socketio, uvicorn; print('imports-ok')"
python -c "from api.app import socket_app; print('app-import-ok')"
```

Expected: `imports-ok` + `app-import-ok`

### Linux/macOS Setup

```bash
# Create venv
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env
cat > .env << 'EOF'
GEMINI_API_KEY=your_api_key_here
YOLO_INFERENCE_RATE=3
SIM_ENV_MODE=single
EOF

# Verify
python -c "from api.app import socket_app; print('✓ Ready')"
```

---

## Configuration

### Environment Variables

All configuration via `.env` file. Defaults compatible with free Gemini tier.

#### Required

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Free Gemini API key from https://aistudio.google.com |

#### Simulation Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_POLICY_MODE` | `manual` | `manual` (no control), `rule_based`, `llm` |
| `SIM_ENV_MODE` | `single` | `single` (1 agent) or `multi` (multi-agent) |
| `LLM_DECISION_INTERVAL` | `10` | Steps between LLM decisions (1=every step, 10=every 10 steps) |
| `YOLO_INFERENCE_RATE` | `3` | Steps between YOLO inferences (3=every 3 steps) |
| `EMERGENCY_HOLD_S` | `30` | Green hold duration for ambulance corridor (15–60 sec) |

#### Frame & Rendering

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAME_SIZE` | `512` | Top-down frame resolution (512–1024 pixels) |
| `TOPDOWN_AUTO_FIT` | `1` | Auto-fit view to intersection (1=yes, 0=no) |
| `TOPDOWN_MAX_SCREEN_RATIO` | `0.65` | Max viewport ratio (0.5–1.0) |
| `FRAME_ENCODE_QUALITY` | `85` | JPEG quality for WebSocket (1–100) |

#### Backend & Streaming

| Variable | Default | Description |
|----------|---------|-------------|
| `BROADCAST_FPS_CAP` | `10` | WebSocket event throttle (1–30 FPS) |
| `API_PORT` | `8000` | FastAPI port |
| `ENABLE_DECISION_LOGGING` | `1` | Log decisions to SQLite (1=yes, 0=no) |

#### Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `TORCH_DEVICE` | `auto` | `cpu`, `cuda`, or `auto` (uses GPU if available) |
| `YOLO_HALF_PRECISION` | `0` | Use FP16 for YOLO (faster, slightly less accurate) |
| `MAX_AGENTS` | `10` | Multi-agent mode max agents |

---

## Usage

### Running the Simulation

#### Option A: Direct Execution (Headless)

```powershell
python main.py
```

Runs the full loop with LLM agent (if configured). Output:
- Simulation frames logged
- Decisions persisted to SQLite
- WebSocket events broadcasted (if API running)

#### Option B: With API Backend (Recommended)

Terminal 1 — Start backend:

```powershell
uvicorn api.app:socket_app --port 8000 --reload
```

Terminal 2 — Control via API:

```powershell
# Start simulation
curl -X POST http://localhost:8000/api/control -H "Content-Type: application/json" -d '{"action":"start"}'

# Check status
curl http://localhost:8000/api/status

# Pause
curl -X POST http://localhost:8000/api/control -d '{"action":"pause"}'

# Resume
curl -X POST http://localhost:8000/api/control -d '{"action":"resume"}'

# Stop
curl -X POST http://localhost:8000/api/control -d '{"action":"stop"}'

# Trigger emergency ambulance
curl -X POST http://localhost:8000/api/emergency \
  -H "Content-Type: application/json" \
  -d '{"ambulance_id":"AMB-001","destination":"AIIMS","origin_direction":"north"}'

# Get KPI summary
curl http://localhost:8000/api/kpi

# Get last 20 decisions
curl "http://localhost:8000/api/decisions?limit=20"
```

### Inspection & Debug Scripts

#### List Available Gemini Models

```powershell
python list_models.py
```

Output: Available Gemini models and context windows

#### Inspect Traffic Lights

```powershell
python inspect_traffic_lights.py
```

Starts a minimal MetaDrive env and prints traffic light phase every 2 seconds. Validates signal control API.

#### Test Instructor API

```powershell
python test_instructor_api.py
```

Verifies Gemini + instructor + Pydantic integration before full simulation.

---

## API Documentation

### REST Endpoints

All endpoints return JSON. Base URL: `http://localhost:8000`

#### Health & Status

**GET /health**
```json
{
  "step": 1250,
  "running": true,
  "paused": false,
  "fps": 24.5,
  "last_error": "",
  "latest_decision": { ... }
}
```

**GET /api/status**

Same as `/health`. Status includes:
- `step`: Current simulation step
- `running`: Simulation active?
- `paused`: Sim paused?
- `fps`: Frames per second
- `latest_decision`: Most recent signal decision

#### Control

**POST /api/control**

Start, pause, resume, stop, or reset simulation.

**Request:**
```json
{
  "action": "start" | "pause" | "resume" | "stop" | "reset"
}
```

**Response:**
```json
{
  "step": 1250,
  "running": true,
  "paused": false,
  "startup_error": null
}
```

#### Emergency

**POST /api/emergency**

Manually trigger ambulance and calculate green corridor route.

**Request:**
```json
{
  "ambulance_id": "AMB-001",
  "destination": "AIIMS",
  "origin_direction": "north"
}
```

**Response:**
```json
{
  "queued": true,
  "ambulance_id": "AMB-001",
  "destination": "AIIMS",
  "origin_direction": "north",
  "route": [
    {"lat": 28.5662, "lon": 77.2125},
    {"lat": 28.5670, "lon": 77.2130},
    ...
  ],
  "nonce": "evt-uuid-1234"
}
```

#### KPI Summary

**GET /api/kpi**

Returns aggregated performance metrics.

**Response:**
```json
{
  "total_vehicles_detected": 450,
  "total_steps": 1250,
  "avg_zone_pce": {
    "north": 12.5,
    "south": 18.3,
    "east": 15.2,
    "west": 14.1
  },
  "ambulance_events": 2,
  "decisions_made": 125,
  "step": 1250,
  "running": true,
  "paused": false
}
```

#### Decision History

**GET /api/decisions?limit=50**

Fetch recent signal decisions from SQLite.

**Response:**
```json
[
  {
    "timestamp": "2026-03-27T10:30:45.123456Z",
    "step": 1250,
    "phase": "north_south",
    "duration_s": 28.5,
    "reasoning": "High PCE density in NS corridor (45.2), moderate EW (22.1)",
    "emergency_active": false,
    "llm_used": true,
    "llm_model": "gemini-2.0-flash",
    "rule_based_fallback": false
  },
  ...
]
```

### WebSocket Events

Socket.IO events for real-time dashboard streaming. Connect to `http://localhost:8000` with Socket.IO client.

#### Broadcast Events

Client auto-receives (no emit required):

| Event | Payload | Frequency |
|-------|---------|-----------|
| `frame` | `{image: "base64_jpeg", step: int}` | ~10 FPS (configurable) |
| `decision` | `{phase: str, duration: float, reasoning: str, step: int}` | Per decision |
| `zones` | `{north: {pce, count}, south: {...}, ...}` | ~10 FPS |
| `kpi` | `{total_detected, avg_pce, ambulance_events, ...}` | ~1 FPS |
| `emergency` | `{ambulance_id, destination, route, status, step}` | Per event |

#### Server-Bound Events

Emit from client to control:

| Event | Args | Response |
|-------|------|----------|
| `control_action` | `{action: "start"\|"pause"\|"resume"\|"stop"}` | Status dict |
| `trigger_emergency` | `{ambulance_id, destination, origin_direction}` | Queued event |

### Example: Python Socket.IO Client

```python
import socketio
import asyncio

sio = socketio.AsyncClient()

@sio.event
async def connect():
    print("Connected to backend")

@sio.on("frame")
async def on_frame(data):
    print(f"Frame received: step {data['step']}")

@sio.on("decision")
async def on_decision(data):
    print(f"Decision: {data['phase']} for {data['duration']}s")

@sio.on("emergency")
async def on_emergency(data):
    print(f"Emergency: {data['ambulance_id']} → {data['destination']}")

async def main():
    await sio.connect("http://localhost:8000")
    await sio.wait()

asyncio.run(main())
```

---

## Module Architecture

### `agent/` — Decision Making Engine

Intelligent signal control using LLM reasoning or rule-based fallback.

| Module | Responsibility |
|--------|-----------------|
| `models.py` | Pydantic data models: `IntersectionState`, `ZoneState`, `SignalDecision` |
| `llm_agent.py` | Gemini multimodal wrapper; takes text + image → returns structured decision |
| `rule_based.py` | Fallback rule-based decisioning (no LLM) using PCE density heuristics |
| `state_builder.py` | Constructs text description of current traffic state for LLM prompt |
| `prompts.py` | System prompt with India-specific traffic rules and PCE standards |
| `safety.py` | Validates decisions against safety constraints (min/max green time, phase validity) |
| `phase_utils.py` | Traffic light phase mapping; starvation prevention through phase rotation |
| `signal_controller.py` | Applies `SignalDecision` to MetaDrive traffic light API |

### `cv/` — Computer Vision Pipeline

YOLOv8n-based vehicle detection and PCE weighting.

| Module | Responsibility |
|--------|-----------------|
| `detector.py` | YOLOv8n inference on frames; zone-based vehicle assignment; ambulance red-heuristic detection |
| `zone_config.py` | Defines N/S/E/W quadrant polygon zones; serves as supervision `PolygonZone` instances |
| `pce_calculator.py` | Computes per-zone PCE totals using IRC/MORTH weights (car=1.0, bus=2.5, motorcycle=0.5, auto=1.2) |
| `annotator.py` | Renders bounding boxes, zone polygons, PCE values onto frames for visualization |

### `api/` — REST/WebSocket Backend

FastAPI server for simulation control and real-time event streaming.

| Module | Responsibility |
|--------|-----------------|
| `app.py` | FastAPI + Socket.IO app; defines all REST endpoints and WebSocket event handlers |
| `broadcaster.py` | Asynchronous event broadcaster; throttles frame/decision/kpi events to reduce bandwidth |

### `emergency/` — Green Corridor System

Ambulance detection and automatic emergency route calculation.

| Module | Responsibility |
|--------|-----------------|
| `event_handler.py` | Debounces ambulance detections; queues manual emergency triggers via API |
| `corridor.py` | Manages corridor lifecycle: activate → hold green on origin direction → maintain route → deactivate |

### `envs/` — Simulation Environment

MetaDrive environment creation and traffic light control.

| Module | Responsibility |
|--------|-----------------|
| `intersection_env.py` | Creates MetaDrive `PGDrive` environment; introspects traffic light API; captures top-down frames |
| `ambulance_spawner.py` | Injects ambulance vehicle into MetaDrive from origin direction |

### `routing/` — Route Optimization

Dijkstra routing on Delhi road graph.

| Module | Responsibility |
|--------|-----------------|
| `router.py` | Route computation; pre-defined routes (AIIMS, GTB_HOSPITAL, SAFDARJUNG); route-to-JSON formatting |
| `delhi_graph.py` | Downloads/caches OpenStreetMap road graph for Delhi using osmnx and networkx |

### `data/` — Persistence & Logging

SQLite-based decision logging and KPI aggregation.

| Module | Responsibility |
|--------|-----------------|
| `logger.py` | Logs signal decisions to SQLite; computes KPI summaries; fetches recent decision history |

---

## Computer Vision Pipeline

### Vehicle Detection

**Model**: YOLOv8n (Nano — ~3.2M parameters, fast inference)

**Classes Detected**:
- Motorcycle (YOLO class 3) → PCE 0.5
- Car (YOLO class 2) → PCE 1.0
- Bus (YOLO class 5) → PCE 2.5
- Truck (YOLO class 7) → PCE 2.5

**Inference Rate**: Configurable (default: every 3 simulation steps)

**Frame Source**: MetaDrive top-down rendering (800×800 pixels, birds-eye view)

### Zone Assignment

Frame divided into 4 zones using `supervision.PolygonZone`:

```
        North Zone
      ┌─────────────┐
      │             │
 West │   Intersection   │ East
Zone  │             │ Zone
      │             │
      └─────────────┘
        South Zone
```

**Zone Polygons**: Defined in `cv/zone_config.py` as lat/lon quadrants or pixel coordinates. Typically:
- **North**: Upper 50% of frame, full width
- **South**: Lower 50% of frame, full width
- **East**: Right 50% of frame, full height
- **West**: Left 50% of frame, full height

### PCE Calculation

**Per-Zone Formula**:

$$
\text{zone\_PCE} = \sum_{vehicle} \text{PCE}_{class}
$$

**Green Time Allocation** (adaptive):

$$
\text{green\_s} = \text{clamp}(15 + 1.5 \times \text{zone\_PCE}, 15, 60)
$$

Where:
- Base green time: 15 seconds (urban standard)
- PCE multiplier: 1.5 seconds per PCE unit
- Min green: 15 seconds
- Max green: 60 seconds

**Two-Wheeler Bonus**: If zone PCE is >30% motorcycle/two-wheeler ratio, add 5 seconds (safety for vulnerable users).

### Ambulance Detection

**Method**: HSV color space masking for red siren reflections

1. Capture frame contours
2. Filter for red hue (HSV H: 0–10 or 170–180)
3. Check contour size and aspect ratio
4. Debounce over 5-frame window (noise reduction)
5. If confirmed: trigger emergency corridor

**Fallback**: Manual trigger via `/api/emergency` endpoint

---

## LLM Agent Details

### Model & Provider

**Provider**: Google Gemini (free tier)

**Active Model**: `gemini-2.0-flash` (or `gemini-2.5-flash`)

**Context Window**: 10,000–100,000 tokens (sufficient for state + image)

**Cost**: Free tier allows 1,500 requests/day + 1M tokens/day. Single simulation typically uses <10% of daily quota.

### Multimodal Input

**Text Component** (structured state):
```json
{
  "junction_id": "J0",
  "step": 1250,
  "zones": {
    "north": {"pce_total": 18.5, "vehicle_count": 12, "avg_speed_kmh": 15},
    "south": {"pce_total": 22.1, "vehicle_count": 15, "avg_speed_kmh": 12},
    "east": {"pce_total": 14.3, "vehicle_count": 9, "avg_speed_kmh": 18},
    "west": {"pce_total": 16.2, "vehicle_count": 11, "avg_speed_kmh": 16}
  },
  "emergency_active": false,
  "last_phase": "north_south",
  "last_duration_s": 25
}
```

**Image Component**: Base64-encoded JPEG frame (800×800) with annotated zone boundaries and vehicle bboxes.

### Decision Output

**Structured Response** (`SignalDecision` Pydantic model):

```python
class SignalDecision(BaseModel):
    phase: Literal["north_south", "east_west"]  # Which lights go green
    duration_s: float  # How long (15–60 seconds)
    reasoning: str     # Explanation for auditing
    emergency_active: bool
    llm_model: str
    timestamp: str
```

### Prompting Strategy

**System Prompt** (in `agent/prompts.py`):
- India traffic rules context (RWH, PCE standards)
- Safety constraints (min 15s green, max 60s, max-wait rules)
- Optimization goal (minimize total vehicle wait time)
- Emergency priority (ambulance corridors override normal phasing)

**User Prompt** (dynamic):
- Current state JSON
- Annotated frame image
- Last decision and duration
- Emergency status

**Output Guarantee**: Instructed to return valid JSON only (via `instructor` library), preventing malformed responses.

### Fallback & Error Handling

If Gemini call fails:
1. Logs error
2. Falls back to rule-based agent
3. Signal decision still applied (no system outage)
4. Flag recorded in decision log for audit

**Rule-Based Fallback**:
```python
# Simplified pseudocode
if emergency_active:
    grant green to ambulance direction
else:
    grant green to highest-PCE zone
    duration = base_duration + 1.5 * max_pce
```

---

## Emergency Corridor System

### Detection Pipeline

```
Frame → HSV Red Mask → Contour Analysis → Debounce → Ambulance Event
                        ↓
                   Size/Aspect Ratio
                   Check (siren-like?)
```

**Debounce Window**: 5 frames (reduces false positives from flickering lights)

**Manual Override**: API endpoint `POST /api/emergency` bypasses detection, useful for:
- Testing
- External dispatch integration
- Non-CCTV ambulance notification

### Corridor Lifecycle

#### 1. **Activate** (T=0s)
- Ambulance detected or manual trigger received
- Route computed via Dijkstra on Delhi OSM graph
- Event queued in internal event handler

#### 2. **Route to Origin** (T=0–2s)
- Path from ambulance current position to intersection
- Ensures ambulance is approaching origin direction

#### 3. **Sustain Green** (T=2–30s)
- Origin direction signal held green (duration configurable, default 30s)
- LLM agent receives `emergency_active=true`
- Phase locked to origin direction
- Bypass normal starvation prevention

#### 4. **Release & Deactivate** (T=30s+)
- Corridor duration expires
- Signal returns to normal phasing
- Decision log flagged with `emergency_active=true`

### Route Computation

**Input**: Ambulance destination (e.g., "AIIMS", "GTB_HOSPITAL")

**Process**:
1. Get current ambulance position (entry direction)
2. Geocode destination hospital
3. Run Dijkstra shortest-path on Delhi OSM graph
4. Return waypoint sequence (lat/lon pairs)

**Output**: Route array:
```json
[
  {"lat": 28.5662, "lon": 77.2125},  // Origin intersection
  {"lat": 28.5670, "lon": 77.2130},  // Next waypoint
  ...
  {"lat": 28.5771, "lon": 77.2245}   // Destination (hospital)
]
```

### Integration with LLM Decision

When `emergency_active=true`:
- LLM receives emergency context in state JSON
- Visual frame shows ambulance bounding box (if detected)
- Prompt requests: "Grant green to [origin_direction] immediately"
- Safety checks may allow temporary violation of max-green (emergency override)

---

## Testing

### Test Framework

**Framework**: pytest

**Test Files**:

| File | Coverage |
|------|----------|
| `tests/test_env_mode_guards.py` | Environment mode validation (single vs. multi-agent) |
| `tests/test_random_graph_env_parsing.py` | Random road graph environment parsing and introspection |
| `test_instructor_api.py` (root) | Pre-implementation verification of Gemini + instructor integration |

### Running Tests

```powershell
# Run all tests
pytest

# Run specific test file
pytest tests/test_env_mode_guards.py -v

# Run with coverage
pytest --cov=. --cov-report=html
```

### Integration Testing

**Manual end-to-end**:

```powershell
# Terminal 1: Start backend
uvicorn api.app:socket_app --port 8000

# Terminal 2: Test scenarios
python -c "
import requests
import time

base = 'http://localhost:8000'

# Start sim
requests.post(f'{base}/api/control', json={'action': 'start'})
time.sleep(5)

# Trigger emergency
requests.post(f'{base}/api/emergency', json={
    'ambulance_id': 'AMB-TEST-001',
    'destination': 'AIIMS',
    'origin_direction': 'north'
})
time.sleep(10)

# Stop
requests.post(f'{base}/api/control', json={'action': 'stop'})

# Check decisions
decisions = requests.get(f'{base}/api/decisions?limit=5').json()
print(f'Logged {len(decisions)} decisions')
"
```

### Debugging Scripts

**Inspect Traffic Lights** (`inspect_traffic_lights.py`):
```powershell
python inspect_traffic_lights.py
# Prints every traffic light status every 2 seconds
```

**Test Gemini API** (`test_instructor_api.py`):
```powershell
python test_instructor_api.py
# Verifies GEMINI_API_KEY, instructor, Pydantic integration
```

**List Models** (`list_models.py`):
```powershell
python list_models.py
# Shows available Gemini models
```

---

## Project Status

### Stages & Completion

| Stage | Task | Status | Notes |
|-------|------|--------|-------|
| **T01** | Environment Setup | ✅ Complete | MetaDrive 0.4.3; single + multi-agent modes |
| **T02** | CV Pipeline | ✅ Complete | YOLOv8n; PCE weighting; zone detection |
| **T03** | LLM Agent | ✅ Complete | Gemini multimodal; instructor validation; rule-based fallback |
| **T04** | Emergency Corridor | ✅ Complete | Ambulance detection; route computation; corridor lifecycle |
| **T05** | Backend API | ✅ Complete | FastAPI; Socket.IO; control endpoints; decision logging |
| **T06** | Dashboard UI | 🔄 Deferred | (Post-MVP) Web frontend for monitoring |
| **T07** | Cloud Deployment | 🔄 Deferred | (Post-MVP) AWS/GCP scaling |
| **T08** | Hardening & Docs | 🔄 In Progress | (Currently updating docs and validation) |

### Current State

- **Core MVP**: Fully functional simulation with all major features
- **API Backend**: Running and tested with Uvicorn
- **Decision Logging**: SQLite persistence active
- **Gemini Integration**: Ready (requires free API key)
- **Testing**: Basic test suite; can be expanded
- **Documentation**: README complete; code comments inline

### Known Limitations

- **Simulator Only**: No real traffic data; uses MetaDrive simulation
- **Single Intersection**: Designed for one J0 intersection; multi-intersection extension possible
- **Day Mode**: No night-time lighting effects
- **Static Routes**: Pre-defined ambulance destinations; dynamic routing possible
- **Frame Rate**: 10 FPS default (can increase with more VRAM)

---

## Troubleshooting

### Common Issues

#### 1. `ModuleNotFoundError: No module named 'metadrive'`

**Cause**: Virtual environment not activated or metadrive not installed.

**Solution**:
```powershell
.\.venv\Scripts\Activate.ps1  # Activate venv
pip install metadrive-simulator==0.4.3
```

#### 2. `GEMINI_API_KEY not found`

**Cause**: `.env` file missing or key not set.

**Solution**:
1. Create `.env` in project root
2. Add: `GEMINI_API_KEY=your_actual_key_here`
3. Get key from https://aistudio.google.com

#### 3. Backend fails with `Address already in use :8000`

**Cause**: Port 8000 already occupied (another service or stale process).

**Solution**:
```powershell
# Kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or use different port
uvicorn api.app:socket_app --port 8001
```

#### 4. YOLOv8n inference very slow (>2 sec/frame)

**Cause**: Running on CPU; low VRAM; low-end GPU.

**Solution**:
```env
# In .env
TORCH_DEVICE=cuda  # Use GPU
YOLO_HALF_PRECISION=1  # FP16 for faster inference
YOLO_INFERENCE_RATE=5  # Reduce inference frequency
```

Or downgrade to YOLOv8s:
```python
# In cv/detector.py
model = YOLO("yolov8s.pt")  # Smaller model
```

#### 5. Frame encoding errors / corrupted WebSocket frames

**Cause**: Frame format mismatch or incomplete JPEG encoding.

**Solution**:
```env
# In .env
FRAME_ENCODE_QUALITY=75  # Lower quality (smaller file)
BROADCAST_FPS_CAP=5  # Reduce broadcast rate
```

#### 6. `OSError: [Errno 10054] Connection reset by peer` (Windows)

**Cause**: Socket timeout or network glitch during WebSocket streaming.

**Solution**:
```python
# In api/app.py, increase timeout
uvicorn api.app:socket_app --port 8000 --timeout-keep-alive 300
```

#### 7. MetaDrive gets stuck / doesn't render

**Cause**: Graphics driver issue or display server not found (Linux/headless).

**Solution**:
```python
# Edit envs/intersection_env.py
config['image_observation'] = True  # Use frame buffer instead of display
config['use_render'] = False  # Headless mode
```

### Getting Help

1. **Check logs**: Backend logs printed to console; check for errors or warnings
2. **Inspect status**: `curl http://localhost:8000/api/status`
3. **Review decisions**: `curl http://localhost:8000/api/decisions?limit=10` to see decision history
4. **Test components independently**:
   - `python test_instructor_api.py` — Gemini API
   - `python inspect_traffic_lights.py` — Traffic light control
   - `python list_models.py` — Model availability
5. **Enable debug logging**: Set env vars for verbose output (if implemented)

---

## Contributing

### Code Style

- **Language**: Python 3.11+
- **Formatter**: Black (if available)
- **Linter**: Pylint / Flake8
- **Docstrings**: Google-style with type hints

Example:
```python
def compute_zone_pce(detections: list[Detection]) -> dict[str, float]:
    """Compute PCE for each zone.
    
    Args:
        detections: List of vehicle detections with YOLO class IDs.
        
    Returns:
        Dict mapping zone names to total PCE values.
    """
    zone_pce = {"north": 0.0, "south": 0.0, "east": 0.0, "west": 0.0}
    for det in detections:
        zone = assign_to_zone(det)
        zone_pce[zone] += get_pce_weight(det.cls)
    return zone_pce
```

### Adding New Features

1. **Create branch**: `git checkout -b feature/your-feature`
2. **Implement**: Follow existing module structure
3. **Test**: Add tests in `tests/` directory
4. **Document**: Update docstrings and README if needed
5. **Submit PR**: Link to issue; describe changes

### Suggested Improvements

- [ ] Multi-intersection routing and coordination
- [ ] Real-time weather integration (rain → slower vehicles)
- [ ] Machine learning phase optimization (instead of rule-based)
- [ ] Dashboard UI (React/Vue web frontend)
- [ ] Real traffic data integration (CCTV feeds, sensors)
- [ ] Speech-to-speech integration for emergency dispatch
- [ ] Night-mode rendering with proper lighting
- [ ] Mobile app for traffic monitoring

---

## Future Roadmap

### Phase 2 (Post-Competition)

- **Dashboard UI**: Web-based monitoring (React + D3.js charts)
- **Multi-Intersection Coordination**: Route ambulances across multiple junctions
- **Real Data Integration**: Cloud SDK for live CCTV streams
- **Model Training**: Custom YOLOv8 fine-tune on Indian traffic
- **Cloud Deployment**: AWS/GCP scalability for real cities

### Phase 3 (Beyond)

- **Autonomous Vehicle Integration**: Signal planning for mixed human/autonomous traffic
- **Seasonal Tuning**: Adapt PCE weights for monsoon/festival conditions
- **Accessibility Features**: Pedestrian signal optimization; blind audio cues
- **Carbon Emission Tracking**: Measure emissions reduction from better signal timing

---

## License & Attribution

### License

This project is part of the **India Innovates 2026** competition. Code developed for demonstration and research purposes.

### Attribution & Credits

- **Simulation**: MetaDrive simulator by Quanyi Zhou et al. (ICLR 2021)
- **Computer Vision**: YOLOv8n by Ultralytics
- **LLM**: Google Gemini API
- **Routing**: OpenStreetMap (OSM) + networkx
- **Traffic Standards**: Indian Road Congress (IRC) PCE guidelines (MORTH)
- **Competition**: India Innovates 2026 organizing committee

### Third-Party Licenses

All dependencies installed via `pip install -r requirements.txt` are subject to their respective licenses:
- MetaDrive: BSD 3-Clause
- Ultralytics: AGPL-3.0
- FastAPI: MIT
- NetworkX: BSD
- OpenStreetMap: ODbL

See individual package READMEs for full license text.

### Citation

If you use this project in research or publication:

```bibtex
@misc{neurosignal2026,
  title={NeuroSignal India: AI-Powered Dynamic Traffic Management},
  author={[Team Members]},
  year={2026},
  publisher={India Innovates},
  note={Competition Project}
}
```

---

## Contact & Support

- **Issues**: Report bugs or feature requests in project issue tracker
- **Questions**: Check FAQ in this README or GitHub Discussions
- **Contributions**: See [Contributing](#contributing) section

---

**Last Updated**: March 27, 2026  
**Status**: MVP Complete ✅

---

*Happy Traffic Management! 🚦 Let the LLM decide.* 🤖
