# Run and Installation Guide (Windows)

## Scope

This document covers local setup, backend startup, API smoke tests, and common runtime fixes.

## Requirements

- Windows 10/11
- Python 3.11+
- PowerShell
- Internet connection for first-time model and asset downloads

## Setup

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r .\requirements.txt -c .\constraints\windows-cpu.txt --index-url https://download.pytorch.org/whl/cpu
```

Fallback if the CPU wheel index is unavailable:

```powershell
uv pip install -r .\requirements.txt -c .\constraints\windows-cpu.txt
```

If script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Preflight Checks

```powershell
python -c "import cv2, fastapi, socketio, uvicorn; print('imports-ok')"
python -c "import torch, torchvision, ultralytics; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'ultralytics', ultralytics.__version__)"
python -c "from api.app import socket_app; print('app-import-ok')"
```

Optional routing check:

```powershell
python routing/delhi_graph.py
```

## Start Backend

```powershell
uvicorn api.app:socket_app --port 8000
```

First run may download MetaDrive assets and yolov8n model weights.

## Azure OpenAI Configuration

Set these variables in `.env` before launching backend:

```env
AZURE_OPENAI_ENDPOINT=https://<resource-or-project>.openai.azure.com
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

Notes:

- `AZURE_OPENAI_DEPLOYMENT` must match the deployment name configured in Azure AI Foundry.
- If Azure variables are missing or requests fail, backend safely falls back to rule-based control.

## Performance and Top-View Tuning

Set these in `.env` for better runtime speed and proper 2D window fit:

```env
YOLO_INFERENCE_RATE=3
PERCEPTION_FRAME_SOURCE=rgb_camera
PERCEPTION_SENSOR_NAME=rgb_camera
PERCEPTION_WIDTH=640
PERCEPTION_HEIGHT=360
USE_POLYGON_ZONES=0
PERCEPTION_FRAME_SOURCE=topdown
PERCEPTION_SENSOR_NAME=rgb_camera
PERCEPTION_WIDTH=640
PERCEPTION_HEIGHT=360
FRAME_SIZE=800
TOPDOWN_AUTO_FIT=1
TOPDOWN_MAX_SCREEN_RATIO=0.65
TOPDOWN_WIDTH=800
TOPDOWN_HEIGHT=800
TOPDOWN_WINDOW=1
TOPDOWN_FILM_SCALE=1.0
TOPDOWN_SCALING=3.0
TOPDOWN_CAMERA_MODE=map_center
DRIVER_STEER=0.0
DRIVER_THROTTLE=0.35
AGENT_POLICY_MODE=idm
SIM_ENV_MODE=multi_agent
TRAFFIC_DENSITY=0.2
VEHICLE_ENABLE_REVERSE=0
AMBULANCE_OBJ_PATH=Models_G0403A078/ambulance.obj
AMBULANCE_MODEL_SCALE=0.7
AMBULANCE_MODEL_Z=0.55
CRASH_VEHICLE_DONE=0
CRASH_OBJECT_DONE=1
OUT_OF_ROAD_DONE=1
MULTI_AGENT_COUNT=12
MULTI_ALLOW_RESPAWN=1
MULTI_CRASH_DONE=0
MULTI_DELAY_DONE=25
INTERSECTION_EXIT_LENGTH=100
INTERSECTION_LANE_NUM=2
TRAFFIC_LIGHT_CYCLE_STEPS=140
RANDOM_GRAPH_NODES=120
RANDOM_GRAPH_EDGES=320
RANDOM_GRAPH_LIGHT_PROB=0.25
PIPELINE_DEBUG=1
SIGNAL_DEBUG=1
```

Guidance:

- Increase `YOLO_INFERENCE_RATE` (for example `4` or `5`) to improve FPS.
- Lower `FRAME_SIZE` / `TOPDOWN_WIDTH` / `TOPDOWN_HEIGHT` (for example `640`) for better speed.
- Keep `TOPDOWN_AUTO_FIT=1` to prevent the pygame top-view window from going off-screen.
- Set `TOPDOWN_WINDOW=0` for headless top-down rendering when visual window is not required.
- Set `AGENT_POLICY_MODE=idm` for MetaDrive built-in autopilot behavior.
- Set `CRASH_VEHICLE_DONE=0` to avoid immediate episode termination on vehicle collisions during demo loops.
- Set `SIM_ENV_MODE=multi_agent` to enable `MultiAgentIntersectionEnv` and control all active agents per step.
- The runner now applies lidar-aware adaptive throttle in manual mode to reduce collision likelihood.
- Set `PIPELINE_DEBUG=1` to print stage-by-stage checkpoints (YOLO counts, LLM controller type/latency, and traffic-light inventory after reset).
- Use `PERCEPTION_FRAME_SOURCE=topdown` as the stable default so YOLO zones and PCE quadrants match the top-view geometry.
- Switch to `PERCEPTION_FRAME_SOURCE=rgb_camera` only when you intentionally recalibrate zones for perspective frames.
- Set `SIGNAL_DEBUG=1` to print per-light direction-target mapping and applied states.

Deterministic signal mapping override (optional):

- If automatic traffic-light direction inference is unstable on your MetaDrive build, set `SIGNAL_LIGHT_DIRECTION_MAP` in `.env`.
- Example: `SIGNAL_LIGHT_DIRECTION_MAP={"north_light":"north","south_light":"south","east_light":"east","west_light":"west"}`

## Environment Mode Selection

- Use `MetaDriveEnv` by setting `SIM_ENV_MODE=single`.
- Use `MultiAgentIntersectionEnv` by setting `SIM_ENV_MODE=multi_agent`.
- Current backend supports both modes and resets episodes when `terminated["__all__"]` or `truncated["__all__"]` is reached in multi-agent mode.

## API Smoke Tests

Run in a second PowerShell terminal:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/kpi" | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/decisions?limit=5" | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/control" -ContentType "application/json" -Body '{"action":"pause"}' | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/control" -ContentType "application/json" -Body '{"action":"resume"}' | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/emergency" -ContentType "application/json" -Body '{"ambulance_id":"AMB_QUICK_01","destination":"AIIMS","origin_direction":"north"}' | ConvertTo-Json -Depth 5
```

## Deterministic FIX-4/5 Verification

Run deterministic runtime verification (fixed seed/profile, 3 trials + 50-step checks):

```powershell
python tools/verify_fix45.py
```

Expected artifact:

- `data/test_frame.png`

## Expected Results

- `/health` returns `status: ok`, `running: true`, and a changing `step` value.
- `/api/kpi` returns aggregate metrics.
- `/api/decisions` returns recent persisted decision rows.
- `/api/control` toggles `paused` correctly.
- `/api/emergency` returns `queued: true` with route details.

## Common Issues

Missing cv2:

```powershell
uv pip install opencv-python
```

Missing socketio module:

```powershell
uv pip install python-socketio
```

`netifaces` or Visual C++ build error during install:

- Ensure `requirements.txt` contains `python-socketio`.
- Ensure `requirements.txt` does not contain `socketio`.

PowerShell parsing error when calling Python executable path:

```powershell
& ".\.venv\Scripts\python.exe" -c "print('ok')"
```

## Security

If the Azure OpenAI API key has been exposed in logs, rotate it immediately and update `.env`.

Validate the rotated key quickly:

```powershell
python -c "from agent.llm_agent import DEFAULT_MODEL; print('llm-model', DEFAULT_MODEL)"
```



