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
uv pip install -r .\requirements.txt
```

If script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Preflight Checks

```powershell
python -c "import cv2, fastapi, socketio, uvicorn; print('imports-ok')"
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

## Performance and Top-View Tuning

Set these in `.env` for better runtime speed and proper 2D window fit:

```env
YOLO_INFERENCE_RATE=3
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

If the Gemini API key has been exposed in logs, rotate it immediately and update `.env`.

