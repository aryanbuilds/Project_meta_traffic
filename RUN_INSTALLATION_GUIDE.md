# NeuroSignal India - Run and Installation Guide (Windows)

This guide is optimized for quick local startup and API smoke testing.

## 1. Prerequisites

- Windows 10/11
- Python 3.11+ (project tested with venv Python 3.11.14)
- PowerShell
- Internet access on first run (MetaDrive assets and YOLO weights may download)

## 2. Create and Activate Virtual Environment

From project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

Use uv (fastest):

```powershell
uv pip install -r .\requirements.txt
```

Alternative with pip:

```powershell
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

## 4. Quick Import Validation

```powershell
python -c "import cv2, fastapi, socketio, uvicorn; print('imports-ok')"
python -c "from api.app import socket_app; print('app-import-ok')"
```

Expected:

- imports-ok
- app-import-ok

Note: You may see a pygame pkg_resources deprecation warning. It is non-blocking for this MVP run.

## 5. Optional Routing Smoke Test

Random graph test mode is enabled in routing/delhi_graph.py.

```powershell
python routing/delhi_graph.py
```

Expected output includes:

- Random test graph: <nodes> nodes, <edges> edges
- A sample route dictionary with non-empty route and route_length

## 6. Start Backend Server

```powershell
uvicorn api.app:socket_app --port 8000
```

On first run you may see:

- MetaDrive assets download
- YOLO model download (yolov8n.pt)

Wait until:

- Application startup complete
- Uvicorn running on http://127.0.0.1:8000

## 7. API Smoke Test Commands

Open a second PowerShell terminal in project root.

### Health

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" | ConvertTo-Json -Depth 5
```

### KPI

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/kpi" | ConvertTo-Json -Depth 5
```

### Recent Decisions

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/decisions?limit=5" | ConvertTo-Json -Depth 5
```

### Pause Simulation

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/control" -ContentType "application/json" -Body '{"action":"pause"}' | ConvertTo-Json -Depth 5
```

### Resume Simulation

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/control" -ContentType "application/json" -Body '{"action":"resume"}' | ConvertTo-Json -Depth 5
```

### Trigger Emergency

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/emergency" -ContentType "application/json" -Body '{"ambulance_id":"AMB_QUICK_01","destination":"AIIMS","origin_direction":"north"}' | ConvertTo-Json -Depth 5
```

## 8. What Success Looks Like

- /health returns:
  - status: ok
  - running: true
  - step increasing over time
- /api/kpi returns aggregate values (decisions, latency, fallback_rate, etc.)
- /api/decisions returns rows from SQLite
- /api/control pause/resume toggles paused state
- /api/emergency returns queued true with route and nonce

## 9. Important Runtime Notes

- Default mode currently favors rule-based fallback when LLM calls fail or are slow.
- If fallback_rate in /api/kpi is high (near 1.0), check Gemini key, quota, and network.
- Data persists in SQLite at data/decisions.db.
- Current routing mode is random-graph local testing (not live Delhi OSM fetch).

## 10. Common Issues and Fixes

### ModuleNotFoundError: cv2

```powershell
uv pip install opencv-python
```

### ModuleNotFoundError: socketio

Install python-socketio (not socketio):

```powershell
uv pip install python-socketio
```

### netifaces / Visual C++ build error during install

Cause: wrong dependency socketio package.
Fix: ensure requirements includes python-socketio and does not include socketio.

### PowerShell command parsing errors when calling Python path directly

Use call operator:

```powershell
& ".\.venv\Scripts\python.exe" -c "print('ok')"
```

## 11. Security Note

Rotate and replace GEMINI_API_KEY if it has been exposed in logs or screenshots.

## 12. Fast Daily Run Sequence

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.app:socket_app --port 8000
```

Then in another terminal:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/kpi" | ConvertTo-Json -Depth 5
```
