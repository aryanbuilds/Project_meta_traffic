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

