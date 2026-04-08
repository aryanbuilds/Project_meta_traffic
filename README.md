---
title: OpenEnv Self-Driving Collision Avoidance
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# OpenEnv Self-Driving Collision Avoidance

A bare-minimum OpenEnv-style environment for Round-1 self-driving collision avoidance.

## What Is Implemented

- Self-driving Agent RL Training (`train_rl.py`) with PPO algorithm.
  - Interactive top-down 2D evaluation.
  - Native 3D Simulation Rendering.
- Deterministic `reset`, `step`, and `state` environment loop.
- Typed models for action, observation, state, reward, tasks, and grading.
- Three benchmark tasks:
  - `easy_open_road`
  - `medium_lane_change`
  - `hard_dense_merge`
- Deterministic episode grader returning normalized score in `[0.0, 1.0]`.
- FastAPI server endpoints:
  - `GET /health`
  - `GET /tasks`
  - `POST /reset`
  - `POST /step`
  - `GET /state`
  - `GET /grade`
- Baseline `inference.py` using OpenAI-compatible API client.

## Directory Layout

```
openenv_selfdriving/
  __init__.py
  models.py
  tasks.py
  graders.py
  environment.py
  client.py
  server/
    __init__.py
    app.py
openenv.yaml
Dockerfile
inference.py
```

## Action and Observation Contract

- Action values:
  - `accelerate`
  - `brake`
  - `lane_left`
  - `lane_right`
  - `maintain`
- Key observation fields:
  - `ego_lane`, `ego_position_m`, `ego_speed_mps`
  - `distance_to_goal_m`
  - `nearest_ahead_by_lane_m`
  - `collision_risk`
  - `recommended_action`

## Local Run

Install dependencies:

```bash
pip install -r openenv_requirements.txt
```

Start server:

```bash
uvicorn openenv_selfdriving.server.app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Docker Run

Build image:

```bash
docker build -t openenv-selfdriving:latest .
```

Run container:

```bash
docker run --rm -p 8000:8000 openenv-selfdriving:latest
```

## Baseline Inference and Scoring

`inference.py` uses an OpenAI-compatible client and these environment variables:

- `API_BASE_URL`: endpoint base URL for chat completions.
- `MODEL_NAME`: model/deployment name.
- `HF_TOKEN` or `OPENAI_API_KEY`: API key.

Example:

```bash
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4o
set OPENAI_API_KEY=your_key_here
python inference.py
```

The script prints JSON summary including per-task scores and `mean_score`.

## Hugging Face Spaces Deployment Notes

1. Ensure `openenv.yaml`, `Dockerfile`, and this `README.md` are in repo root.
2. Push repository to a Docker Space.
3. Space should expose FastAPI app on port `8000`.
4. Set secret variables in Space settings if running remote LLM inference:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN` or `OPENAI_API_KEY`
