---
title: OpenEnv Self-Driving Collision Avoidance
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
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

- `API_BASE_URL`: endpoint base URL for chat completions (default: `https://api.openai.com/v1`).
- `MODEL_NAME`: model/deployment name (default: `gpt-4o`).
- `HF_TOKEN` or `OPENAI_API_KEY`: API key. Falls back to deterministic heuristic policy if unset.

Example:

```bash
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4o
set OPENAI_API_KEY=your_key_here
python inference.py
```

### Stdout Format

The script emits structured log lines per the OpenEnv spec:

```
[START] task=easy_open_road env=openenv-selfdriving-collision-avoidance model=gpt-4o
[STEP] step=1 action=accelerate reward=0.12 done=false error=null
[STEP] step=2 action=maintain reward=0.08 done=false error=null
...
[END] success=true steps=15 score=0.85 rewards=0.12,0.08,...
```

One `[START]`/`[END]` block per task (3 total). After all tasks, a JSON summary with per-task scores and `mean_score` is printed.

### Baseline Scores (deterministic fallback policy, seed=42)

| Task | Score | Reached Goal | Collisions | Unsafe Events |
|------|-------|--------------|------------|---------------|
| easy_open_road | 0.77 | Yes | 0 | 0 |
| medium_lane_change | 0.75 | Yes | 0 | 1 |
| hard_dense_merge | 0.67 | Yes | 0 | 3 |

**Mean score: 0.73**

The deterministic fallback policy uses the environment's built-in `recommended_action` heuristic. An LLM-based agent should be able to score higher by planning multi-step lane changes and anticipating dynamic hazards in the hard task.

## Hugging Face Spaces Deployment Notes

1. Ensure `openenv.yaml`, `Dockerfile`, and this `README.md` are in repo root.
2. Push repository to a Docker Space.
3. Space should expose FastAPI app on port `8000`.
4. Set secret variables in Space settings if running remote LLM inference:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN` or `OPENAI_API_KEY`
