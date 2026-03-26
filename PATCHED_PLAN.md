# Patched Plan (2026-03-26)

## What was patched

1. LLM integration API references updated to `instructor.from_provider("google/<model>")`.
2. Default model references updated to `gemini-2.5-flash`.
3. Auto-rickshaw PCE references corrected to `1.2`.
4. MetaDrive signal control API marked as runtime-validated (do not assume helper method exists).
5. TrafficDojo repository reference flagged as unverified for Phase 2.

## Why this patch was required

- The plan contained outdated Instructor integration calls (`from_gemini`) and older model names.
- Signal control calls in MetaDrive are version-sensitive and should be verified against runtime objects.
- The PCE constant mismatch would bias downstream decisions and benchmark claims.

## Execution status after patch

- Documentation patched: complete.
- Implementation started: bootstrap code creation for T01/T03 now in progress.

## Immediate next coding tasks

1. Create `config/india_vehicles.py` canonical constants.
2. Create `agent/models.py` and `agent/prompts.py` with patched rules.
3. Create a runtime-safe Gemini client wrapper using `instructor.from_provider`.
4. Create initial MetaDrive environment bootstrap with top-down frame save.
5. Add quick verification script to introspect traffic-light APIs in the running environment.

## Work completed in this run

- Added config constants: `config/india_vehicles.py`
- Added agent core:
  - `agent/models.py`
  - `agent/prompts.py`
  - `agent/state_builder.py`
  - `agent/rule_based.py`
  - `agent/safety.py`
  - `agent/llm_agent.py`
- Added simulation bootstrap:
  - `envs/intersection_env.py`
  - `inspect_traffic_lights.py`
  - `main.py`
- Added CV scaffold:
  - `cv/zone_config.py`
  - `cv/pce_calculator.py`
  - `cv/detector.py`
  - `cv/annotator.py`

## Known runtime blocker in this environment

- Python runtime is not available in this sandbox, so compile/runtime tests could not be executed here.

## Additional completion (T04 + hardening)

- Added emergency routing layer:
  - `routing/delhi_graph.py`
  - `routing/router.py`
- Added emergency event/corridor pipeline:
  - `emergency/event_handler.py`
  - `emergency/corridor.py`
- Added signal + spawn + logging support:
  - `agent/phase_utils.py`
  - `agent/signal_controller.py`
  - `envs/ambulance_spawner.py`
  - `data/logger.py`
- Hardened existing agent modules:
  - stricter Pydantic model configs
  - normalized phase mapping and starvation override behavior
  - emergency prompt/template wiring in state builder

## Runtime validation still pending locally

Python runtime is unavailable in this sandbox, so these checks are pending in your machine:

1. `python inspect_traffic_lights.py`
2. `python main.py`
3. Route/event smoke tests for `routing/` and `emergency/`

## Additional completion (T05 started: backend + streaming)

- Added backend API package:
  - `api/__init__.py`
  - `api/app.py` (FastAPI + Socket.IO ASGI app, lifecycle startup/shutdown, health/control/emergency/kpi/decisions endpoints)
  - `api/broadcaster.py` (schema-versioned `frame`, `decision`, `zones`, `kpi`, `emergency` events with throttling)
- Refactored runtime loop for API-managed execution:
  - `main.py` now provides `SimulationRunner` with start/pause/resume/stop/reset and background loop integration
  - integrated CV detection, PCE computation, LLM decisioning, safety enforcement, signal apply, emergency queue/corridor activation
- Extended persistence for dashboard/API reads:
  - `data/logger.py` now includes `log_decision(...)`, `fetch_recent_decisions(...)`, `get_kpi_summary()`
- Updated execution tracking:
  - `Stages.json` T05 and subtasks `T05.1`-`T05.4` set to `in_progress`

## Local verification checklist for this stage

1. `uvicorn api.app:socket_app --port 8000 --reload`
2. `curl http://localhost:8000/health`
3. `curl http://localhost:8000/api/kpi`
4. `curl "http://localhost:8000/api/decisions?limit=20"`
5. `curl -X POST http://localhost:8000/api/control -H "Content-Type: application/json" -d "{\"action\":\"pause\"}"`
6. `curl -X POST http://localhost:8000/api/control -H "Content-Type: application/json" -d "{\"action\":\"resume\"}"`
7. `curl -X POST http://localhost:8000/api/emergency -H "Content-Type: application/json" -d "{\"ambulance_id\":\"AMB_DEMO_01\",\"destination\":\"AIIMS\",\"origin_direction\":\"north\"}"`

## Notes

- This environment still cannot execute full Python runtime smoke tests, so runtime behavior must be validated locally using the checklist above.

## Routing mode update (per latest direction)

- `routing/delhi_graph.py` now generates/loads a local weighted random graph for routing tests.
- Delhi OSM graph connectivity is deferred for now.
- `Stages.json` T04.1 and component notes updated to reflect random-graph-first workflow.
