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
