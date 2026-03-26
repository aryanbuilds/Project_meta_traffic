"""Apply signal decisions to MetaDrive traffic lights."""

from __future__ import annotations

from agent.models import SignalDecision
from agent.phase_utils import phase_serves_direction


def _iter_lights(env):
    traffic_manager = getattr(env.engine, "traffic_manager", None)
    lights = getattr(traffic_manager, "traffic_lights", None)
    if isinstance(lights, dict):
        return list(lights.items())
    return []


def _light_direction(light_id: str) -> str | None:
    lid = str(light_id).lower()
    for d in ("north", "south", "east", "west"):
        if d in lid or d[0] in lid:
            return d
    return None


def apply_decision(env, decision: SignalDecision, step: int = 0) -> dict:
    lights = _iter_lights(env)
    applied = {"step": step, "phase": decision.phase, "count": 0}
    if not lights:
        return applied

    for light_id, light in lights:
        direction = _light_direction(light_id)
        should_green = direction is not None and phase_serves_direction(decision.phase, direction)
        try:
            if should_green:
                if hasattr(light, "set_green"):
                    light.set_green()
                elif hasattr(light, "set_status"):
                    light.set_status("green")
            else:
                if hasattr(light, "set_red"):
                    light.set_red()
                elif hasattr(light, "set_status"):
                    light.set_status("red")
            applied["count"] += 1
        except Exception:
            continue

    return applied
