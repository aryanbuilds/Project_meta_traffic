"""Apply signal decisions to MetaDrive traffic lights."""

from __future__ import annotations

from agent.models import SignalDecision
from agent.phase_utils import phase_serves_direction


def _iter_lights(env):
    traffic_manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
    lights = getattr(traffic_manager, "traffic_lights", None)
    if isinstance(lights, dict):
        return list(lights.items())
    return []


def _light_direction(light_id: str) -> str | None:
    lid = str(light_id).lower().replace("-", "_")
    token_map = {
        "north": ("north", "n_", "_n", " n "),
        "south": ("south", "s_", "_s", " s "),
        "east": ("east", "e_", "_e", " e "),
        "west": ("west", "w_", "_w", " w "),
    }
    for direction, tokens in token_map.items():
        if any(tok in lid for tok in tokens):
            return direction
    return None


def _set_light(light, target: str) -> bool:
    if target == "green":
        if hasattr(light, "set_green"):
            light.set_green()
            return True
        if hasattr(light, "set_status"):
            light.set_status("green")
            return True
    elif target == "yellow":
        if hasattr(light, "set_yellow"):
            light.set_yellow()
            return True
        if hasattr(light, "set_status"):
            light.set_status("yellow")
            return True
    else:
        if hasattr(light, "set_red"):
            light.set_red()
            return True
        if hasattr(light, "set_status"):
            light.set_status("red")
            return True
    return False


def apply_decision(env, decision: SignalDecision, step: int = 0) -> dict:
    lights = _iter_lights(env)
    applied = {
        "step": step,
        "phase": decision.phase,
        "green_count": 0,
        "red_count": 0,
        "unknown_direction_count": 0,
    }
    if not lights:
        return applied

    for light_id, light in lights:
        direction = _light_direction(str(light_id))
        if direction is None:
            applied["unknown_direction_count"] += 1
            continue

        should_green = phase_serves_direction(decision.phase, direction)
        try:
            changed = _set_light(light, "green" if should_green else "red")
            if not changed:
                continue
            if should_green:
                applied["green_count"] += 1
            else:
                applied["red_count"] += 1
        except Exception:  # noqa: BLE001
            continue

    applied["count"] = applied["green_count"] + applied["red_count"]
    return applied
