"""Apply signal decisions to MetaDrive traffic lights."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Literal
import numpy as np
from agent.models import SignalDecision
from agent.phase_utils import phase_serves_direction


def _iter_lights(env):
    traffic_manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
    lights = getattr(traffic_manager, "traffic_lights", None)
    if isinstance(lights, dict):
        return list(lights.items())
    return []


@lru_cache(maxsize=1)
def _configured_light_direction_map() -> dict[str, str]:
    raw = os.getenv("SIGNAL_LIGHT_DIRECTION_MAP", "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(obj, dict):
        return {}

    out: dict[str, str] = {}
    for key, value in obj.items():
        direction = _normalize_direction_token(value)
        if direction in {"north", "south", "east", "west"}:
            out[str(key)] = direction
    return out


def _normalize_direction_token(value: Any) -> str | None:
    if value is None:
        return None
    token = str(value).strip().lower().replace("-", "_")
    if token in {"north", "south", "east", "west"}:
        return token
    if token in {"ns", "north_south", "northsouth"}:
        return "north_south"
    if token in {"ew", "east_west", "eastwest"}:
        return "east_west"
    return None


def _direction_from_heading(theta: Any) -> str | None:
    try:
        angle = float(theta)
    except (TypeError, ValueError):
        return None

    import math

    x = math.cos(angle)
    y = math.sin(angle)
    if abs(x) >= abs(y):
        return "east" if x >= 0 else "west"
    return "north" if y >= 0 else "south"


def _light_direction(light_id: str, light: Any | None = None) -> str | None:
    configured = _configured_light_direction_map().get(str(light_id))
    if configured in {"north", "south", "east", "west"}:
        return configured

    if light is not None:
        attr_names = (
            "direction",
            "traffic_direction",
            "road_direction",
            "phase_direction",
            "served_direction",
            "compass_direction",
            "approach_direction",
        )
        for attr_name in attr_names:
            direction = _normalize_direction_token(getattr(light, attr_name, None))
            if direction in {"north", "south", "east", "west"}:
                return direction
            if direction in {"north_south", "east_west"}:
                return None

        lane = getattr(light, "lane", None)
        if lane is not None:
            for attr_name in ("traffic_direction", "direction", "approach_direction"):
                direction = _normalize_direction_token(getattr(lane, attr_name, None))
                if direction in {"north", "south", "east", "west"}:
                    return direction
            heading_getters = (
                getattr(lane, "heading_theta_at", None),
                getattr(lane, "heading_at", None),
            )
            for getter in heading_getters:
                if not callable(getter):
                    continue
                try:
                    heading = getter(0.1)
                    if not isinstance(heading, (str, bytes)):
                        heading_arr = np.asarray(heading).reshape(-1)
                    else:
                        heading_arr = np.asarray([], dtype=float)
                    if heading_arr.size >= 2:
                        import math

                        heading = math.atan2(float(heading_arr[1]), float(heading_arr[0]))
                    direction = _direction_from_heading(heading)
                    if direction is not None:
                        return direction
                except Exception:
                    continue

        road = getattr(light, "road", None)
        if road is not None:
            for attr_name in ("traffic_direction", "direction"):
                direction = _normalize_direction_token(getattr(road, attr_name, None))
                if direction in {"north", "south", "east", "west"}:
                    return direction

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


def apply_decision(
    env,
    decision: SignalDecision,
    step: int = 0,
    debug: bool = False,
    control_mode: Literal["normal", "yellow_all", "all_red"] = "normal",
) -> dict:
    lights = _iter_lights(env)
    applied = {
        "step": step,
        "phase": decision.phase,
        "control_mode": control_mode,
        "green_count": 0,
        "yellow_count": 0,
        "red_count": 0,
        "unknown_direction_count": 0,
    }
    if debug:
        applied["lights"] = []
    if not lights:
        return applied

    for light_id, light in lights:
        direction = _light_direction(str(light_id), light)
        if direction is None:
            applied["unknown_direction_count"] += 1
            changed = _set_light(light, "red")
            if changed:
                applied["red_count"] += 1
            if debug:
                applied["lights"].append(
                    {
                        "light_id": str(light_id),
                        "resolved_direction": None,
                        "applied_target": "red",
                        "applied": bool(changed),
                    }
                )
            continue

        should_green = phase_serves_direction(decision.phase, direction)
        if control_mode == "yellow_all":
            target = "yellow"
        elif control_mode == "all_red":
            target = "red"
        else:
            target = "green" if should_green else "red"
        try:
            changed = _set_light(light, target)
            if not changed:
                if debug:
                    applied["lights"].append(
                        {
                            "light_id": str(light_id),
                            "resolved_direction": direction,
                            "applied_target": target,
                            "applied": False,
                        }
                    )
                continue
            if target == "green":
                applied["green_count"] += 1
            elif target == "yellow":
                applied["yellow_count"] += 1
            else:
                applied["red_count"] += 1
            if debug:
                applied["lights"].append(
                    {
                        "light_id": str(light_id),
                        "resolved_direction": direction,
                        "applied_target": target,
                        "applied": True,
                    }
                )
        except Exception:  # noqa: BLE001
            if debug:
                applied["lights"].append(
                    {
                        "light_id": str(light_id),
                        "resolved_direction": direction,
                        "applied_target": target,
                        "applied": False,
                    }
                )
            continue

    applied["count"] = applied["green_count"] + applied["yellow_count"] + applied["red_count"]
    return applied





