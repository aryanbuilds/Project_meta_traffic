"""MetaDrive intersection bootstrap and runtime API introspection."""

from pathlib import Path
from typing import Any

import numpy as np
from metadrive import MetaDriveEnv


DEFAULT_ENV_CONFIG = {
    "num_scenarios": 1,
    "start_seed": 42,
    "use_render": True,
    "map": "X",
}


def create_intersection_env(config: dict[str, Any] | None = None) -> MetaDriveEnv:
    merged = dict(DEFAULT_ENV_CONFIG)
    if config:
        merged.update(config)
    env = MetaDriveEnv(merged)
    return env


def get_traffic_light_debug_info(env: MetaDriveEnv) -> dict[str, Any]:
    traffic_manager = getattr(env.engine, "traffic_manager", None)
    lights = getattr(traffic_manager, "traffic_lights", None)
    methods = [m for m in dir(traffic_manager) if "light" in m.lower()] if traffic_manager else []

    first_light_methods: list[str] = []
    light_count = 0
    if isinstance(lights, dict):
        light_count = len(lights)
        if lights:
            first = next(iter(lights.values()))
            first_light_methods = [m for m in dir(first) if m.startswith("set_")]

    return {
        "traffic_manager_type": type(traffic_manager).__name__ if traffic_manager else None,
        "traffic_manager_light_methods": methods,
        "traffic_lights_count": light_count,
        "first_light_set_methods": first_light_methods,
    }


def render_topdown_frame(env: MetaDriveEnv) -> np.ndarray:
    frame = env.render(mode="topdown", film_size=(800, 800), screen_size=(800, 800))
    if frame is None:
        raise RuntimeError("Top-down renderer returned None")
    return frame


def save_test_frame(frame: np.ndarray, out_path: str = "data/test_frame.png") -> str:
    import cv2

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame)
    if not ok:
        raise RuntimeError(f"Failed to save frame to {path}")
    return str(path)


def bootstrap_and_capture(config: dict[str, Any] | None = None) -> dict[str, Any]:
    env = create_intersection_env(config)
    try:
        env.reset()
        frame = render_topdown_frame(env)
        frame_path = save_test_frame(frame)
        debug = get_traffic_light_debug_info(env)
        debug["test_frame_path"] = frame_path
        return debug
    finally:
        env.close()
