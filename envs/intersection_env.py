"""MetaDrive intersection bootstrap and runtime API introspection."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from metadrive import MetaDriveEnv

try:
    from metadrive.envs.marl_envs.marl_intersection import MultiAgentIntersectionEnv
except Exception:  # noqa: BLE001
    MultiAgentIntersectionEnv = None


DEFAULT_ENV_CONFIG = {
    "num_scenarios": 1,
    "start_seed": 42,
    "use_render": True,
    "map": "X",
}

DEFAULT_MULTI_ENV_CONFIG = {
    "num_agents": 12,
    "allow_respawn": True,
    "crash_done": False,
    "delay_done": 25,
    "traffic_density": 0.15,
    "start_seed": 42,
    "use_render": True,
    "map_config": {
        "exit_length": 100,
        "lane_num": 2,
    },
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _screen_size() -> tuple[int, int]:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        return width, height
    except Exception:
        return 1366, 768


@lru_cache(maxsize=1)
def _topdown_render_config() -> dict[str, Any]:
    base_size = max(320, int(os.getenv("FRAME_SIZE", "800")))
    width = max(320, int(os.getenv("TOPDOWN_WIDTH", str(base_size))))
    height = max(320, int(os.getenv("TOPDOWN_HEIGHT", str(base_size))))

    if _env_bool("TOPDOWN_AUTO_FIT", True):
        screen_w, screen_h = _screen_size()
        ratio = float(os.getenv("TOPDOWN_MAX_SCREEN_RATIO", "0.65"))
        max_w = max(320, int(screen_w * ratio))
        max_h = max(320, int(screen_h * ratio))
        scale = min(max_w / width, max_h / height, 1.0)
        width = max(320, int(width * scale))
        height = max(320, int(height * scale))

    film_scale = max(1.0, float(os.getenv("TOPDOWN_FILM_SCALE", "1.0")))
    film_size = (int(width * film_scale), int(height * film_scale))
    config: dict[str, Any] = {
        "window": _env_bool("TOPDOWN_WINDOW", True),
        "screen_size": (width, height),
        "film_size": film_size,
    }

    scaling_raw = os.getenv("TOPDOWN_SCALING", "3.0")
    if scaling_raw:
        config["scaling"] = float(scaling_raw)

    config["camera_mode"] = os.getenv("TOPDOWN_CAMERA_MODE", "map_center").strip().lower()

    return config


def create_intersection_env(config: dict[str, Any] | None = None) -> Any:
    env_mode = os.getenv("SIM_ENV_MODE", "single").strip().lower()
    use_multi = env_mode in {"multi", "multi_agent", "multi-agent"}
    policy_mode = os.getenv("AGENT_POLICY_MODE", "manual").strip().lower()

    if use_multi and MultiAgentIntersectionEnv is None:
        raise RuntimeError(
            "SIM_ENV_MODE is set to multi-agent, but MultiAgentIntersectionEnv is unavailable "
            "in this MetaDrive build. Switch SIM_ENV_MODE=single or install a MetaDrive version "
            "that provides metadrive.envs.marl_envs.marl_intersection.MultiAgentIntersectionEnv."
        )
    if use_multi and policy_mode == "idm":
        raise RuntimeError(
            "AGENT_POLICY_MODE=idm is not supported with SIM_ENV_MODE=multi_agent in this project. "
            "Use AGENT_POLICY_MODE=manual for multi-agent runs, or switch SIM_ENV_MODE=single to use IDMPolicy."
        )

    if use_multi and MultiAgentIntersectionEnv is not None:
        merged = dict(DEFAULT_MULTI_ENV_CONFIG)
        merged["num_agents"] = _env_int("MULTI_AGENT_COUNT", merged["num_agents"])
        merged["allow_respawn"] = _env_bool("MULTI_ALLOW_RESPAWN", merged["allow_respawn"])
        merged["crash_done"] = _env_bool("MULTI_CRASH_DONE", merged["crash_done"])
        merged["delay_done"] = _env_int("MULTI_DELAY_DONE", merged["delay_done"])
        merged["traffic_density"] = _env_float("TRAFFIC_DENSITY", merged["traffic_density"])

        map_cfg = dict(merged["map_config"])
        map_cfg["exit_length"] = _env_int("INTERSECTION_EXIT_LENGTH", map_cfg["exit_length"])
        map_cfg["lane_num"] = _env_int("INTERSECTION_LANE_NUM", map_cfg["lane_num"])
        merged["map_config"] = map_cfg

        merged["start_seed"] = _env_int("SIM_SEED", merged["start_seed"])
        merged["use_render"] = _env_bool("METADRIVE_USE_RENDER", merged["use_render"])
        if config:
            merged.update(config)
        return MultiAgentIntersectionEnv(merged)

    merged = dict(DEFAULT_ENV_CONFIG)
    merged["start_seed"] = _env_int("SIM_SEED", merged["start_seed"])
    merged["num_scenarios"] = _env_int("METADRIVE_NUM_SCENARIOS", merged["num_scenarios"])
    merged["map"] = os.getenv("METADRIVE_MAP", merged["map"])
    merged["use_render"] = _env_bool("METADRIVE_USE_RENDER", merged["use_render"])
    if "METADRIVE_SHOW_INTERFACE" in os.environ:
        merged["show_interface"] = _env_bool("METADRIVE_SHOW_INTERFACE", False)
    if policy_mode == "idm":
        from metadrive.policy.idm_policy import IDMPolicy

        merged["agent_policy"] = IDMPolicy

    if "CRASH_VEHICLE_DONE" in os.environ:
        merged["crash_vehicle_done"] = _env_bool("CRASH_VEHICLE_DONE", True)
    if "CRASH_OBJECT_DONE" in os.environ:
        merged["crash_object_done"] = _env_bool("CRASH_OBJECT_DONE", True)
    if "OUT_OF_ROAD_DONE" in os.environ:
        merged["out_of_road_done"] = _env_bool("OUT_OF_ROAD_DONE", True)
    if "TRAFFIC_DENSITY" in os.environ:
        merged["traffic_density"] = _env_float("TRAFFIC_DENSITY", 0.1)

    if config:
        merged.update(config)
    return MetaDriveEnv(merged)


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
    render_cfg = dict(_topdown_render_config())
    camera_mode = str(render_cfg.pop("camera_mode", "map_center"))
    if camera_mode == "map_center":
        current_map = getattr(env, "current_map", None)
        get_center = getattr(current_map, "get_center_point", None)
        if callable(get_center):
            try:
                render_cfg["camera_position"] = get_center()
            except Exception:
                pass
    frame = env.render(mode="topdown", **render_cfg)
    if frame is None:
        raise RuntimeError("Top-down renderer returned None")
    return frame


def cycle_traffic_lights(env: Any, step: int) -> None:
    manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
    lights = getattr(manager, "traffic_lights", None)
    if not isinstance(lights, dict) or not lights:
        return

    cycle_steps = max(20, _env_int("TRAFFIC_LIGHT_CYCLE_STEPS", 140))
    phase = (step // cycle_steps) % 3
    for light in lights.values():
        try:
            if phase == 0 and hasattr(light, "set_green"):
                light.set_green()
            elif phase == 1 and hasattr(light, "set_yellow"):
                light.set_yellow()
            elif phase == 2 and hasattr(light, "set_red"):
                light.set_red()
        except Exception:
            continue


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




