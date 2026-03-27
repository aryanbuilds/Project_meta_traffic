"""MetaDrive intersection bootstrap and runtime API introspection."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from metadrive import MetaDriveEnv

from agent.signal_controller import _light_direction as _resolve_light_direction

load_dotenv()

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
    "out_of_road_done": False,
    "delay_done": 8,
    "traffic_density": 0.30,
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


def _supports_config_key(env_cls: type[Any], key: str) -> bool:
    """Guard class-specific config keys to avoid MetaDrive KeyError on startup."""
    try:
        defaults = env_cls.default_config()
    except Exception:
        return False
    try:
        return key in defaults
    except Exception:
        return False


def _set_if_supported(config: dict[str, Any], env_cls: type[Any], key: str, value: Any) -> None:
    if _supports_config_key(env_cls, key):
        config[key] = value


def _set_first_supported(config: dict[str, Any], env_cls: type[Any], values: list[tuple[str, Any]]) -> None:
    for key, value in values:
        if _supports_config_key(env_cls, key):
            config[key] = value
            break


def _safe_idm_policy_cls(base_cls: type[Any]) -> type[Any]:
    if not _env_bool("IDM_CONSERVATIVE_MODE", True):
        return base_cls

    attrs = {
        "TIME_WANTED": _env_float("IDM_TIME_WANTED", float(getattr(base_cls, "TIME_WANTED", 2.0))),
        "DISTANCE_WANTED": _env_float("IDM_DISTANCE_WANTED", 8.0),
        "NORMAL_SPEED": _env_float("IDM_NORMAL_SPEED", float(getattr(base_cls, "NORMAL_SPEED", 18.0))),
        "LANE_CHANGE_FREQ": _env_int("IDM_LANE_CHANGE_FREQ", int(getattr(base_cls, "LANE_CHANGE_FREQ", 120))),
        "SAFE_LANE_CHANGE_DISTANCE": _env_float(
            "IDM_SAFE_LANE_CHANGE_DISTANCE",
            float(getattr(base_cls, "SAFE_LANE_CHANGE_DISTANCE", 20.0)),
        ),
    }
    return type("ConservativeIDMPolicy", (base_cls,), attrs)


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

    if use_multi and MultiAgentIntersectionEnv is not None:
        merged = dict(DEFAULT_MULTI_ENV_CONFIG)
        _set_if_supported(merged, MultiAgentIntersectionEnv, "map", os.getenv("METADRIVE_MAP", "X"))
        _set_if_supported(merged, MultiAgentIntersectionEnv, "no_light", False)
        _set_if_supported(merged, MultiAgentIntersectionEnv, "traffic_light_status", True)
        merged["num_agents"] = _env_int("MULTI_AGENT_COUNT", merged["num_agents"])
        merged["allow_respawn"] = _env_bool("MULTI_ALLOW_RESPAWN", merged["allow_respawn"])
        merged["crash_done"] = _env_bool("MULTI_CRASH_DONE", merged["crash_done"])
        merged["out_of_road_done"] = _env_bool("MULTI_OUT_OF_ROAD_DONE", merged["out_of_road_done"])
        merged["delay_done"] = _env_int("MULTI_DELAY_DONE", merged["delay_done"])
        merged["traffic_density"] = _env_float("TRAFFIC_DENSITY", merged["traffic_density"])

        map_cfg = dict(merged["map_config"])
        map_cfg["exit_length"] = _env_int("INTERSECTION_EXIT_LENGTH", map_cfg["exit_length"])
        map_cfg["lane_num"] = _env_int("INTERSECTION_LANE_NUM", map_cfg["lane_num"])
        merged["map_config"] = map_cfg

        merged["start_seed"] = _env_int("SIM_SEED", merged["start_seed"])
        merged["use_render"] = _env_bool("METADRIVE_USE_RENDER", merged["use_render"])
        if _supports_config_key(MultiAgentIntersectionEnv, "vehicle_config"):
            vehicle_cfg = dict(merged.get("vehicle_config", {}))
            vehicle_cfg["enable_reverse"] = _env_bool("VEHICLE_ENABLE_REVERSE", False)
            merged["vehicle_config"] = vehicle_cfg
        if policy_mode == "idm":
            from metadrive.policy.idm_policy import IDMPolicy

            merged["agent_policy"] = _safe_idm_policy_cls(IDMPolicy)
            _set_if_supported(
                merged,
                MultiAgentIntersectionEnv,
                "enable_idm_lane_change",
                _env_bool("IDM_ENABLE_LANE_CHANGE", False),
            )
            _set_if_supported(
                merged,
                MultiAgentIntersectionEnv,
                "disable_idm_deceleration",
                _env_bool("IDM_DISABLE_DECELERATION", False),
            )

        if config:
            merged.update(config)
        return MultiAgentIntersectionEnv(merged)

    merged = dict(DEFAULT_ENV_CONFIG)
    merged["start_seed"] = _env_int("SIM_SEED", merged["start_seed"])
    merged["num_scenarios"] = _env_int("METADRIVE_NUM_SCENARIOS", merged["num_scenarios"])
    merged["map"] = os.getenv("METADRIVE_MAP", merged["map"])
    merged["use_render"] = _env_bool("METADRIVE_USE_RENDER", merged["use_render"])
    _set_if_supported(merged, MetaDriveEnv, "no_light", False)
    _set_if_supported(merged, MetaDriveEnv, "traffic_light_status", True)
    if "METADRIVE_SHOW_INTERFACE" in os.environ:
        merged["show_interface"] = _env_bool("METADRIVE_SHOW_INTERFACE", False)
    if _supports_config_key(MetaDriveEnv, "vehicle_config"):
        vehicle_cfg = dict(merged.get("vehicle_config", {}))
        vehicle_cfg["enable_reverse"] = _env_bool("VEHICLE_ENABLE_REVERSE", False)
        merged["vehicle_config"] = vehicle_cfg

    if policy_mode == "idm":
        from metadrive.policy.idm_policy import IDMPolicy

        merged["agent_policy"] = _safe_idm_policy_cls(IDMPolicy)
        _set_if_supported(
            merged,
            MetaDriveEnv,
            "enable_idm_lane_change",
            _env_bool("IDM_ENABLE_LANE_CHANGE", False),
        )
        _set_if_supported(
            merged,
            MetaDriveEnv,
            "disable_idm_deceleration",
            _env_bool("IDM_DISABLE_DECELERATION", False),
        )

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


def ensure_traffic_lights_ready(env: Any, min_count: int = 4) -> int:
    """Best-effort light provisioning for maps lacking pre-created traffic lights."""
    manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
    if manager is None:
        return 0

    lights = getattr(manager, "traffic_lights", None)
    if isinstance(lights, dict) and len(lights) >= min_count:
        return len(lights)
    if not isinstance(lights, dict):
        lights = {}
        setattr(manager, "traffic_lights", lights)

    try:
        from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
    except Exception:
        return len(lights)

    current_map = getattr(env, "current_map", None)
    road_network = getattr(current_map, "road_network", None)
    graph = getattr(road_network, "graph", None)
    if not isinstance(graph, dict):
        return len(lights)

    center_xy = None
    get_center = getattr(current_map, "get_center_point", None)
    if callable(get_center):
        try:
            center_raw = np.asarray(get_center()).reshape(-1)
            if center_raw.size >= 2:
                center_xy = np.asarray([float(center_raw[0]), float(center_raw[1])], dtype=float)
        except Exception:
            center_xy = None

    def _lane_point_xy(lane: Any, s_ratio: float) -> np.ndarray | None:
        lane_length = getattr(lane, "length", None)
        if lane_length is None:
            lane_length = getattr(lane, "LENGTH", None)
        if lane_length is None:
            lane_length = 20.0
        try:
            s = max(0.0, float(lane_length) * float(s_ratio))
        except Exception:
            s = 0.0

        position_fn = getattr(lane, "position", None)
        if callable(position_fn):
            try:
                pos = position_fn(s, 0.0)
                arr = np.asarray(pos).reshape(-1)
                if arr.size >= 2:
                    return np.asarray([float(arr[0]), float(arr[1])], dtype=float)
            except Exception:
                pass

        point_fn = getattr(lane, "point", None)
        if callable(point_fn):
            try:
                pos = point_fn(s)
                arr = np.asarray(pos).reshape(-1)
                if arr.size >= 2:
                    return np.asarray([float(arr[0]), float(arr[1])], dtype=float)
            except Exception:
                pass

        return None

    def _lane_center_score(lane: Any) -> float:
        if center_xy is None:
            return float("inf")
        sample_points = [
            _lane_point_xy(lane, 0.05),
            _lane_point_xy(lane, 0.5),
            _lane_point_xy(lane, 0.95),
        ]
        distances = [float(np.linalg.norm(p - center_xy)) for p in sample_points if p is not None]
        if not distances:
            return float("inf")
        near = min(distances)
        far = max(distances)
        # Prefer lanes that approach center but are not entirely centered.
        return near + (0.15 * max(0.0, 25.0 - (far - near)))

    def _lane_direction(lane: Any) -> str | None:
        heading_getters = (
            getattr(lane, "heading_theta_at", None),
            getattr(lane, "heading_at", None),
        )
        for getter in heading_getters:
            if not callable(getter):
                continue
            try:
                heading = getter(0.1)
            except Exception:
                continue
            try:
                angle: float | None = None
                arr = np.asarray(heading).reshape(-1)
                if arr.size >= 2:
                    import math

                    angle = float(math.atan2(float(arr[1]), float(arr[0])))
                elif arr.size == 1:
                    angle = float(arr[0])
                elif isinstance(heading, (int, float)):
                    angle = float(heading)
                if angle is None:
                    continue
                import math

                x = math.cos(angle)
                y = math.sin(angle)
                if abs(x) >= abs(y):
                    return "east" if x >= 0 else "west"
                return "north" if y >= 0 else "south"
            except Exception:
                continue
        return None

    by_direction: dict[str, tuple[Any, float]] = {}
    candidates: list[tuple[Any, str | None, float]] = []
    for _, destinations in graph.items():
        if not isinstance(destinations, dict):
            continue
        for _, lanes in destinations.items():
            if not isinstance(lanes, (list, tuple)):
                continue
            for lane in lanes:
                lane_type = type(lane).__name__.lower()
                if "straight" not in lane_type:
                    continue
                direction = _lane_direction(lane)
                if any(lane is existing for existing, _, _ in candidates):
                    continue
                center_score = _lane_center_score(lane)
                candidates.append((lane, direction, center_score))
                if direction in {"north", "south", "east", "west"}:
                    existing = by_direction.get(direction)
                    if existing is None or center_score < existing[1]:
                        by_direction[direction] = (lane, center_score)

    candidates.sort(key=lambda item: item[2])

    prioritized: list[tuple[Any, str | None, float]] = []
    for direction in ("north", "south", "east", "west"):
        by_dir = by_direction.get(direction)
        lane = by_dir[0] if by_dir is not None else None
        score = by_dir[1] if by_dir is not None else float("inf")
        if lane is not None:
            prioritized.append((lane, direction, score))
    for lane, direction, center_score in candidates:
        if any(lane is existing for existing, _, _ in prioritized):
            continue
        prioritized.append((lane, direction, center_score))

    for lane, direction, center_score in prioritized:
        if len(lights) >= min_count:
            break
        try:
            light = env.engine.spawn_object(BaseTrafficLight, lane=lane)
        except Exception:
            continue
        if light is None:
            continue
        try:
            if hasattr(light, "set_red"):
                light.set_red()
            if direction in {"north", "south", "east", "west"}:
                setattr(light, "direction", direction)
            setattr(light, "fallback_center_score", float(center_score))
        except Exception:
            pass
        lid = str(getattr(light, "id", f"fallback_light_{len(lights)}"))
        lights[lid] = light

    return len(lights)


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


def _force_light_state(light: Any, target: str) -> bool:
    if target == "green":
        if hasattr(light, "set_green"):
            light.set_green()
            return True
    elif target == "yellow":
        if hasattr(light, "set_yellow"):
            light.set_yellow()
            return True
    else:
        if hasattr(light, "set_red"):
            light.set_red()
            return True

    setter = getattr(light, "set_status", None)
    if callable(setter):
        try:
            setter(target)
            return True
        except Exception:
            return False
    return False


def _read_light_status(light: Any) -> Any:
    return getattr(light, "status", None)


def _noop_action(env: Any) -> Any:
    class_name = type(env).__name__.lower()
    if "multiagent" in class_name or "marl" in class_name:
        agents = getattr(env, "agents", None)
        if isinstance(agents, dict):
            return {aid: np.asarray([0.0, 0.0], dtype=np.float32) for aid in agents.keys()}
        return {}

    agents = getattr(env, "agents", None)
    if isinstance(agents, dict) and len(agents) > 1:
        return {aid: np.asarray([0.0, 0.0], dtype=np.float32) for aid in agents.keys()}
    return np.asarray([0.0, 0.0], dtype=np.float32)


def debug_traffic_light_obedience(
    env: Any,
    steps: int = 6,
    targets: list[str] | None = None,
    reset: bool = True,
) -> dict[str, Any]:
    """Force light states step-by-step and return a deterministic report."""
    if targets is None or len(targets) == 0:
        targets = ["green", "yellow", "red"]

    if reset:
        env.reset()
    ensure_traffic_lights_ready(env)

    manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
    lights = getattr(manager, "traffic_lights", None)
    light_items = list(lights.items()) if isinstance(lights, dict) else []
    report: dict[str, Any] = {
        "debug": get_traffic_light_debug_info(env),
        "steps": [],
    }

    for step_idx in range(max(0, int(steps))):
        target = targets[step_idx % len(targets)]
        step_entry: dict[str, Any] = {
            "step": step_idx,
            "target": target,
            "lights": [],
        }

        for light_id, light in light_items:
            resolved_direction = _resolve_light_direction(light_id, light)
            applied = _force_light_state(light, target)
            step_entry["lights"].append(
                {
                    "light_id": str(light_id),
                    "resolved_direction": resolved_direction,
                    "applied_target": target,
                    "applied": applied,
                    "status": _read_light_status(light),
                }
            )

        step_out = env.step(_noop_action(env))
        if isinstance(step_out, tuple):
            if len(step_out) >= 5:
                obs, _, terminated, truncated, info = step_out[:5]
            elif len(step_out) == 4:
                obs, _, terminated, info = step_out
                truncated = False
            elif len(step_out) == 0:
                obs, terminated, truncated, info = None, False, False, {}
            else:
                obs, terminated, truncated, info = step_out[0], False, False, {}
        else:
            obs, terminated, truncated, info = step_out, False, False, {}

        def _done_flag(value: Any) -> bool:
            if isinstance(value, dict):
                return bool(value.get("__all__", False))
            return bool(value)

        if isinstance(obs, dict):
            step_entry["vehicle_count"] = len(obs)
        else:
            step_entry["vehicle_count"] = len(getattr(env, "agents", {}) or {})
        terminated_flag = _done_flag(terminated)
        truncated_flag = _done_flag(truncated)
        step_entry["terminated"] = terminated_flag
        step_entry["truncated"] = truncated_flag
        step_entry["info"] = info if isinstance(info, dict) else {}
        report["steps"].append(step_entry)

        if terminated_flag or truncated_flag:
            env.reset()
            manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
            lights = getattr(manager, "traffic_lights", None)
            light_items = list(lights.items()) if isinstance(lights, dict) else []

    return report


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
        ensure_traffic_lights_ready(env)
        frame = render_topdown_frame(env)
        frame_path = save_test_frame(frame)
        debug = get_traffic_light_debug_info(env)
        debug["test_frame_path"] = frame_path
        return debug
    finally:
        env.close()





