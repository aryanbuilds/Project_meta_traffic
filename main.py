"""Simulation runner for direct testing and API-managed execution."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from typing import Any

import numpy as np

from api.broadcaster import SimBroadcaster
from agent.llm_agent import decide_signal
from agent.models import IntersectionState, ZoneState
from agent.phase_utils import phase_serves_direction
from agent.safety import enforce_safety
from agent.signal_controller import _light_direction as _resolve_light_direction, apply_decision
from cv.annotator import annotate_frame
from cv.detector import detect_zones, get_detector_health
from cv.pce_calculator import compute_all_zones
from cv.zone_config import save_zones_debug_image
from data.logger import get_kpi_summary, log_decision
from emergency.corridor import CorridorCoordinator
from emergency.event_handler import AmbulanceEventHandler
from envs.intersection_env import (
    cycle_traffic_lights,
    create_intersection_env,
    ensure_traffic_lights_ready,
    render_perception_frame,
    render_topdown_frame,
    save_test_frame,
)


class SimulationRunner:
    def __init__(self, sio=None) -> None:
        self.sio = sio
        self.env = None
        self.step = 0
        self.running = False
        self.paused = False
        self.fps = 0.0
        self.last_error = ""
        self.latest_decision: dict | None = None
        self._last_obs: Any = None
        self._last_info: dict[str, Any] = {}
        self.episode_index = 0
        self.episodes_completed = 0
        self.arrive_dest_count = 0
        self.episode_start_pos: list[float] | None = None
        self.episode_target = ""
        self.last_episode_result = ""
        self.agent_policy_mode = os.getenv("AGENT_POLICY_MODE", "manual").strip().lower()
        self.sim_env_mode = os.getenv("SIM_ENV_MODE", "single").strip().lower()
        self._controlled_agent_id: Any = None
        self._last_obs_by_agent: dict[Any, Any] = {}
        self.crash_vehicle_events = 0
        self.out_of_road_events = 0
        self._single_incident_latched = {"crash_vehicle": False, "out_of_road": False}
        self._multi_incident_latched: dict[Any, dict[str, bool]] = {}

        self.wait_s = {"north": 0, "south": 0, "east": 0, "west": 0}
        self.last_decision_step = -1
        self._last_tick = time.perf_counter()
        self._task: asyncio.Task | None = None
        self._corridor_task: asyncio.Task | None = None

        self.junction_id = "J0"
        self.llm_interval = max(1, int(os.getenv("LLM_DECISION_INTERVAL", "10")))
        self.emergency_hold_s = max(15, int(os.getenv("EMERGENCY_HOLD_S", "30")))
        self.yolo_inference_rate = max(1, int(os.getenv("YOLO_INFERENCE_RATE", "3")))
        self.yellow_duration_s = max(2, int(os.getenv("YELLOW_DURATION_S", "5")))
        self.auto_cycle_traffic_lights = os.getenv("AUTO_CYCLE_TRAFFIC_LIGHTS", "0").strip() == "1"
        self.pipeline_debug = os.getenv("PIPELINE_DEBUG", "0").strip() == "1"
        self.use_polygon_zones = os.getenv("USE_POLYGON_ZONES", "0").strip() == "1"
        self.broadcaster = SimBroadcaster(sio, fps_cap=int(os.getenv("BROADCAST_FPS_CAP", "10"))) if sio else None
        self._last_detection = {
            "north": [],
            "south": [],
            "east": [],
            "west": [],
            "ambulance_detected": False,
            "ambulance_direction": None,
            "bboxes": [],
        }

        self.event_handler = AmbulanceEventHandler()
        self.corridor = CorridorCoordinator()
        self._active_decision = None
        self._active_decision_start_step = -1
        self.llm_success_count = 0
        self.llm_fallback_count = 0
        self.last_signal_apply: dict[str, Any] = {}
        self.yolo_checkpoint_count = 0
        self.pce_checkpoint_count = 0

        if self.auto_cycle_traffic_lights and self.pipeline_debug:
            print("PIPELINE|event=warn|message=AUTO_CYCLE_TRAFFIC_LIGHTS enabled; this can override LLM/emergency signal logic")

    async def start(self) -> dict:
        if self.running:
            return self.status()

        if self.env is None:
            self.env = create_intersection_env()
            self._reset_env()

        self.running = True
        self.paused = False
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_loop())
        return self.status()

    async def stop(self) -> dict:
        self.running = False
        self.paused = False

        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError, SystemExit):
                await self._task
            self._task = None

        if self._corridor_task is not None and not self._corridor_task.done():
            self._corridor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, SystemExit):
                await self._corridor_task
            self._corridor_task = None

        if self.env is not None:
            self.env.close()
            self.env = None

        return self.status()

    async def pause(self) -> dict:
        self.paused = True
        return self.status()

    async def resume(self) -> dict:
        if self.running:
            self.paused = False
        return self.status()

    async def reset(self) -> dict:
        if self.env is None:
            self.env = create_intersection_env()
        self._reset_env()
        self.step = 0
        self.latest_decision = None
        self._active_decision = None
        self._active_decision_start_step = -1
        self.last_decision_step = -1
        self.wait_s = {"north": 0, "south": 0, "east": 0, "west": 0}
        self._last_detection = {
            "north": [],
            "south": [],
            "east": [],
            "west": [],
            "ambulance_detected": False,
            "ambulance_direction": None,
            "bboxes": [],
        }
        self.last_error = ""
        self.last_episode_result = ""
        self.crash_vehicle_events = 0
        self.out_of_road_events = 0
        self._single_incident_latched = {"crash_vehicle": False, "out_of_road": False}
        self._multi_incident_latched = {}
        self.llm_success_count = 0
        self.llm_fallback_count = 0
        self.last_signal_apply = {}
        self.yolo_checkpoint_count = 0
        self.pce_checkpoint_count = 0

        return self.status()

    def status(self) -> dict:
        status = {
            "status": "ok" if not self.last_error else "degraded",
            "running": self.running,
            "paused": self.paused,
            "step": self.step,
            "fps": round(self.fps, 2),
            "corridor_active": self.corridor.state.active,
            "episode_index": self.episode_index,
            "episodes_completed": self.episodes_completed,
            "arrive_dest_count": self.arrive_dest_count,
            "episode_start_pos": self.episode_start_pos,
            "episode_target": self.episode_target,
            "agent_policy_mode": self.agent_policy_mode,
            "sim_env_mode": self.sim_env_mode,
            "controlled_agent_id": str(self._controlled_agent_id) if self._controlled_agent_id is not None else None,
            "last_episode_result": self.last_episode_result,
            "crash_vehicle_events": self.crash_vehicle_events,
            "out_of_road_events": self.out_of_road_events,
            "manual_throttle_scale": self._incident_throttle_scale(),
            "last_error": self.last_error,
            "llm_success_count": self.llm_success_count,
            "llm_fallback_count": self.llm_fallback_count,
            "last_signal_apply": self.last_signal_apply,
            "yolo_checkpoint_count": self.yolo_checkpoint_count,
            "pce_checkpoint_count": self.pce_checkpoint_count,
        }
        status.update(get_detector_health())
        return status

    async def _run_loop(self) -> None:
        while self.running:
            if self.paused or self.env is None:
                await asyncio.sleep(0.05)
                continue

            try:
                await self._run_step()
            except SystemExit:
                self.last_error = "Render window closed by user"
                self.running = False
                break
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                await asyncio.sleep(0.1)

            await asyncio.sleep(0)

    async def _run_step(self) -> None:
        assert self.env is not None
        self.step += 1
        action = self._driver_action()
        step_out = self.env.step(action)
        step_tuple = step_out if isinstance(step_out, tuple) else (step_out,)
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        if self._is_multi_agent_mode():
            obs, terminated, truncated, info = self._parse_multi_agent_step(step_tuple)
        else:
            obs, terminated, truncated, info = self._parse_single_agent_step(step_tuple)

        if self.auto_cycle_traffic_lights and not self.corridor.state.active:
            cycle_traffic_lights(self.env, self.step)

        if terminated or truncated:
            self.episodes_completed += 1
            if self._is_multi_agent_mode():
                self._update_episode_result_multi(info)
            else:
                if bool(self._last_info.get("arrive_dest", False)):
                    self.arrive_dest_count += 1
                    self.last_episode_result = "arrive_dest"
                elif bool(self._last_info.get("out_of_road", False)):
                    self.last_episode_result = "out_of_road"
                elif bool(self._last_info.get("crash_vehicle", False)):
                    self.last_episode_result = "crash_vehicle"
                else:
                    self.last_episode_result = "terminated"
            self._reset_env()

        frame = render_topdown_frame(self.env)
        perception_frame = render_perception_frame(self.env, obs=self._last_obs)
        should_infer = self.step == 1 or (self.step % self.yolo_inference_rate == 0)
        if should_infer:
            detection = detect_zones(perception_frame, use_polygon_zones=self.use_polygon_zones)
            self._last_detection = detection
            self.yolo_checkpoint_count += 1
        else:
            detection = self._last_detection
        zone_pce = compute_all_zones(detection, wait_s=self.wait_s)
        self.pce_checkpoint_count += 1
        if self.pipeline_debug:
            zone_counts = {d: int(zone_pce[d]["count"]) for d in ("north", "south", "east", "west")}
            frame_source = "perspective" if not self.use_polygon_zones else "mixed"
            print(
                "PIPELINE|"
                f"step={self.step}|infer={int(should_infer)}|counts={zone_counts}|"
                f"ambulance={int(bool(detection.get('ambulance_detected', False)))}|"
                f"detector_ready={int(bool(detection.get('detector_ready', False)))}|"
                f"detector_error={detection.get('detector_error_code')}|"
                f"frame_source={frame_source}"
            )

        await self._handle_emergency_detection(detection, fresh=should_infer)
        await self._pump_emergency_queue()

        if not self.corridor.state.active:
            self._apply_signal_schedule()

        decision_payload = self.latest_decision
        if not self.corridor.state.active and self._should_decide_now():
            state = self._build_state(zone_pce, detection)
            decision, controller_type, latency_ms = await asyncio.to_thread(decide_signal, state, perception_frame)
            safe_decision = enforce_safety(decision, state)
            self._active_decision = safe_decision
            self._active_decision_start_step = self.step
            self._apply_signal_schedule()
            signal_debug = os.getenv("SIGNAL_DEBUG", "0").strip() == "1"
            signal_result = apply_decision(self.env, safe_decision, step=self.step, debug=signal_debug, control_mode="normal")
            self.last_signal_apply = signal_result
            if signal_debug:
                print(f"SIGNAL|step={self.step}|result={signal_result}")
            if self.pipeline_debug:
                print(
                    "PIPELINE|"
                    f"step={self.step}|controller={controller_type}|"
                    f"latency_ms={latency_ms:.2f}|phase={safe_decision.phase}|"
                    f"duration_s={safe_decision.duration_s}|signal_apply_count={signal_result.get('count', 0)}"
                )
            if controller_type == "llm":
                self.llm_success_count += 1
            else:
                self.llm_fallback_count += 1

            self.latest_decision = safe_decision.model_dump()
            self.last_decision_step = self.step
            self.latest_decision["yellow_phase"] = False
            decision_payload = self.latest_decision

            self._update_waits(zone_pce, safe_decision.phase)
            avg_wait = self._avg_wait()
            log_decision(
                step=self.step,
                junction_id=self.junction_id,
                phase=safe_decision.phase,
                duration_s=safe_decision.duration_s,
                reasoning=safe_decision.reasoning,
                emergency=safe_decision.emergency_detected,
                pce_north=zone_pce["north"]["total_pce"],
                pce_south=zone_pce["south"]["total_pce"],
                pce_east=zone_pce["east"]["total_pce"],
                pce_west=zone_pce["west"]["total_pce"],
                avg_wait_s=avg_wait,
                latency_ms=latency_ms,
                controller_type=controller_type,
            )

            if self.broadcaster is not None:
                await self.broadcaster.emit_decision(
                    step=self.step,
                    decision=self.latest_decision,
                    controller_type=controller_type,
                    latency_ms=latency_ms,
                )
                await self.broadcaster.emit_kpi(get_kpi_summary(), step=self.step)
        else:
            if self._active_decision is not None:
                elapsed = max(0, self.step - self._active_decision_start_step)
                in_yellow = elapsed >= int(self._active_decision.duration_s)
                phase = "none" if in_yellow else str(self._active_decision.phase)
                if self.latest_decision is not None:
                    self.latest_decision["yellow_phase"] = in_yellow
            else:
                phase = self.latest_decision["phase"] if self.latest_decision else "north_south"
            self._update_waits(zone_pce, phase)

        if self.broadcaster is not None:
            annotated = annotate_frame(perception_frame, detection, zone_pce=zone_pce, decision=decision_payload)
            await self.broadcaster.emit_frame(annotated, step=self.step)
            await self.broadcaster.emit_zones(zone_pce, step=self.step)

        dt = time.perf_counter() - self._last_tick
        if dt > 0:
            self.fps = 1.0 / dt
        self._last_tick = time.perf_counter()

    async def _handle_emergency_detection(self, detection: dict, fresh: bool = True) -> None:
        if not fresh:
            return
        detected = bool(detection.get("ambulance_detected", False))
        direction = detection.get("ambulance_direction")
        await self.event_handler.on_cv_frame(detected, direction)

    async def _pump_emergency_queue(self) -> None:
        if self._corridor_task is not None and self._corridor_task.done():
            self.event_handler.corridor_active = False
            self._corridor_task = None

        if self.corridor.state.active or self._corridor_task is not None:
            return

        try:
            event = self.event_handler.queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        self.event_handler.corridor_active = True
        self._active_decision = None
        self._active_decision_start_step = -1
        assert self.env is not None
        self._corridor_task = asyncio.create_task(
            self.corridor.activate_corridor(
                event,
                env=self.env,
                sio=self.sio,
                hold_s=self.emergency_hold_s,
            )
        )

    def _build_state(self, zone_pce: dict[str, dict], detection: dict) -> IntersectionState:
        ambulance_direction = detection.get("ambulance_direction")
        zones = {}
        for direction in ("north", "south", "east", "west"):
            z = zone_pce[direction]
            zones[direction] = ZoneState(
                pce=float(z["total_pce"]),
                count=int(z["count"]),
                tw_ratio=float(z["tw_ratio"]),
                wait_s=int(z["wait_s"]),
                ambulance_detected=bool(detection.get("ambulance_detected") and ambulance_direction == direction),
            )

        return IntersectionState(
            junction_id=self.junction_id,
            step=self.step,
            zones=zones,
            emergency_active=self.corridor.state.active,
            emergency_route=self.corridor.state.route,
            emergency_direction=ambulance_direction if ambulance_direction in {"north", "south", "east", "west"} else None,
            emergency_destination="AIIMS" if self.corridor.state.active else None,
            emergency_ambulance_id=self.corridor.state.ambulance_id,
        )

    def _update_waits(self, zone_pce: dict[str, dict], phase: str) -> None:
        for direction in ("north", "south", "east", "west"):
            has_vehicles = int(zone_pce[direction]["count"]) > 0
            if phase == "none":
                self.wait_s[direction] = self.wait_s[direction] + 1 if has_vehicles else 0
            elif has_vehicles and not phase_serves_direction(phase, direction):
                self.wait_s[direction] += 1
            else:
                self.wait_s[direction] = 0

    def _avg_wait(self) -> float:
        return sum(self.wait_s.values()) / 4.0

    def _should_decide_now(self) -> bool:
        if self._active_decision is None:
            return True
        return self.last_decision_step < 0 or (self.step - self.last_decision_step) >= self.llm_interval

    def _incident_throttle_scale(self) -> float:
        incidents = self.crash_vehicle_events + self.out_of_road_events
        if incidents >= 20:
            return 0.45
        if incidents >= 10:
            return 0.6
        if incidents >= 5:
            return 0.75
        return 1.0

    def _driver_action(self) -> Any:
        if self._is_multi_agent_mode():
            return self._driver_action_multi()
        return self._driver_action_single()

    def _driver_action_single(self) -> np.ndarray:
        if self.agent_policy_mode == "idm":
            # IDM policy computes control internally; keep env action neutral.
            return np.asarray([0.0, 0.0], dtype=np.float32)
        steer = float(os.getenv("DRIVER_STEER", "0.0"))
        base_throttle = float(os.getenv("DRIVER_THROTTLE", "0.35"))
        base_throttle *= self._incident_throttle_scale()
        throttle = self._adaptive_throttle(self._last_obs, base_throttle)
        return np.asarray([steer, throttle], dtype=np.float32)

    def _driver_action_multi(self) -> dict[Any, np.ndarray]:
        steer = float(os.getenv("DRIVER_STEER", "0.0"))
        base_throttle = float(os.getenv("DRIVER_THROTTLE", "0.35"))
        base_throttle *= self._incident_throttle_scale()
        active_ids = self._active_agent_ids()
        if not active_ids:
            return {}

        if self.agent_policy_mode == "idm":
            # IDM policy computes per-agent controls internally; keep actions neutral.
            return {aid: np.asarray([0.0, 0.0], dtype=np.float32) for aid in active_ids}

        actions: dict[Any, np.ndarray] = {}
        for aid in active_ids:
            obs = self._last_obs_by_agent.get(aid)
            throttle = self._adaptive_throttle(obs, base_throttle)
            actions[aid] = np.asarray([steer, throttle], dtype=np.float32)
        return actions

    def _adaptive_throttle(self, obs: Any, base_throttle: float) -> float:
        min_lidar = self._min_lidar_distance(obs)
        if min_lidar is None:
            return float(np.clip(base_throttle, -1.0, 1.0))
        if min_lidar < 0.12:
            return -0.4
        if min_lidar < 0.2:
            return 0.0
        if min_lidar < 0.35:
            return min(base_throttle, 0.15)
        return float(np.clip(base_throttle, -1.0, 1.0))

    def _min_lidar_distance(self, obs: Any) -> float | None:
        if not isinstance(obs, dict):
            return None
        lidar = obs.get("lidar")
        if lidar is None:
            return None
        arr = np.asarray(lidar).reshape(-1)
        if arr.size == 0:
            return None
        valid = arr[np.isfinite(arr)]
        valid = valid[valid > 0]
        if valid.size == 0:
            return None
        return float(np.min(valid))

    def _is_multi_agent_mode(self) -> bool:
        return self.sim_env_mode in {"multi", "multi_agent", "multi-agent"}

    def _active_agent_ids(self) -> list[Any]:
        env_agents = getattr(self.env, "agents", None)
        if isinstance(env_agents, dict) and env_agents:
            return list(env_agents.keys())
        return list(self._last_obs_by_agent.keys())

    def _parse_single_agent_step(self, step_tuple: tuple[Any, ...]) -> tuple[Any, bool, bool, dict[str, Any]]:
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        if len(step_tuple) == 5:
            obs = step_tuple[0]
            terminated = bool(step_tuple[2])
            truncated = bool(step_tuple[3])
            info = step_tuple[4] if isinstance(step_tuple[4], dict) else {}
        elif len(step_tuple) == 4:
            obs = step_tuple[0]
            terminated = bool(step_tuple[2])
            info = step_tuple[3] if isinstance(step_tuple[3], dict) else {}
        else:
            obs = step_tuple[0]

        self._last_obs = obs
        self._last_info = info if isinstance(info, dict) else {}
        self._record_single_incidents(self._last_info)
        return obs, terminated, truncated, self._last_info

    def _parse_multi_agent_step(self, step_tuple: tuple[Any, ...]) -> tuple[Any, bool, bool, dict[str, Any]]:
        terminated_any = False
        truncated_any = False
        info: dict[str, Any] = {}
        obs = step_tuple[0]

        if len(step_tuple) >= 5:
            terminated_map = step_tuple[2] if isinstance(step_tuple[2], dict) else {}
            truncated_map = step_tuple[3] if isinstance(step_tuple[3], dict) else {}
            info = step_tuple[4] if isinstance(step_tuple[4], dict) else {}
            terminated_any = bool(terminated_map.get("__all__", False))
            truncated_any = bool(truncated_map.get("__all__", False))
        elif len(step_tuple) == 4:
            terminated_map = step_tuple[2] if isinstance(step_tuple[2], dict) else {}
            info = step_tuple[3] if isinstance(step_tuple[3], dict) else {}
            terminated_any = bool(terminated_map.get("__all__", False))

        if isinstance(obs, dict):
            self._last_obs_by_agent = dict(obs)
            if self._controlled_agent_id not in self._last_obs_by_agent:
                self._controlled_agent_id = next(iter(self._last_obs_by_agent.keys()), None)
            self._last_obs = self._last_obs_by_agent.get(self._controlled_agent_id)
        else:
            self._last_obs = obs

        self._last_info = info if isinstance(info, dict) else {}
        self._record_multi_incidents(self._last_info)
        return obs, terminated_any, truncated_any, self._last_info

    def _record_single_incidents(self, info: dict[str, Any]) -> None:
        crash = bool(info.get("crash_vehicle", False))
        out_of_road = bool(info.get("out_of_road", False))

        if crash and not self._single_incident_latched.get("crash_vehicle", False):
            self.crash_vehicle_events += 1
        if out_of_road and not self._single_incident_latched.get("out_of_road", False):
            self.out_of_road_events += 1

        self._single_incident_latched["crash_vehicle"] = crash
        self._single_incident_latched["out_of_road"] = out_of_road

    def _record_multi_incidents(self, info_map: dict[str, Any]) -> None:
        for aid, value in info_map.items():
            if aid == "__all__" or not isinstance(value, dict):
                continue
            prev = self._multi_incident_latched.get(aid, {"crash_vehicle": False, "out_of_road": False})
            crash = bool(value.get("crash_vehicle", False))
            out_of_road = bool(value.get("out_of_road", False))

            if crash and not prev.get("crash_vehicle", False):
                self.crash_vehicle_events += 1
            if out_of_road and not prev.get("out_of_road", False):
                self.out_of_road_events += 1

            self._multi_incident_latched[aid] = {
                "crash_vehicle": crash,
                "out_of_road": out_of_road,
            }

    def _update_episode_result_multi(self, info_map: dict[str, Any]) -> None:
        result = "terminated"
        arrived_any = False
        for _, value in info_map.items():
            if not isinstance(value, dict):
                continue
            if bool(value.get("crash_vehicle", False)):
                result = "crash_vehicle"
                break
            if bool(value.get("out_of_road", False)) and result != "crash_vehicle":
                result = "out_of_road"
            if bool(value.get("arrive_dest", False)):
                arrived_any = True
        if arrived_any:
            self.arrive_dest_count += 1
            if result == "terminated":
                result = "arrive_dest"
        self.last_episode_result = result

    def _reset_env(self) -> None:
        assert self.env is not None
        reset_out = self.env.reset()
        light_count = ensure_traffic_lights_ready(self.env)
        if self.pipeline_debug:
            manager = getattr(getattr(self.env, "engine", None), "traffic_manager", None)
            lights = getattr(manager, "traffic_lights", None)
            light_meta = []
            if isinstance(lights, dict):
                for lid, light in lights.items():
                    light_meta.append(
                        {
                            "id": str(lid),
                            "direction": getattr(light, "direction", None),
                            "resolved_direction": _resolve_light_direction(str(lid), light),
                        }
                    )
            print(f"PIPELINE|event=reset|episode={self.episode_index + 1}|lights={light_count}|light_meta={light_meta[:8]}")
        self.episode_index += 1
        self._single_incident_latched = {"crash_vehicle": False, "out_of_road": False}
        self._multi_incident_latched = {}
        if isinstance(reset_out, tuple):
            self._last_obs = reset_out[0]
            maybe_info = reset_out[1] if len(reset_out) > 1 else {}
            self._last_info = maybe_info if isinstance(maybe_info, dict) else {}
        else:
            self._last_obs = reset_out
            self._last_info = {}

        if isinstance(self._last_obs, dict):
            self._last_obs_by_agent = dict(self._last_obs)
            self._controlled_agent_id = next(iter(self._last_obs_by_agent.keys()), None)
            self._last_obs = self._last_obs_by_agent.get(self._controlled_agent_id)
        else:
            self._last_obs_by_agent = {}
            self._controlled_agent_id = None

        try:
            agent = getattr(self.env, "agent", None)
            position = getattr(agent, "position", None)
            if position is not None:
                self.episode_start_pos = [float(position[0]), float(position[1])]
            else:
                self.episode_start_pos = None

            navigation = getattr(agent, "navigation", None)
            destination = ""
            if navigation is not None:
                checkpoints = getattr(navigation, "checkpoints", None)
                if checkpoints:
                    destination = str(checkpoints[-1])
                elif hasattr(navigation, "final_road"):
                    destination = str(getattr(navigation, "final_road"))
            self.episode_target = destination
        except Exception:
            self.episode_start_pos = None
            self.episode_target = ""

        try:
            save_zones_debug_image("data/zones_debug.png")
        except Exception:
            pass

    def _apply_signal_schedule(self) -> None:
        if self.env is None or self._active_decision is None:
            return

        elapsed = max(0, self.step - self._active_decision_start_step)
        green_s = int(self._active_decision.duration_s)
        yellow_s = int(self.yellow_duration_s)
        if elapsed < green_s:
            self.last_signal_apply = apply_decision(self.env, self._active_decision, step=self.step, control_mode="normal")
            return
        if elapsed < green_s + yellow_s:
            self.last_signal_apply = apply_decision(self.env, self._active_decision, step=self.step, control_mode="yellow_all")
            return
        self.last_signal_apply = apply_decision(self.env, self._active_decision, step=self.step, control_mode="all_red")
        self._active_decision = None
        self._active_decision_start_step = -1


async def run(steps: int = 300) -> None:
    runner = SimulationRunner(sio=None)
    await runner.start()
    try:
        while runner.running and runner.step < steps:
            await asyncio.sleep(0.05)

        if runner.env is not None:
            frame = render_topdown_frame(runner.env)
            save_test_frame(frame, out_path=f"data/frame_{runner.step:04d}.png")

        print(f"Simulation complete: steps={runner.step}, fps={runner.fps:.2f}")
    finally:
        await runner.stop()


if __name__ == "__main__":
    asyncio.run(run())
























