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
from agent.signal_controller import apply_decision
from cv.annotator import annotate_frame
from cv.detector import detect_zones
from cv.pce_calculator import compute_all_zones
from data.logger import get_kpi_summary, log_decision
from emergency.corridor import CorridorCoordinator
from emergency.event_handler import AmbulanceEventHandler
from envs.intersection_env import create_intersection_env, render_topdown_frame, save_test_frame


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
        self.last_episode_result = ""

        self.wait_s = {"north": 0, "south": 0, "east": 0, "west": 0}
        self.last_decision_step = -1
        self._last_tick = time.perf_counter()
        self._task: asyncio.Task | None = None
        self._corridor_task: asyncio.Task | None = None

        self.junction_id = "J0"
        self.llm_interval = max(1, int(os.getenv("LLM_DECISION_INTERVAL", "10")))
        self.emergency_hold_s = max(15, int(os.getenv("EMERGENCY_HOLD_S", "30")))
        self.yolo_inference_rate = max(1, int(os.getenv("YOLO_INFERENCE_RATE", "3")))
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
        return self.status()

    def status(self) -> dict:
        return {
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
            "last_episode_result": self.last_episode_result,
            "last_error": self.last_error,
        }

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

        if terminated or truncated:
            self.episodes_completed += 1
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
        should_infer = self.step == 1 or (self.step % self.yolo_inference_rate == 0)
        if should_infer:
            detection = detect_zones(frame)
            self._last_detection = detection
        else:
            detection = self._last_detection
        zone_pce = compute_all_zones(detection, wait_s=self.wait_s)

        await self._handle_emergency_detection(detection)
        await self._pump_emergency_queue()

        decision_payload = self.latest_decision
        if self._should_decide_now():
            state = self._build_state(zone_pce, detection)
            decision, controller_type, latency_ms = await asyncio.to_thread(decide_signal, state, frame)
            safe_decision = enforce_safety(decision, state)
            apply_decision(self.env, safe_decision, step=self.step)

            self.latest_decision = safe_decision.model_dump()
            self.last_decision_step = self.step
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
            phase = self.latest_decision["phase"] if self.latest_decision else "north_south"
            self._update_waits(zone_pce, phase)

        if self.broadcaster is not None:
            annotated = annotate_frame(frame, detection, zone_pce=zone_pce, decision=decision_payload)
            await self.broadcaster.emit_frame(annotated, step=self.step)
            await self.broadcaster.emit_zones(zone_pce, step=self.step)

        dt = time.perf_counter() - self._last_tick
        if dt > 0:
            self.fps = 1.0 / dt
        self._last_tick = time.perf_counter()

    async def _handle_emergency_detection(self, detection: dict) -> None:
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
            if has_vehicles and not phase_serves_direction(phase, direction):
                self.wait_s[direction] += 1
            else:
                self.wait_s[direction] = 0

    def _avg_wait(self) -> float:
        return sum(self.wait_s.values()) / 4.0

    def _should_decide_now(self) -> bool:
        return self.last_decision_step < 0 or (self.step - self.last_decision_step) >= self.llm_interval

    def _driver_action(self) -> np.ndarray:
        # Keep a stable forward-driving baseline to avoid random stuck behavior.
        steer = float(os.getenv("DRIVER_STEER", "0.0"))
        throttle = float(os.getenv("DRIVER_THROTTLE", "0.35"))
        return np.asarray([steer, throttle], dtype=np.float32)

    def _reset_env(self) -> None:
        assert self.env is not None
        reset_out = self.env.reset()
        self.episode_index += 1
        if isinstance(reset_out, tuple):
            self._last_obs = reset_out[0]
            maybe_info = reset_out[1] if len(reset_out) > 1 else {}
            self._last_info = maybe_info if isinstance(maybe_info, dict) else {}
        else:
            self._last_obs = reset_out
            self._last_info = {}

        try:
            agent = getattr(self.env, "agent", None)
            position = getattr(agent, "position", None)
            if position is not None:
                self.episode_start_pos = [float(position[0]), float(position[1])]
            else:
                self.episode_start_pos = None
        except Exception:
            self.episode_start_pos = None


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
