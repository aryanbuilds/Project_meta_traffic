"""Minimal OpenEnv-style self-driving collision-avoidance simulator."""

from __future__ import annotations

import uuid
from copy import deepcopy
from random import Random

from openenv_selfdriving.graders import grade_task
from openenv_selfdriving.models import (
    EgoState,
    ObstacleState,
    RewardBreakdown,
    SelfDrivingAction,
    SelfDrivingObservation,
    SelfDrivingReward,
    SelfDrivingState,
    StepOutput,
    TaskSpec,
)
from openenv_selfdriving.tasks import TASK_CONFIGS, task_specs


class SelfDrivingOpenEnv:
    """Deterministic environment with reset/step/state APIs."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = Random(seed)
        self._seed = seed
        self._state: SelfDrivingState | None = None
        self._last_action_error: str | None = None

    def list_tasks(self) -> list[TaskSpec]:
        return task_specs()

    def reset(self, task_id: str = "easy_open_road", seed: int | None = None) -> SelfDrivingObservation:
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id: {task_id}")
        if seed is not None:
            self._seed = int(seed)
            self._rng = Random(self._seed)

        self._last_action_error = None
        cfg = TASK_CONFIGS[task_id]
        obstacles = [
            ObstacleState(
                obstacle_id=o.obstacle_id,
                lane=o.lane,
                position_m=float(o.position_m),
                speed_mps=float(o.speed_mps),
            )
            for o in cfg.obstacles
        ]

        self._state = SelfDrivingState(
            episode_id=str(uuid.uuid4()),
            task_id=task_id,
            step_count=0,
            max_steps=cfg.spec.max_steps,
            goal_position_m=cfg.spec.goal_position_m,
            ego=EgoState(lane=cfg.start_lane, position_m=0.0, speed_mps=cfg.start_speed_mps),
            obstacles=obstacles,
            collisions=0,
            unsafe_events=0,
            reached_goal=False,
            done=False,
            done_reason="running",
            cumulative_reward=0.0,
        )
        return self._build_observation()

    def step(self, action: SelfDrivingAction) -> StepOutput:
        state = self._require_state()
        self._last_action_error = None

        if state.done:
            self._last_action_error = "episode_already_done"
            reward = SelfDrivingReward(
                value=0.0,
                breakdown=RewardBreakdown(
                    progress=0.0,
                    safety=0.0,
                    efficiency=0.0,
                    collision=0.0,
                    goal=0.0,
                    total=0.0,
                ),
            )
            return StepOutput(
                observation=self._build_observation(),
                reward=reward,
                done=True,
                info={"reason": "episode_already_done", "last_action_error": self._last_action_error},
            )

        prev_position = float(state.ego.position_m)
        self._apply_action(action.action)
        self._advance_world()
        self._apply_task_events()

        state.step_count += 1

        collision = self._detect_collision()
        if collision:
            state.collisions += 1
            state.done = True
            state.done_reason = "collision"

        unsafe = self._unsafe_headway()
        if unsafe:
            state.unsafe_events += 1

        if state.ego.position_m >= state.goal_position_m:
            state.reached_goal = True
            state.done = True
            state.done_reason = "reached_goal"

        if state.step_count >= state.max_steps and not state.done:
            state.done = True
            state.done_reason = "max_steps"

        reward = self._compute_reward(prev_position=prev_position, collision=collision, unsafe=unsafe)
        state.cumulative_reward += reward.value

        obs = self._build_observation()
        info: dict[str, float | int | str | bool | None] = {
            "task_id": state.task_id,
            "step_count": state.step_count,
            "collisions": state.collisions,
            "unsafe_events": state.unsafe_events,
            "done_reason": state.done_reason,
            "cumulative_reward": round(state.cumulative_reward, 4),
            "last_action_error": self._last_action_error,
        }
        if state.done:
            report = grade_task(state)
            info["grade_score"] = round(report.score, 4)
            info["grade_notes"] = report.notes

        return StepOutput(observation=obs, reward=reward, done=state.done, info=info)

    def state(self) -> SelfDrivingState:
        return deepcopy(self._require_state())

    def grade_current_episode(self):
        return grade_task(self._require_state())

    def _require_state(self) -> SelfDrivingState:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    def _apply_action(self, action: str) -> None:
        state = self._require_state()
        ego = state.ego

        if action == "accelerate":
            ego.speed_mps = min(22.0, ego.speed_mps + 2.0)
        elif action == "brake":
            ego.speed_mps = max(0.0, ego.speed_mps - 3.0)
        elif action == "lane_left":
            if ego.lane > 0:
                ego.lane -= 1
            else:
                state.unsafe_events += 1
                self._last_action_error = "lane_boundary_violated"
        elif action == "lane_right":
            if ego.lane < 2:
                ego.lane += 1
            else:
                state.unsafe_events += 1
                self._last_action_error = "lane_boundary_violated"
        elif action == "maintain":
            pass
        else:
            state.unsafe_events += 1
            self._last_action_error = f"unknown_action:{action}"

    def _advance_world(self) -> None:
        state = self._require_state()
        dt = 1.0
        state.ego.position_m += state.ego.speed_mps * dt
        for obstacle in state.obstacles:
            obstacle.position_m += obstacle.speed_mps * dt

    def _apply_task_events(self) -> None:
        state = self._require_state()
        if state.task_id != "hard_dense_merge":
            return

        # Merge event: right lane traffic moves into center lane early.
        if 12 <= state.step_count <= 16:
            for obstacle in state.obstacles:
                if obstacle.obstacle_id == "right_merge" and obstacle.lane == 2:
                    obstacle.lane = 1
                    obstacle.speed_mps = max(2.0, obstacle.speed_mps - 1.5)

        # Sudden braking event for the far lead vehicle.
        if state.step_count >= 25:
            for obstacle in state.obstacles:
                if obstacle.obstacle_id == "far_brake":
                    obstacle.speed_mps = max(1.0, obstacle.speed_mps - 0.8)

        # Lane-cutter event: obstacle suddenly swerves from right to center lane.
        if 20 <= state.step_count <= 24:
            for obstacle in state.obstacles:
                if obstacle.obstacle_id == "lane_cutter" and obstacle.lane == 2:
                    obstacle.lane = 1
                    obstacle.speed_mps = max(3.0, obstacle.speed_mps - 2.0)

        # Left fast vehicle slows abruptly (simulates erratic two-wheeler behavior).
        if state.step_count >= 30:
            for obstacle in state.obstacles:
                if obstacle.obstacle_id == "left_fast":
                    obstacle.speed_mps = max(2.0, obstacle.speed_mps - 1.0)

        # Late-stage squeeze: left blocker shifts to center lane.
        if 50 <= state.step_count <= 55:
            for obstacle in state.obstacles:
                if obstacle.obstacle_id == "left_blocker" and obstacle.lane == 0:
                    obstacle.lane = 1

    def _detect_collision(self) -> bool:
        state = self._require_state()
        ego = state.ego
        for obstacle in state.obstacles:
            if obstacle.lane != ego.lane:
                continue
            if abs(obstacle.position_m - ego.position_m) < 2.5:
                return True
        return False

    def _unsafe_headway(self) -> bool:
        state = self._require_state()
        ego = state.ego
        ahead = [o.position_m - ego.position_m for o in state.obstacles if o.lane == ego.lane and o.position_m >= ego.position_m]
        if not ahead:
            return False
        return min(ahead) < 6.0

    def _nearest_ahead_by_lane(self) -> dict[str, float | None]:
        state = self._require_state()
        ego_pos = state.ego.position_m
        out: dict[str, float | None] = {"left": None, "center": None, "right": None}
        lane_name = {0: "left", 1: "center", 2: "right"}
        for lane in (0, 1, 2):
            dists = [o.position_m - ego_pos for o in state.obstacles if o.lane == lane and o.position_m >= ego_pos]
            out[lane_name[lane]] = round(min(dists), 2) if dists else None
        return out

    def _recommended_action(self) -> str:
        state = self._require_state()
        lane_map = self._nearest_ahead_by_lane()
        current = lane_map["left" if state.ego.lane == 0 else "center" if state.ego.lane == 1 else "right"]

        if current is not None and current < 12.0:
            if state.ego.lane > 0 and (lane_map["left"] is None or lane_map["left"] > current + 4.0):
                return "lane_left"
            if state.ego.lane < 2 and (lane_map["right"] is None or lane_map["right"] > current + 4.0):
                return "lane_right"
            return "brake"
        if state.ego.speed_mps < 12.0:
            return "accelerate"
        return "maintain"

    def _collision_risk(self) -> float:
        state = self._require_state()
        ego = state.ego
        nearest = [o.position_m - ego.position_m for o in state.obstacles if o.lane == ego.lane and o.position_m >= ego.position_m]
        if not nearest:
            return 0.0
        d = min(nearest)
        if d <= 3.0:
            return 1.0
        if d <= 6.0:
            return 0.75
        if d <= 10.0:
            return 0.4
        return 0.1

    def _build_observation(self) -> SelfDrivingObservation:
        state = self._require_state()
        return SelfDrivingObservation(
            task_id=state.task_id,
            step_count=state.step_count,
            ego_lane=state.ego.lane,
            ego_position_m=round(state.ego.position_m, 2),
            ego_speed_mps=round(state.ego.speed_mps, 2),
            distance_to_goal_m=round(max(0.0, state.goal_position_m - state.ego.position_m), 2),
            nearest_ahead_by_lane_m=self._nearest_ahead_by_lane(),
            collision_risk=round(self._collision_risk(), 2),
            recommended_action=self._recommended_action(),
            message=(
                "Avoid collisions and reach the goal quickly. "
                "Use lane changes only when safe headway exists."
            ),
            last_action_error=self._last_action_error,
        )

    def _compute_reward(self, prev_position: float, collision: bool, unsafe: bool) -> SelfDrivingReward:
        state = self._require_state()
        progress = max(0.0, state.ego.position_m - prev_position) / max(1.0, state.goal_position_m)
        progress_reward = min(0.6, progress * 1.5)

        safety_penalty = -0.08 if unsafe else 0.0
        efficiency_penalty = -0.01
        collision_penalty = -1.0 if collision else 0.0
        goal_bonus = 0.5 if state.reached_goal else 0.0

        total = progress_reward + safety_penalty + efficiency_penalty + collision_penalty + goal_bonus
        total = max(-1.0, min(1.0, total))

        breakdown = RewardBreakdown(
            progress=round(progress_reward, 4),
            safety=round(safety_penalty, 4),
            efficiency=round(efficiency_penalty, 4),
            collision=round(collision_penalty, 4),
            goal=round(goal_bonus, 4),
            total=round(total, 4),
        )
        return SelfDrivingReward(value=round(total, 4), breakdown=breakdown)
