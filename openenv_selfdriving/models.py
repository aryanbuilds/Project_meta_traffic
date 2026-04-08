"""Typed models for self-driving collision-avoidance OpenEnv environment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ActionName = Literal["accelerate", "brake", "lane_left", "lane_right", "maintain"]
TaskDifficulty = Literal["easy", "medium", "hard"]


class ObstacleState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obstacle_id: str
    lane: int = Field(ge=0, le=2)
    position_m: float = Field(ge=0.0)
    speed_mps: float = Field(ge=0.0, le=35.0)


class EgoState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lane: int = Field(ge=0, le=2)
    position_m: float = Field(ge=0.0)
    speed_mps: float = Field(ge=0.0, le=35.0)


class SelfDrivingState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    task_id: str
    step_count: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    goal_position_m: float = Field(gt=0.0)
    ego: EgoState
    obstacles: list[ObstacleState]
    collisions: int = Field(ge=0)
    unsafe_events: int = Field(ge=0)
    reached_goal: bool = False
    done: bool = False
    done_reason: str = "running"
    cumulative_reward: float = 0.0


class SelfDrivingObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    step_count: int = Field(ge=0)
    ego_lane: int = Field(ge=0, le=2)
    ego_position_m: float = Field(ge=0.0)
    ego_speed_mps: float = Field(ge=0.0, le=35.0)
    distance_to_goal_m: float = Field(ge=0.0)
    nearest_ahead_by_lane_m: dict[str, float | None]
    collision_risk: float = Field(ge=0.0, le=1.0)
    recommended_action: ActionName
    message: str


class SelfDrivingAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ActionName


class RewardBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    progress: float
    safety: float
    efficiency: float
    collision: float
    goal: float
    total: float


class SelfDrivingReward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=-1.0, le=1.0)
    breakdown: RewardBreakdown


class StepOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: SelfDrivingObservation
    reward: SelfDrivingReward
    done: bool
    info: dict[str, float | int | str | bool]


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    title: str
    difficulty: TaskDifficulty
    description: str
    goal_position_m: float = Field(gt=0.0)
    max_steps: int = Field(ge=10)


class GradeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    reached_goal: bool
    collisions: int = Field(ge=0)
    unsafe_events: int = Field(ge=0)
    progress_ratio: float = Field(ge=0.0, le=1.0)
    notes: str
