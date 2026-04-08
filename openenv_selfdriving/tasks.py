"""Task definitions for the self-driving collision avoidance environment."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from openenv_selfdriving.models import TaskSpec


class ObstacleSpawn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obstacle_id: str
    lane: int = Field(ge=0, le=2)
    position_m: float = Field(ge=0.0)
    speed_mps: float = Field(ge=0.0, le=35.0)


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: TaskSpec
    start_lane: int = Field(ge=0, le=2)
    start_speed_mps: float = Field(ge=0.0, le=35.0)
    obstacles: list[ObstacleSpawn]


TASK_CONFIGS: dict[str, TaskConfig] = {
    "easy_open_road": TaskConfig(
        spec=TaskSpec(
            task_id="easy_open_road",
            title="Open Road Cruising",
            difficulty="easy",
            description="Drive to destination while maintaining safe distance from one slow lead car.",
            goal_position_m=120.0,
            max_steps=80,
        ),
        start_lane=1,
        start_speed_mps=8.0,
        obstacles=[
            ObstacleSpawn(obstacle_id="lead_1", lane=1, position_m=55.0, speed_mps=5.0),
            ObstacleSpawn(obstacle_id="left_far", lane=0, position_m=85.0, speed_mps=7.5),
        ],
    ),
    "medium_lane_change": TaskConfig(
        spec=TaskSpec(
            task_id="medium_lane_change",
            title="Safe Lane Change",
            difficulty="medium",
            description="A blocked lane requires strategic lane change and speed adaptation.",
            goal_position_m=140.0,
            max_steps=90,
        ),
        start_lane=1,
        start_speed_mps=9.0,
        obstacles=[
            ObstacleSpawn(obstacle_id="blocker", lane=1, position_m=32.0, speed_mps=3.0),
            ObstacleSpawn(obstacle_id="left_peer", lane=0, position_m=35.0, speed_mps=9.0),
            ObstacleSpawn(obstacle_id="right_flow", lane=2, position_m=70.0, speed_mps=7.0),
        ],
    ),
    "hard_dense_merge": TaskConfig(
        spec=TaskSpec(
            task_id="hard_dense_merge",
            title="Dense Traffic Merge",
            difficulty="hard",
            description="Navigate dense traffic with changing hazards while avoiding collisions.",
            goal_position_m=160.0,
            max_steps=100,
        ),
        start_lane=1,
        start_speed_mps=10.0,
        obstacles=[
            ObstacleSpawn(obstacle_id="front_slow", lane=1, position_m=25.0, speed_mps=4.0),
            ObstacleSpawn(obstacle_id="left_mid", lane=0, position_m=28.0, speed_mps=6.0),
            ObstacleSpawn(obstacle_id="right_merge", lane=2, position_m=30.0, speed_mps=5.0),
            ObstacleSpawn(obstacle_id="far_brake", lane=1, position_m=95.0, speed_mps=3.0),
        ],
    ),
}


def task_specs() -> list[TaskSpec]:
    return [cfg.spec for cfg in TASK_CONFIGS.values()]
