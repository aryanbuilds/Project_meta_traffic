"""Pydantic models for traffic signal decisioning."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


VALID_PHASE = Literal["north", "south", "east", "west", "north_south", "east_west"]
VALID_DIRECTION = Literal["north", "south", "east", "west"]


class ZoneState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pce: float = Field(ge=0.0)
    count: int = Field(ge=0)
    tw_ratio: float = Field(ge=0.0, le=1.0)
    wait_s: int = Field(ge=0)
    ambulance_detected: bool = False


class IntersectionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    junction_id: str
    step: int = Field(ge=0)
    zones: dict[VALID_DIRECTION, ZoneState]
    emergency_active: bool = False
    emergency_route: list[str] | None = None
    emergency_direction: VALID_DIRECTION | None = None
    emergency_destination: str | None = None
    emergency_ambulance_id: str | None = None


class SignalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: VALID_PHASE
    duration_s: int = Field(ge=15, le=60)
    skip_phases: list[str] = Field(default_factory=list)
    emergency_detected: bool = False
    emergency_direction: VALID_DIRECTION | None = None
    reasoning: str = Field(min_length=20)
