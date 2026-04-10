"""FastAPI app exposing reset/step/state APIs for OpenEnv validation."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from openenv_selfdriving.environment import SelfDrivingOpenEnv
from openenv_selfdriving.models import GradeReport, SelfDrivingAction, SelfDrivingObservation, SelfDrivingState, StepOutput, TaskSpec


class ResetRequest(BaseModel):
    task_id: str = Field(default="easy_open_road")
    seed: int | None = None


env = SelfDrivingOpenEnv(seed=42)


@asynccontextmanager
async def lifespan(_: FastAPI):
    env.reset(task_id="easy_open_road", seed=42)
    yield


app = FastAPI(title="OpenEnv Self-Driving Collision Avoidance", version="0.1.0", lifespan=lifespan)


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks", response_model=list[TaskSpec])
async def tasks() -> list[TaskSpec]:
    return env.list_tasks()


@app.post("/reset", response_model=SelfDrivingObservation)
async def reset(req: ResetRequest) -> SelfDrivingObservation:
    return env.reset(task_id=req.task_id, seed=req.seed)


@app.post("/step", response_model=StepOutput)
async def step(action: SelfDrivingAction) -> StepOutput:
    return env.step(action)


@app.get("/state", response_model=SelfDrivingState)
async def state() -> SelfDrivingState:
    return env.state()


@app.get("/grade", response_model=GradeReport)
async def grade() -> GradeReport:
    return env.grade_current_episode()
