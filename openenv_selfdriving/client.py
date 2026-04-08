"""Simple HTTP client for the self-driving OpenEnv server."""

from __future__ import annotations

from typing import Any

import requests

from openenv_selfdriving.models import (
    GradeReport,
    SelfDrivingAction,
    SelfDrivingObservation,
    SelfDrivingState,
    StepOutput,
    TaskSpec,
)


class SelfDrivingEnvClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def reset(self, task_id: str = "easy_open_road", seed: int | None = None) -> SelfDrivingObservation:
        payload: dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = int(seed)
        r = requests.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return SelfDrivingObservation.model_validate(r.json())

    def step(self, action: SelfDrivingAction) -> StepOutput:
        r = requests.post(f"{self.base_url}/step", json=action.model_dump(), timeout=self.timeout_s)
        r.raise_for_status()
        return StepOutput.model_validate(r.json())

    def state(self) -> SelfDrivingState:
        r = requests.get(f"{self.base_url}/state", timeout=self.timeout_s)
        r.raise_for_status()
        return SelfDrivingState.model_validate(r.json())

    def list_tasks(self) -> list[TaskSpec]:
        r = requests.get(f"{self.base_url}/tasks", timeout=self.timeout_s)
        r.raise_for_status()
        return [TaskSpec.model_validate(x) for x in r.json()]

    def grade(self) -> GradeReport:
        r = requests.get(f"{self.base_url}/grade", timeout=self.timeout_s)
        r.raise_for_status()
        return GradeReport.model_validate(r.json())
