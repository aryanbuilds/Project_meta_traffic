"""Deterministic task graders for OpenEnv self-driving tasks."""

from __future__ import annotations

from openenv_selfdriving.models import GradeReport, SelfDrivingState


def grade_task(state: SelfDrivingState) -> GradeReport:
    progress_ratio = min(1.0, state.ego.position_m / max(1.0, state.goal_position_m))
    reached_goal = bool(state.reached_goal)

    goal_component = 0.6 if reached_goal else (0.35 * progress_ratio)
    efficiency_bonus = 0.0
    if reached_goal:
        efficiency_bonus = 0.2 * max(0.0, 1.0 - (state.step_count / max(1, state.max_steps)))

    collision_penalty = 0.6 if state.collisions > 0 else 0.0
    unsafe_penalty = min(0.3, state.unsafe_events * 0.03)

    score = goal_component + efficiency_bonus - collision_penalty - unsafe_penalty
    score = max(0.0, min(1.0, score))

    notes = (
        f"goal={int(reached_goal)} progress={progress_ratio:.2f} "
        f"collisions={state.collisions} unsafe={state.unsafe_events}"
    )

    return GradeReport(
        task_id=state.task_id,
        score=score,
        reached_goal=reached_goal,
        collisions=state.collisions,
        unsafe_events=state.unsafe_events,
        progress_ratio=progress_ratio,
        notes=notes,
    )
