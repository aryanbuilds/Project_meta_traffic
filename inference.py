"""Round-1 baseline inference for OpenEnv self-driving collision avoidance.

Mandatory stdout format: [START], [STEP], [END] lines per hackathon spec.
Uses OpenAI-compatible client for LLM action selection.
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

from openenv_selfdriving.environment import SelfDrivingOpenEnv
from openenv_selfdriving.models import SelfDrivingAction, SelfDrivingObservation


ALLOWED_ACTIONS = ["accelerate", "brake", "lane_left", "lane_right", "maintain"]
DEFAULT_SEED = 42
MAX_STEPS_BUFFER = 5
BENCHMARK = "openenv-selfdriving-collision-avoidance"


def _fallback_policy(observation: SelfDrivingObservation) -> str:
    """Safety-first deterministic fallback when no LLM call is possible."""
    return observation.recommended_action


def _build_prompt(observation: SelfDrivingObservation, history: list[str]) -> str:
    history_text = " | ".join(history[-4:]) if history else "none"
    return (
        "You control a self-driving car in a 3-lane road. "
        "Return exactly one action from this list: "
        "accelerate, brake, lane_left, lane_right, maintain.\n"
        "Prioritize collision avoidance and then progress.\n"
        f"task_id={observation.task_id}; step={observation.step_count}; "
        f"ego_lane={observation.ego_lane}; ego_speed={observation.ego_speed_mps}; "
        f"dist_goal={observation.distance_to_goal_m}; "
        f"risk={observation.collision_risk}; "
        f"nearest={json.dumps(observation.nearest_ahead_by_lane_m)}; "
        f"history={history_text}\n"
        "Answer with action only."
    )


def _query_llm(
    client: OpenAI,
    model_name: str,
    observation: SelfDrivingObservation,
    history: list[str],
) -> str:
    prompt = _build_prompt(observation, history)
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": "You are a cautious autonomous-driving policy.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    raw = (response.choices[0].message.content or "").strip().lower()
    token = raw.split()[0] if raw else ""
    if token in ALLOWED_ACTIONS:
        return token
    return _fallback_policy(observation)


def _fmt_bool(v: bool) -> str:
    return "true" if v else "false"


def run_task(
    env: SelfDrivingOpenEnv,
    task_id: str,
    client: OpenAI | None,
    model_name: str,
) -> dict[str, Any]:
    observation = env.reset(task_id=task_id, seed=DEFAULT_SEED)
    history: list[str] = []
    rewards: list[float] = []
    step_limit = env.state().max_steps + MAX_STEPS_BUFFER

    # [START]
    print(f"[START] task={task_id} env={BENCHMARK} model={model_name}", flush=True)

    done = False
    last_error: str | None = None
    step_n = 0

    try:
        for _ in range(step_limit):
            if done:
                break
            if client is None:
                action_name = _fallback_policy(observation)
            else:
                try:
                    action_name = _query_llm(client, model_name, observation, history)
                except Exception:
                    action_name = _fallback_policy(observation)

            step_out = env.step(SelfDrivingAction(action=action_name))
            history.append(action_name)
            observation = step_out.observation
            done = step_out.done
            reward_val = step_out.reward.value
            rewards.append(reward_val)
            step_n += 1

            last_error = step_out.info.get("last_action_error")
            error_str = last_error if last_error else "null"

            # [STEP]
            print(
                f"[STEP] step={step_n} action={action_name} "
                f"reward={reward_val:.2f} done={_fmt_bool(done)} "
                f"error={error_str}",
                flush=True,
            )
    finally:
        state = env.state()
        grade = env.grade_current_episode()
        score = grade.score
        success = state.reached_goal
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        # [END]
        print(
            f"[END] success={_fmt_bool(success)} steps={step_n} "
            f"score={score:.2f} rewards={rewards_str}",
            flush=True,
        )

    return {
        "task_id": task_id,
        "done": state.done,
        "done_reason": state.done_reason,
        "steps": state.step_count,
        "collisions": state.collisions,
        "unsafe_events": state.unsafe_events,
        "reached_goal": state.reached_goal,
        "last_reward": round(rewards[-1], 4) if rewards else 0.0,
        "grade_score": round(grade.score, 4),
    }


def main() -> None:
    random.seed(DEFAULT_SEED)

    api_base_url = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
    model_name = os.getenv("MODEL_NAME", "google/gemma-4-31b-it:free")
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

    client: OpenAI | None = None
    if api_key:
        client = OpenAI(base_url=api_base_url, api_key=api_key)

    env = SelfDrivingOpenEnv(seed=DEFAULT_SEED)
    task_ids = [task.task_id for task in env.list_tasks()]

    results: list[dict[str, Any]] = []
    for task_id in task_ids:
        results.append(run_task(env, task_id, client, model_name))

    mean_score = sum(r["grade_score"] for r in results) / max(1, len(results))
    summary = {
        "model": model_name,
        "api_base_url": api_base_url,
        "used_llm": bool(client is not None),
        "seed": DEFAULT_SEED,
        "mean_score": round(mean_score, 4),
        "results": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
