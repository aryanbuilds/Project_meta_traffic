import subprocess
import sys

from openenv_selfdriving.environment import SelfDrivingOpenEnv
from openenv_selfdriving.models import SelfDrivingAction


def test_tasks_available_and_well_formed():
    env = SelfDrivingOpenEnv(seed=42)
    tasks = env.list_tasks()
    task_ids = {t.task_id for t in tasks}

    assert len(tasks) >= 3
    assert {"easy_open_road", "medium_lane_change", "hard_dense_merge"}.issubset(task_ids)


def test_reset_step_state_contract():
    env = SelfDrivingOpenEnv(seed=42)
    obs = env.reset(task_id="easy_open_road", seed=42)

    assert obs.task_id == "easy_open_road"
    assert obs.step_count == 0
    assert obs.last_action_error is None

    step = env.step(SelfDrivingAction(action="maintain"))
    state = env.state()

    assert step.observation.step_count == 1
    assert state.step_count == 1
    assert isinstance(step.done, bool)
    assert -1.0 <= step.reward.value <= 1.0


def test_grade_range_for_all_tasks():
    env = SelfDrivingOpenEnv(seed=42)

    for task in ["easy_open_road", "medium_lane_change", "hard_dense_merge"]:
        obs = env.reset(task_id=task, seed=42)
        for _ in range(12):
            step = env.step(SelfDrivingAction(action=obs.recommended_action))
            obs = step.observation
            if step.done:
                break

        grade = env.grade_current_episode()
        assert grade.task_id == task
        assert 0.0 <= grade.score <= 1.0


def test_last_action_error_on_boundary():
    env = SelfDrivingOpenEnv(seed=42)
    env.reset(task_id="easy_open_road", seed=42)

    # Ego starts at lane 1. Move left to lane 0, then try lane_left again.
    step = env.step(SelfDrivingAction(action="lane_left"))
    assert step.observation.last_action_error is None  # lane 1 -> 0 is valid

    step = env.step(SelfDrivingAction(action="lane_left"))
    assert step.observation.last_action_error == "lane_boundary_violated"
    assert step.info["last_action_error"] == "lane_boundary_violated"


def test_last_action_error_episode_done():
    env = SelfDrivingOpenEnv(seed=42)
    env.reset(task_id="easy_open_road", seed=42)

    # Run until done
    for _ in range(200):
        step = env.step(SelfDrivingAction(action="accelerate"))
        if step.done:
            break

    # Step after done
    step = env.step(SelfDrivingAction(action="maintain"))
    assert step.done is True
    assert step.info.get("last_action_error") == "episode_already_done"


def test_inference_stdout_format():
    """Verify inference.py emits [START]/[STEP]/[END] lines."""
    result = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    lines = result.stdout.strip().split("\n")

    start_lines = [l for l in lines if l.startswith("[START]")]
    step_lines = [l for l in lines if l.startswith("[STEP]")]
    end_lines = [l for l in lines if l.startswith("[END]")]

    assert len(start_lines) == 3, f"Expected 3 [START] lines, got {len(start_lines)}"
    assert len(end_lines) == 3, f"Expected 3 [END] lines, got {len(end_lines)}"
    assert len(step_lines) > 0, "Expected at least one [STEP] line"

    # Verify format of first [START] line
    s = start_lines[0]
    assert "task=" in s
    assert "env=" in s
    assert "model=" in s

    # Verify format of first [STEP] line
    st = step_lines[0]
    assert "step=" in st
    assert "action=" in st
    assert "reward=" in st
    assert "done=" in st
    assert "error=" in st

    # Verify format of first [END] line
    e = end_lines[0]
    assert "success=" in e
    assert "steps=" in e
    assert "score=" in e
    assert "rewards=" in e
