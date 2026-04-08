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
