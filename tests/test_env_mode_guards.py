import pytest

from envs import intersection_env


def test_multi_agent_mode_requires_backend(monkeypatch):
    monkeypatch.setenv("SIM_ENV_MODE", "multi_agent")
    monkeypatch.setenv("AGENT_POLICY_MODE", "manual")
    monkeypatch.setattr(intersection_env, "MultiAgentIntersectionEnv", None)

    with pytest.raises(RuntimeError, match="MultiAgentIntersectionEnv"):
        intersection_env.create_intersection_env()


def test_multi_agent_mode_idm_sets_conservative_policy(monkeypatch):
    monkeypatch.setenv("SIM_ENV_MODE", "multi_agent")
    monkeypatch.setenv("AGENT_POLICY_MODE", "idm")
    monkeypatch.setenv("IDM_CONSERVATIVE_MODE", "1")
    monkeypatch.setenv("IDM_TIME_WANTED", "2.4")
    monkeypatch.setenv("IDM_DISTANCE_WANTED", "16")
    monkeypatch.setenv("IDM_ENABLE_LANE_CHANGE", "0")

    captured = {}

    class FakeMultiAgentEnv:
        @staticmethod
        def default_config():
            return {
                "enable_idm_lane_change": False,
                "disable_idm_deceleration": False,
            }

        def __init__(self, cfg):
            captured["cfg"] = cfg

    monkeypatch.setattr(intersection_env, "MultiAgentIntersectionEnv", FakeMultiAgentEnv)

    env = intersection_env.create_intersection_env()
    assert isinstance(env, FakeMultiAgentEnv)

    cfg = captured["cfg"]
    assert "agent_policy" in cfg
    assert cfg["enable_idm_lane_change"] is False
    assert cfg["disable_idm_deceleration"] is False
    assert pytest.approx(2.4) == getattr(cfg["agent_policy"], "TIME_WANTED")
    assert pytest.approx(16.0) == getattr(cfg["agent_policy"], "DISTANCE_WANTED")


def test_multi_agent_mode_skips_unsupported_idm_keys(monkeypatch):
    monkeypatch.setenv("SIM_ENV_MODE", "multi_agent")
    monkeypatch.setenv("AGENT_POLICY_MODE", "idm")

    captured = {}

    class FakeMultiAgentEnv:
        @staticmethod
        def default_config():
            return {"num_agents": 4}

        def __init__(self, cfg):
            captured["cfg"] = cfg

    monkeypatch.setattr(intersection_env, "MultiAgentIntersectionEnv", FakeMultiAgentEnv)

    intersection_env.create_intersection_env()
    cfg = captured["cfg"]
    assert "agent_policy" in cfg
    assert "enable_idm_lane_change" not in cfg
    assert "disable_idm_deceleration" not in cfg
