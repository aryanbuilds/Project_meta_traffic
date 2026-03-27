import pytest

from envs import intersection_env


def test_multi_agent_mode_requires_backend(monkeypatch):
    monkeypatch.setenv("SIM_ENV_MODE", "multi_agent")
    monkeypatch.setenv("AGENT_POLICY_MODE", "manual")
    monkeypatch.setattr(intersection_env, "MultiAgentIntersectionEnv", None)

    with pytest.raises(RuntimeError, match="MultiAgentIntersectionEnv"):
        intersection_env.create_intersection_env()


def test_multi_agent_mode_rejects_idm(monkeypatch):
    monkeypatch.setenv("SIM_ENV_MODE", "multi_agent")
    monkeypatch.setenv("AGENT_POLICY_MODE", "idm")
    monkeypatch.setattr(intersection_env, "MultiAgentIntersectionEnv", object())

    with pytest.raises(RuntimeError, match="not supported"):
        intersection_env.create_intersection_env()
