from envs import intersection_env


class FakeTrafficLight:
    def __init__(self, status="red", direction=None, lane=None):
        self.status = status
        self.direction = direction
        self.lane = lane

    def set_green(self):
        self.status = "green"

    def set_yellow(self):
        self.status = "yellow"

    def set_red(self):
        self.status = "red"


class FakeTrafficManager:
    def __init__(self, lights):
        self.traffic_lights = lights


class FakeEngine:
    def __init__(self, lights):
        self.traffic_manager = FakeTrafficManager(lights)


class FakeEnv:
    def __init__(self, lights):
        self.engine = FakeEngine(lights)
        self.reset_calls = 0
        self.step_calls = 0
        self.agents = {}

    def reset(self):
        self.reset_calls += 1
        return {}

    def step(self, action):
        self.step_calls += 1
        return {}, {}, False, False, {}


class FakeMultiAgentEnv:
    @staticmethod
    def default_config():
        return {
            "map": "SSS",
            "no_light": True,
            "traffic_light_status": False,
            "num_agents": 4,
        }

    def __init__(self, cfg):
        self.cfg = cfg


def test_multi_agent_intersection_sets_supported_light_keys_and_map(monkeypatch):
    monkeypatch.setenv("SIM_ENV_MODE", "multi_agent")
    monkeypatch.setenv("AGENT_POLICY_MODE", "manual")
    monkeypatch.setenv("METADRIVE_MAP", "X")
    monkeypatch.setattr(intersection_env, "MultiAgentIntersectionEnv", FakeMultiAgentEnv)

    env = intersection_env.create_intersection_env()

    assert isinstance(env, FakeMultiAgentEnv)
    assert env.cfg["map"] == "X"
    assert env.cfg["no_light"] is False
    assert env.cfg["traffic_light_status"] is True


def test_multi_agent_intersection_keeps_default_map_when_not_overridden(monkeypatch):
    monkeypatch.setenv("SIM_ENV_MODE", "multi_agent")
    monkeypatch.setenv("AGENT_POLICY_MODE", "manual")
    monkeypatch.delenv("METADRIVE_MAP", raising=False)
    monkeypatch.setattr(intersection_env, "MultiAgentIntersectionEnv", FakeMultiAgentEnv)

    env = intersection_env.create_intersection_env()

    assert "map" not in env.cfg
    assert env.cfg["no_light"] is False
    assert env.cfg["traffic_light_status"] is True


def test_debug_traffic_light_obedience_reports_light_status(monkeypatch):
    lane = type("Lane", (), {"heading_theta_at": lambda self, _: 1.57})()
    lights = {
        "north_light": FakeTrafficLight(direction="north"),
        "east_light": FakeTrafficLight(lane=lane),
    }
    env = FakeEnv(lights)

    report = intersection_env.debug_traffic_light_obedience(env, steps=2, targets=["red", "green"], reset=False)

    assert report["debug"]["traffic_lights_count"] == 2
    assert len(report["steps"]) == 2
    assert report["steps"][0]["lights"][0]["resolved_direction"] == "north"
    assert report["steps"][1]["lights"][1]["resolved_direction"] in {"north", "east"}
    assert report["steps"][0]["lights"][0]["applied_target"] == "red"
    assert report["steps"][0]["lights"][0]["status"] == "red"
    assert report["steps"][1]["lights"][0]["status"] == "green"
    assert env.step_calls == 2


def test_noop_action_single_agent_dict_uses_vector_shape():
    env = FakeEnv({})
    env.agents = {"agent0": object()}

    action = intersection_env._noop_action(env)

    assert hasattr(action, "shape")
    assert tuple(action.shape) == (2,)
