from agent.models import SignalDecision
from agent.signal_controller import _light_direction, apply_decision


class FakeLane:
    def __init__(self, heading):
        self._heading = heading

    def heading_theta_at(self, longitudinal):
        return self._heading


class FakeLight:
    def __init__(self, **attrs):
        self.status = attrs.pop("status", "red")
        self.__dict__.update(attrs)

    def set_green(self):
        self.status = "green"

    def set_red(self):
        self.status = "red"

    def set_yellow(self):
        self.status = "yellow"


class FakeTrafficManager:
    def __init__(self, lights):
        self.traffic_lights = lights


class FakeEngine:
    def __init__(self, lights):
        self.traffic_manager = FakeTrafficManager(lights)


class FakeEnv:
    def __init__(self, lights):
        self.engine = FakeEngine(lights)


def test_light_direction_prefers_explicit_attribute():
    light = FakeLight(direction="west")

    assert _light_direction("unhelpful-id", light) == "west"


def test_light_direction_uses_lane_heading_when_attribute_missing():
    light = FakeLight(lane=FakeLane(1.57))

    assert _light_direction("unhelpful-id", light) == "north"


def test_apply_decision_debug_payload_includes_targets_and_directions():
    lights = {
        "north_light": FakeLight(direction="north"),
        "east_light": FakeLight(lane=FakeLane(0.0)),
    }
    env = FakeEnv(lights)
    decision = SignalDecision(
        phase="north_south",
        duration_s=30,
        reasoning="A deterministic explanation for the selected signal phase.",
    )

    payload = apply_decision(env, decision, step=7, debug=True)

    assert payload["step"] == 7
    assert payload["count"] == 2
    assert payload["green_count"] == 1
    assert payload["red_count"] == 1
    assert payload["lights"][0]["resolved_direction"] == "north"
    assert payload["lights"][0]["applied_target"] == "green"
    assert payload["lights"][1]["resolved_direction"] == "east"
    assert payload["lights"][1]["applied_target"] == "red"


def test_apply_decision_can_force_yellow_and_all_red_modes():
    lights = {
        "north_light": FakeLight(direction="north"),
        "east_light": FakeLight(direction="east"),
    }
    env = FakeEnv(lights)
    decision = SignalDecision(
        phase="north_south",
        duration_s=30,
        reasoning="A deterministic explanation for scheduled control mode testing.",
    )

    yellow_payload = apply_decision(env, decision, step=9, control_mode="yellow_all")
    assert yellow_payload["yellow_count"] == 2
    assert lights["north_light"].status == "yellow"
    assert lights["east_light"].status == "yellow"

    red_payload = apply_decision(env, decision, step=10, control_mode="all_red")
    assert red_payload["red_count"] == 2
    assert lights["north_light"].status == "red"
    assert lights["east_light"].status == "red"
