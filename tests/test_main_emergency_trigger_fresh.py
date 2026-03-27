import asyncio

from main import SimulationRunner


class _StubEventHandler:
    def __init__(self):
        self.calls = []

    async def on_cv_frame(self, detected: bool, direction: str | None):
        self.calls.append((detected, direction))


def test_emergency_detection_ignores_stale_cv_frames():
    runner = SimulationRunner(sio=None)
    stub = _StubEventHandler()
    runner.event_handler = stub

    asyncio.run(
        runner._handle_emergency_detection(
            {
                "ambulance_detected": True,
                "ambulance_direction": "north",
            },
            fresh=False,
        )
    )
    assert stub.calls == []

    asyncio.run(
        runner._handle_emergency_detection(
            {
                "ambulance_detected": True,
                "ambulance_direction": "north",
            },
            fresh=True,
        )
    )
    assert stub.calls == [(True, "north")]