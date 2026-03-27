from main import SimulationRunner


def test_runner_status_exposes_llm_signal_and_detector_health(monkeypatch):
    monkeypatch.setattr(
        "main.get_detector_health",
        lambda: {
            "detector_ready": True,
            "last_detector_error": {"code": None, "message": ""},
            "detector_worker_mode": False,
            "detector_worker_alive": False,
            "detector_total_attempts": 3,
            "detector_total_failures": 0,
            "detector_startup_winerror_count": 0,
            "detector_early_failures": 0,
        },
    )
    runner = SimulationRunner(sio=None)
    runner.llm_success_count = 2
    runner.llm_fallback_count = 1
    runner.last_signal_apply = {"count": 4, "unknown_direction_count": 0}

    status = runner.status()

    assert status["llm_success_count"] == 2
    assert status["llm_fallback_count"] == 1
    assert status["last_signal_apply"]["count"] == 4
    assert status["detector_ready"] is True
    assert status["detector_total_attempts"] == 3
