import numpy as np

from cv import detector


def test_detector_failure_contract_keeps_schema(monkeypatch):
    detector.reset_detector_runtime_for_tests()

    def _boom(self):
        raise OSError("[WinError 1114] A dynamic link library initialization routine failed: c10.dll")

    monkeypatch.setattr(detector._DetectorRuntime, "_ensure_model", _boom)
    monkeypatch.setattr(detector._DetectorRuntime, "_start_worker", lambda self, reason: False)

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    out = detector.detect_zones(frame)

    assert set(("north", "south", "east", "west", "ambulance_detected", "ambulance_direction", "bboxes")).issubset(
        out.keys()
    )
    assert out["north"] == []
    assert out["south"] == []
    assert out["east"] == []
    assert out["west"] == []
    assert out["ambulance_detected"] is False
    assert out["ambulance_direction"] is None
    assert out["detector_ready"] is False
    assert out["detector_error_code"] == "TORCH_DLL_INIT_FAILED"
    assert isinstance(out["detector_error_message"], str)


def test_detector_health_includes_observability_fields(monkeypatch):
    detector.reset_detector_runtime_for_tests()

    def _boom(self):
        raise RuntimeError("model load failed")

    monkeypatch.setattr(detector._DetectorRuntime, "_ensure_model", _boom)
    monkeypatch.setattr(detector._DetectorRuntime, "_start_worker", lambda self, reason: False)

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    detector.detect_zones(frame)
    health = detector.get_detector_health()

    assert "detector_ready" in health
    assert "last_detector_error" in health
    assert "detector_worker_mode" in health
    assert "detector_worker_alive" in health
    assert "detector_total_attempts" in health
    assert "detector_total_failures" in health
