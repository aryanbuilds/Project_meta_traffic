"""Deterministic FIX-4/5 verification harness for Windows CPU baseline."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv.detector import detect_zones, get_detector_health, reset_detector_runtime_for_tests
from envs.intersection_env import create_intersection_env, ensure_traffic_lights_ready, render_topdown_frame, save_test_frame
from main import SimulationRunner

PROFILE_DEFAULTS = {
    "SIM_SEED": "42",
    "SIM_ENV_MODE": "single",
    "TRAFFIC_DENSITY": "0.3",
    "PIPELINE_DEBUG": "1",
    "SIGNAL_DEBUG": "1",
    "AUTO_CYCLE_TRAFFIC_LIGHTS": "0",
}
WARMUP_STEPS = 12
TRIALS = 3
MIN_TRIAL_PASSES = 2
FIX5_STEPS = 50


@dataclass
class TrialResult:
    trial: int
    non_zero_zones: int
    zone_counts: dict[str, int]
    detector_ready: bool
    detector_error_code: str | None
    passed: bool


def apply_profile_defaults() -> None:
    for key, value in PROFILE_DEFAULTS.items():
        os.environ.setdefault(key, value)


def import_sanity() -> dict:
    try:
        import torch
        import torchvision
        import ultralytics

        return {
            "ok": True,
            "torch": getattr(torch, "__version__", "unknown"),
            "torchvision": getattr(torchvision, "__version__", "unknown"),
            "ultralytics": getattr(ultralytics, "__version__", "unknown"),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc),
        }


def _noop_action(env):
    agents = getattr(env, "agents", None)
    if isinstance(agents, dict) and agents:
        return {aid: np.asarray([0.0, 0.0], dtype=np.float32) for aid in agents.keys()}
    return np.asarray([0.0, 0.0], dtype=np.float32)


def run_fix4_trials() -> list[TrialResult]:
    results: list[TrialResult] = []
    for trial in range(1, TRIALS + 1):
        env = create_intersection_env({"use_render": False, "traffic_density": float(PROFILE_DEFAULTS["TRAFFIC_DENSITY"])})
        try:
            env.reset()
            ensure_traffic_lights_ready(env)
            for _ in range(WARMUP_STEPS):
                env.step(_noop_action(env))
            frame = render_topdown_frame(env)
            if trial == 1:
                save_test_frame(frame, "data/test_frame.png")
            detection = detect_zones(frame)
            counts = {d: int(len(detection.get(d, []))) for d in ("north", "south", "east", "west")}
            non_zero = sum(1 for value in counts.values() if value > 0)
            results.append(
                TrialResult(
                    trial=trial,
                    non_zero_zones=non_zero,
                    zone_counts=counts,
                    detector_ready=bool(detection.get("detector_ready", False)),
                    detector_error_code=detection.get("detector_error_code"),
                    passed=non_zero >= 2,
                )
            )
        finally:
            env.close()
    return results


async def run_fix5_checks() -> dict:
    runner = SimulationRunner(sio=None)
    await runner.start()
    try:
        while runner.running and runner.step < FIX5_STEPS:
            await asyncio.sleep(0.05)
        status = runner.status()
    finally:
        await runner.stop()

    checks = {
        "yolo_checkpoint_present": int(status.get("yolo_checkpoint_count", 0)) > 0,
        "pce_checkpoint_present": int(status.get("pce_checkpoint_count", 0)) > 0,
        "llm_seen": int(status.get("llm_success_count", 0)) > 0,
        "signal_apply_count_gt_zero": int((status.get("last_signal_apply") or {}).get("count", 0)) > 0,
        "no_unhandled_detector_exception": not bool(status.get("last_error")),
    }
    return {
        "status": status,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    apply_profile_defaults()
    reset_detector_runtime_for_tests()

    import_result = import_sanity()
    fix4_trials = run_fix4_trials() if import_result.get("ok", False) else []
    fix4_pass_count = sum(1 for t in fix4_trials if t.passed)
    fix4_passed = fix4_pass_count >= MIN_TRIAL_PASSES

    fix5_result = asyncio.run(run_fix5_checks()) if import_result.get("ok", False) else {
        "passed": False,
        "checks": {},
        "status": {},
    }

    summary = {
        "import_sanity": import_result,
        "fix4": {
            "warmup_steps": WARMUP_STEPS,
            "trials": [t.__dict__ for t in fix4_trials],
            "pass_count": fix4_pass_count,
            "required_passes": MIN_TRIAL_PASSES,
            "passed": fix4_passed,
        },
        "fix5": fix5_result,
        "detector_health": get_detector_health(),
    }
    print(json.dumps(summary, indent=2))

    overall_pass = bool(import_result.get("ok", False) and fix4_passed and fix5_result.get("passed", False))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
