"""Rule-based fallback controller."""

import math

from agent.models import IntersectionState, SignalDecision
from agent.phase_utils import phase_for_direction


def rule_based_decision(state: IntersectionState) -> SignalDecision:
    best_dir = "north"
    best_score = -1.0
    best_pce = 0.0
    best_tw = 0.0
    best_wait = 0

    for direction, zone in state.zones.items():
        score = zone.pce * math.log1p(zone.wait_s) if zone.count > 0 else 0.0
        if score > best_score:
            best_score = score
            best_dir = direction
            best_pce = zone.pce
            best_tw = zone.tw_ratio
            best_wait = zone.wait_s

    duration = int(15 + 1.5 * best_pce + (5 if best_tw > 0.6 else 0))
    duration = max(15, min(60, duration))

    skip_phases = [d for d, z in state.zones.items() if z.count == 0]

    emergency_direction = None
    emergency_detected = False
    if state.emergency_active:
        emergency_detected = True
        if state.emergency_direction:
            emergency_direction = state.emergency_direction
            best_dir = state.emergency_direction
            duration = max(duration, 30)

    reasoning = (
        f"Rule-based: dir={best_dir}, score={best_score:.2f}, "
        f"PCE={best_pce:.1f}, wait={best_wait}s"
    )

    return SignalDecision(
        phase=phase_for_direction(best_dir),
        duration_s=duration,
        skip_phases=skip_phases,
        emergency_detected=emergency_detected,
        emergency_direction=emergency_direction,
        reasoning=reasoning,
    )
