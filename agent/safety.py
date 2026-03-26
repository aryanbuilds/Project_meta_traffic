"""Safety layer for signal decisions."""

from agent.models import IntersectionState, SignalDecision
from agent.phase_utils import phase_for_direction, phase_serves_direction


def enforce_safety(decision: SignalDecision, state: IntersectionState) -> SignalDecision:
    safe = decision.model_copy(deep=True)

    safe.duration_s = max(15, min(60, safe.duration_s))

    # Starvation override to a served phase.
    for direction, zone in state.zones.items():
        if zone.wait_s > 90 and zone.count > 0 and not phase_serves_direction(safe.phase, direction):
            safe.phase = phase_for_direction(direction)
            safe.duration_s = max(safe.duration_s, 20)
            safe.reasoning += f" | starvation_override={direction}"
            break

    if (safe.emergency_detected or state.emergency_active) and safe.duration_s < 30:
        safe.duration_s = 30

    safe.skip_phases = [
        d for d in safe.skip_phases
        if state.zones.get(d) and state.zones[d].count == 0
    ]
    return safe
