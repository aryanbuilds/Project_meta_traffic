"""Safety layer for signal decisions."""

import logging

from agent.models import IntersectionState, SignalDecision
from agent.phase_utils import phase_for_direction, phase_serves_direction

logger = logging.getLogger(__name__)


def enforce_safety(decision: SignalDecision, state: IntersectionState) -> SignalDecision:
    safe = decision.model_copy(deep=True)

    original_duration = safe.duration_s
    safe.duration_s = max(15, min(60, safe.duration_s))
    if safe.duration_s != original_duration:
        logger.warning("Safety clamp duration_s: %s -> %s", original_duration, safe.duration_s)

    # Starvation override to a served phase.
    for direction, zone in state.zones.items():
        if zone.wait_s > 90 and zone.count > 0 and not phase_serves_direction(safe.phase, direction):
            old_phase = safe.phase
            safe.phase = phase_for_direction(direction)
            safe.duration_s = max(safe.duration_s, 20)
            safe.reasoning += f" | starvation_override={direction}"
            logger.warning(
                "Safety starvation override: phase %s -> %s (wait_s=%s, count=%s)",
                old_phase,
                safe.phase,
                zone.wait_s,
                zone.count,
            )
            break

    if (safe.emergency_detected or state.emergency_active) and safe.duration_s < 30:
        logger.warning("Safety emergency minimum duration enforced: %s -> 30", safe.duration_s)
        safe.duration_s = 30

    original_skip = list(safe.skip_phases)
    safe.skip_phases = [d for d in safe.skip_phases if state.zones.get(d) and state.zones[d].count == 0]
    if safe.skip_phases != original_skip:
        logger.info("Safety pruned skip_phases: %s -> %s", original_skip, safe.skip_phases)

    return safe
