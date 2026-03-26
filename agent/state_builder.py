"""Build text prompt from structured intersection state."""

from agent.models import IntersectionState
from agent.prompts import EMERGENCY_CONTEXT_TEMPLATE


def _recommended_green(pce: float, tw_ratio: float) -> int:
    base = 15 + 1.5 * pce + (5 if tw_ratio > 0.6 else 0)
    return int(max(15, min(60, base)))


def build_state_prompt(state: IntersectionState) -> str:
    lines = [f"Junction {state.junction_id} | Step {state.step}", "Approach states:"]
    for direction in ("north", "south", "east", "west"):
        zone = state.zones[direction]
        flags: list[str] = []
        if zone.wait_s > 90:
            flags.append(f"STARVATION WARNING: {zone.wait_s}s")
        if zone.ambulance_detected:
            flags.append("AMBULANCE DETECTED")
        if zone.count == 0:
            flags.append("SKIP - EMPTY")

        flags_text = f" [{' | '.join(flags)}]" if flags else ""
        lines.append(
            f"  {direction.upper()}: {zone.count} vehicles, "
            f"PCE={zone.pce:.1f}, wait={zone.wait_s}s, "
            f"bikes={zone.tw_ratio * 100:.0f}%, "
            f"recommended_green={_recommended_green(zone.pce, zone.tw_ratio)}s"
            f"{flags_text}"
        )

    if state.emergency_active:
        route = state.emergency_route or []
        route_text = " -> ".join(route) if route else "unknown"
        direction = state.emergency_direction or "north"
        destination = state.emergency_destination or "UNKNOWN_DEST"
        ambulance_id = state.emergency_ambulance_id or "AMB_001"
        downstream = ", ".join(route[1:]) if len(route) > 1 else "none"
        lines.append(
            EMERGENCY_CONTEXT_TEMPLATE.format(
                ambulance_id=ambulance_id,
                direction=direction,
                route_text=route_text,
                destination=destination,
                downstream=downstream,
            )
        )

    return "\n".join(lines)
