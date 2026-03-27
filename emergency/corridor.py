"""Green corridor coordinator for emergency events."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from agent.models import SignalDecision
from agent.signal_controller import apply_decision
from data.logger import log_corridor_event
from envs.ambulance_spawner import spawn_ambulance
from routing.router import route_to_prompt_text


@dataclass(slots=True)
class CorridorState:
    active: bool = False
    route: list[str] | None = None
    remaining_s: int = 0
    ambulance_id: str | None = None


class CorridorCoordinator:
    def __init__(self) -> None:
        self.state = CorridorState()

    async def activate_corridor(self, event, env, sio=None, hold_s: int = 30) -> CorridorState:
        if self.state.active:
            return self.state

        self.state.active = True
        self.state.route = list(event.route)
        self.state.remaining_s = hold_s
        self.state.ambulance_id = event.ambulance_id

        spawn_ambulance(env, event.origin_direction)

        emergency_decision = SignalDecision(
            phase=event.origin_direction,
            duration_s=max(30, hold_s),
            skip_phases=[],
            emergency_detected=True,
            emergency_direction=event.origin_direction,
            reasoning=(
                f"Emergency corridor active for {event.ambulance_id}, "
                f"route={route_to_prompt_text(event.route, event.destination)} "
                f"source={getattr(event, 'route_source', 'demo')}"
            ),
        )

        apply_decision(env, emergency_decision, step=0)

        if sio is not None:
            await sio.emit(
                "emergency",
                {
                    "active": True,
                    "route": event.route,
                    "remaining_s": self.state.remaining_s,
                    "ambulance_id": event.ambulance_id,
                    "route_source": getattr(event, "route_source", "demo"),
                    "route_distance_km": getattr(event, "route_distance_km", None),
                    "active_junction": event.route[0] if event.route else None,
                },
            )

        route = list(event.route) if event.route else ["J0"]
        for remaining in range(hold_s, 0, -1):
            self.state.remaining_s = remaining
            route_idx = (hold_s - remaining) % max(1, len(route))
            active_junction = route[route_idx]
            apply_decision(env, emergency_decision, step=(hold_s - remaining))
            if sio is not None:
                await sio.emit(
                    "emergency",
                    {
                        "active": True,
                        "route": event.route,
                        "remaining_s": remaining,
                        "ambulance_id": event.ambulance_id,
                        "active_junction": active_junction,
                        "route_source": getattr(event, "route_source", "demo"),
                        "route_distance_km": getattr(event, "route_distance_km", None),
                    },
                )
            await asyncio.sleep(1)

        log_corridor_event(
            ambulance_id=event.ambulance_id,
            destination=event.destination,
            origin_direction=event.origin_direction,
            route_text=route_to_prompt_text(event.route, event.destination),
            duration_s=hold_s,
        )

        self.state.active = False
        self.state.remaining_s = 0
        if sio is not None:
            await sio.emit("emergency", {"active": False, "route": event.route, "remaining_s": 0})

        return self.state
