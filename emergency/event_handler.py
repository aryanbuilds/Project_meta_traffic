"""Event ingestion/debounce for emergency corridor activation."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass

from routing.router import get_demo_route


VALID_DIRECTIONS = {"north", "south", "east", "west"}


@dataclass(slots=True)
class AmbulanceEvent:
    ambulance_id: str
    origin_direction: str
    destination: str
    route: list[str]
    nonce: str
    timestamp: float


class AmbulanceEventHandler:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[AmbulanceEvent] = asyncio.Queue()
        self.detection_streak = 0
        self.corridor_active = False
        self.required_streak = max(1, int(os.getenv("EMERGENCY_DEBOUNCE_FRAMES", "3")))
        self.cooldown_s = max(0.0, float(os.getenv("EMERGENCY_EVENT_COOLDOWN_S", "5")))
        self._last_event_ts = 0.0

    async def on_cv_frame(self, ambulance_detected: bool, ambulance_direction: str | None) -> AmbulanceEvent | None:
        if ambulance_detected:
            self.detection_streak += 1
        else:
            self.detection_streak = 0
            return None

        now = time.time()
        in_cooldown = (now - self._last_event_ts) < self.cooldown_s
        if self.detection_streak >= self.required_streak and not self.corridor_active and not in_cooldown:
            self.detection_streak = 0
            event = await self._make_event(
                ambulance_id="AMB_001",
                origin_direction=ambulance_direction or "north",
                destination="AIIMS",
            )
            self._last_event_ts = now
            await self.queue.put(event)
            return event
        return None

    async def trigger_manual(self, ambulance_id: str, destination: str, origin_direction: str = "north") -> AmbulanceEvent:
        event = await self._make_event(
            ambulance_id=ambulance_id,
            origin_direction=origin_direction,
            destination=destination,
        )
        self._last_event_ts = time.time()
        await self.queue.put(event)
        return event

    async def next_event(self) -> AmbulanceEvent:
        return await self.queue.get()

    async def _make_event(self, ambulance_id: str, origin_direction: str, destination: str) -> AmbulanceEvent:
        direction = (origin_direction or "north").strip().lower()
        if direction not in VALID_DIRECTIONS:
            direction = "north"
        safe_destination = (destination or "AIIMS").strip() or "AIIMS"

        route = get_demo_route(safe_destination)
        return AmbulanceEvent(
            ambulance_id=(ambulance_id or "AMB_001").strip() or "AMB_001",
            origin_direction=direction,
            destination=safe_destination,
            route=route,
            nonce=uuid.uuid4().hex,
            timestamp=time.time(),
        )
