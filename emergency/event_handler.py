"""Event ingestion/debounce for emergency corridor activation."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

from routing.router import get_demo_route


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

    async def on_cv_frame(self, ambulance_detected: bool, ambulance_direction: str | None) -> AmbulanceEvent | None:
        if ambulance_detected:
            self.detection_streak += 1
        else:
            self.detection_streak = 0
            return None

        if self.detection_streak >= 3 and not self.corridor_active:
            self.detection_streak = 0
            event = await self._make_event(
                ambulance_id="AMB_001",
                origin_direction=ambulance_direction or "north",
                destination="AIIMS",
            )
            await self.queue.put(event)
            return event
        return None

    async def trigger_manual(self, ambulance_id: str, destination: str, origin_direction: str = "north") -> AmbulanceEvent:
        event = await self._make_event(
            ambulance_id=ambulance_id,
            origin_direction=origin_direction,
            destination=destination,
        )
        await self.queue.put(event)
        return event

    async def next_event(self) -> AmbulanceEvent:
        return await self.queue.get()

    async def _make_event(self, ambulance_id: str, origin_direction: str, destination: str) -> AmbulanceEvent:
        route = get_demo_route(destination)
        return AmbulanceEvent(
            ambulance_id=ambulance_id,
            origin_direction=origin_direction,
            destination=destination,
            route=route,
            nonce=uuid.uuid4().hex,
            timestamp=time.time(),
        )
