"""Socket.IO broadcaster for simulation events."""

from __future__ import annotations

import time
from typing import Any

from cv.annotator import to_base64_jpeg


class SimBroadcaster:
    def __init__(self, sio, fps_cap: int = 10, kpi_interval_s: float = 2.0) -> None:
        self.sio = sio
        self.frame_interval_s = 1.0 / max(1, fps_cap)
        self.kpi_interval_s = max(0.5, float(kpi_interval_s))
        self._last_frame_emit = 0.0
        self._last_kpi_emit = 0.0

    async def emit_frame(self, jpeg_bytes: bytes, step: int) -> None:
        now = time.monotonic()
        if now - self._last_frame_emit < self.frame_interval_s:
            return
        self._last_frame_emit = now
        await self.sio.emit(
            "frame",
            {
                "schema_version": 1,
                "step": step,
                "jpeg_base64": to_base64_jpeg(jpeg_bytes),
                "timestamp": time.time(),
            },
        )

    async def emit_decision(
        self,
        *,
        step: int,
        decision: dict[str, Any],
        controller_type: str,
        latency_ms: float,
    ) -> None:
        await self.sio.emit(
            "decision",
            {
                "schema_version": 1,
                "step": step,
                "controller_type": controller_type,
                "latency_ms": round(float(latency_ms), 2),
                "decision": decision,
                "timestamp": time.time(),
            },
        )

    async def emit_zones(self, zone_pce: dict[str, dict], step: int) -> None:
        await self.sio.emit(
            "zones",
            {
                "schema_version": 1,
                "step": step,
                "zones": zone_pce,
                "timestamp": time.time(),
            },
        )

    async def emit_kpi(self, kpi: dict[str, Any], step: int) -> None:
        now = time.monotonic()
        if now - self._last_kpi_emit < self.kpi_interval_s:
            return
        self._last_kpi_emit = now
        await self.sio.emit(
            "kpi",
            {
                "schema_version": 1,
                "step": step,
                "kpi": kpi,
                "timestamp": time.time(),
            },
        )

    async def emit_emergency(self, payload: dict[str, Any]) -> None:
        body = {
            "schema_version": 1,
            "timestamp": time.time(),
        }
        body.update(payload)
        await self.sio.emit("emergency", body)
