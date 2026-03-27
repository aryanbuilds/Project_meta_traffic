"""FastAPI + Socket.IO backend for simulation control and streaming."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
import socketio

from data.logger import fetch_recent_decisions, get_kpi_summary, init_db
from main import SimulationRunner

load_dotenv()

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    runner = SimulationRunner(sio=sio)
    app.state.runner = runner
    app.state.startup_error = None

    try:
        await runner.start()
    except Exception as exc:  # noqa: BLE001
        # Keep API alive so operator can inspect status and retry via /api/control.
        app.state.startup_error = str(exc)

    try:
        yield
    finally:
        try:
            await runner.stop()
        except Exception:
            pass


app = FastAPI(title="NeuroSignal Backend", version="0.1.0", lifespan=lifespan)


class ControlRequest(BaseModel):
    action: Literal["start", "pause", "resume", "stop", "reset"]


class EmergencyRequest(BaseModel):
    ambulance_id: str = Field(min_length=3, max_length=32)
    destination: str = Field(min_length=3, max_length=64)
    origin_direction: Literal["north", "south", "east", "west"] = "north"


@sio.event
async def connect(sid, environ, auth):
    return True


@sio.event
async def disconnect(sid):
    return None


def _status_with_startup_error(app_obj: FastAPI) -> dict:
    status = app_obj.state.runner.status()
    if app_obj.state.startup_error:
        status["startup_error"] = app_obj.state.startup_error
    return status


@app.get("/health")
async def health() -> dict:
    return _status_with_startup_error(app)


@app.get("/api/status")
async def status() -> dict:
    return _status_with_startup_error(app)


@app.post("/api/control")
async def control(req: ControlRequest) -> dict:
    runner: SimulationRunner = app.state.runner
    if req.action == "start":
        await runner.start()
        app.state.startup_error = None
    elif req.action == "pause":
        await runner.pause()
    elif req.action == "resume":
        await runner.resume()
    elif req.action == "stop":
        await runner.stop()
    elif req.action == "reset":
        await runner.reset()
    return _status_with_startup_error(app)


@app.post("/api/emergency")
async def trigger_emergency(req: EmergencyRequest) -> dict:
    runner: SimulationRunner = app.state.runner
    event = await runner.event_handler.trigger_manual(
        ambulance_id=req.ambulance_id,
        destination=req.destination,
        origin_direction=req.origin_direction,
    )
    return {
        "queued": True,
        "ambulance_id": event.ambulance_id,
        "destination": event.destination,
        "origin_direction": event.origin_direction,
        "route": event.route,
        "route_source": event.route_source,
        "route_distance_km": event.route_distance_km,
        "nonce": event.nonce,
    }


@app.get("/api/kpi")
async def kpi() -> dict:
    runner: SimulationRunner = app.state.runner
    out = get_kpi_summary()
    out["step"] = runner.step
    out["running"] = runner.running
    out["paused"] = runner.paused
    if app.state.startup_error:
        out["startup_error"] = app.state.startup_error
    return out


@app.get("/api/decisions")
async def decisions(limit: int = Query(default=50, ge=1, le=500)) -> list[dict]:
    return fetch_recent_decisions(limit=limit)


socket_app = socketio.ASGIApp(sio, app)
