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
    await runner.start()
    try:
        yield
    finally:
        await runner.stop()


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


@app.get("/health")
async def health() -> dict:
    return app.state.runner.status()


@app.post("/api/control")
async def control(req: ControlRequest) -> dict:
    runner: SimulationRunner = app.state.runner
    if req.action == "start":
        await runner.start()
    elif req.action == "pause":
        await runner.pause()
    elif req.action == "resume":
        await runner.resume()
    elif req.action == "stop":
        await runner.stop()
    elif req.action == "reset":
        await runner.reset()
    return runner.status()


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
        "nonce": event.nonce,
    }


@app.get("/api/kpi")
async def kpi() -> dict:
    runner: SimulationRunner = app.state.runner
    out = get_kpi_summary()
    out["step"] = runner.step
    out["running"] = runner.running
    out["paused"] = runner.paused
    return out


@app.get("/api/decisions")
async def decisions(limit: int = Query(default=50, ge=1, le=500)) -> list[dict]:
    return fetch_recent_decisions(limit=limit)


socket_app = socketio.ASGIApp(sio, app)
