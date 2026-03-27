"""Spawn ambulance vehicles in MetaDrive."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

AMBULANCE_COLOR = (1.0, 0.0, 0.0)
DEFAULT_AMBULANCE_OBJ = "Models_G0403A078/ambulance.obj"


def _attach_custom_ambulance_mesh(env, vehicle) -> bool:
    """Best-effort custom visual override using local OBJ mesh."""
    obj_path = Path(os.getenv("AMBULANCE_OBJ_PATH", DEFAULT_AMBULANCE_OBJ)).expanduser()
    if not obj_path.is_absolute():
        obj_path = Path.cwd() / obj_path
    if not obj_path.exists():
        return False

    engine = getattr(env, "engine", None)
    loader = getattr(engine, "loader", None)
    if loader is None or not hasattr(loader, "loadModel"):
        return False

    try:
        model = loader.loadModel(str(obj_path))
    except Exception:
        return False
    if model is None:
        return False

    parent = getattr(vehicle, "origin", None)
    if parent is None:
        parent = getattr(vehicle, "origin_np", None)
    if parent is None:
        return False

    try:
        model.reparentTo(parent)
        model.setColor(1.0, 0.0, 0.0, 1.0)
        scale = float(os.getenv("AMBULANCE_MODEL_SCALE", "0.7"))
        model.setScale(scale)
        model.setPos(0.0, 0.0, float(os.getenv("AMBULANCE_MODEL_Z", "0.55")))
        model.setHpr(0.0, 0.0, 0.0)
        setattr(vehicle, "_ambulance_model_np", model)
    except Exception:
        return False

    return True


def spawn_ambulance(env, origin_direction: str) -> str:
    """Best-effort ambulance spawn.

    Falls back to a virtual id if spawn APIs are unavailable in the current env.
    """
    vehicle_id = f"AMB_{uuid.uuid4().hex[:8]}"

    try:
        # MetaDrive API can vary by version; use defensive calls.
        from metadrive.component.vehicle.vehicle_type import DefaultVehicle

        traffic_manager = getattr(env.engine, "traffic_manager", None)
        if traffic_manager and hasattr(traffic_manager, "spawn_object"):
            vehicle = traffic_manager.spawn_object(DefaultVehicle)
        elif hasattr(env.engine, "spawn_object"):
            vehicle = env.engine.spawn_object(DefaultVehicle)
        else:
            return vehicle_id

        if vehicle is not None:
            vehicle_id = str(getattr(vehicle, "id", vehicle_id))
            has_custom_mesh = _attach_custom_ambulance_mesh(env, vehicle)
            if not has_custom_mesh:
                if hasattr(vehicle, "set_color"):
                    vehicle.set_color(AMBULANCE_COLOR)
                elif hasattr(vehicle, "set_paint_color"):
                    vehicle.set_paint_color(AMBULANCE_COLOR)
            if hasattr(vehicle, "meta") and isinstance(vehicle.meta, dict):
                vehicle.meta["type"] = "ambulance"
                vehicle.meta["origin_direction"] = origin_direction
                vehicle.meta["custom_mesh"] = has_custom_mesh
    except Exception:
        return vehicle_id

    return vehicle_id
