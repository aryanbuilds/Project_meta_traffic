"""Spawn ambulance vehicles in MetaDrive."""

from __future__ import annotations

import uuid


AMBULANCE_COLOR = (1.0, 0.0, 0.0)


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
            if hasattr(vehicle, "set_color"):
                vehicle.set_color(AMBULANCE_COLOR)
            elif hasattr(vehicle, "set_paint_color"):
                vehicle.set_paint_color(AMBULANCE_COLOR)
            if hasattr(vehicle, "meta") and isinstance(vehicle.meta, dict):
                vehicle.meta["type"] = "ambulance"
                vehicle.meta["origin_direction"] = origin_direction
    except Exception:
        return vehicle_id

    return vehicle_id
