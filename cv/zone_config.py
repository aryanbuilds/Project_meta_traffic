"""Static polygon zones for top-down frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import supervision as sv
except Exception:  # noqa: BLE001
    sv = None

FRAME_SIZE = 800
CENTER_MIN = 300
CENTER_MAX = 500

NORTH_ZONE = np.array([[0, 0], [800, 0], [800, CENTER_MIN], [0, CENTER_MIN]], dtype=np.int32)
SOUTH_ZONE = np.array([[0, CENTER_MAX], [800, CENTER_MAX], [800, 800], [0, 800]], dtype=np.int32)
WEST_ZONE = np.array([[0, 0], [CENTER_MIN, 0], [CENTER_MIN, 800], [0, 800]], dtype=np.int32)
EAST_ZONE = np.array([[CENTER_MAX, 0], [800, 0], [800, 800], [CENTER_MAX, 800]], dtype=np.int32)

ZONE_POLYGONS = {
    "north": NORTH_ZONE,
    "south": SOUTH_ZONE,
    "east": EAST_ZONE,
    "west": WEST_ZONE,
}


def build_polygon_zones(frame_width: int = FRAME_SIZE, frame_height: int = FRAME_SIZE) -> dict[str, Any]:
    if sv is None:
        return {}
    resolution = (int(frame_width), int(frame_height))
    zones: dict[str, Any] = {}
    polygon_zone_cls = getattr(sv, "PolygonZone", None)
    if polygon_zone_cls is None:
        return zones
    for name, poly in ZONE_POLYGONS.items():
        poly64 = np.asarray(poly, dtype=np.int64)
        try:
            zones[name] = polygon_zone_cls(polygon=poly64, frame_resolution_wh=resolution)
        except Exception:
            zones[name] = polygon_zone_cls(poly64, resolution)
    return zones


def save_zones_debug_image(
    out_path: str = "data/zones_debug.png",
    frame_size: int = FRAME_SIZE,
    background_bgr: tuple[int, int, int] = (28, 28, 28),
) -> str:
    canvas = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
    canvas[:, :] = background_bgr
    colors = {
        "north": (255, 0, 0),
        "south": (0, 255, 0),
        "east": (0, 165, 255),
        "west": (255, 0, 255),
    }
    for name, poly in ZONE_POLYGONS.items():
        cv2.polylines(canvas, [poly], True, colors[name], 2)
        x, y, w, h = cv2.boundingRect(poly)
        cv2.putText(
            canvas,
            name.upper(),
            (x + max(4, w // 4), y + max(18, h // 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            colors[name],
            2,
        )

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), canvas)
    if not ok:
        raise RuntimeError(f"Failed to write zone debug image to {path}")
    return str(path)
