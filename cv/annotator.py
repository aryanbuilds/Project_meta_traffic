"""Frame annotation utilities for dashboard streaming."""

import base64

import cv2
import numpy as np

from cv.zone_config import EAST_ZONE, NORTH_ZONE, SOUTH_ZONE, WEST_ZONE

ZONE_COLORS = {
    "north": (255, 0, 0),
    "south": (0, 255, 0),
    "east": (0, 165, 255),
    "west": (255, 0, 255),
}


def _phase_serves(phase: str, direction: str) -> bool:
    if phase == "north_south":
        return direction in {"north", "south"}
    if phase == "east_west":
        return direction in {"east", "west"}
    return phase == direction


def _draw_signal_badges(image: np.ndarray, decision: dict | None) -> None:
    if not decision:
        return
    phase = str(decision.get("phase", "north_south"))
    emergency = bool(decision.get("emergency_detected", False))
    yellow_mode = bool(decision.get("yellow_phase", False))

    positions = {
        "north": (image.shape[1] // 2 - 45, 26),
        "south": (image.shape[1] // 2 - 45, image.shape[0] - 18),
        "east": (image.shape[1] - 96, image.shape[0] // 2),
        "west": (6, image.shape[0] // 2),
    }
    for direction in ("north", "south", "east", "west"):
        if yellow_mode:
            text = "YELLOW"
            color = (0, 200, 255)
        else:
            green = _phase_serves(phase, direction)
            text = "GREEN" if green else "RED"
            color = (0, 190, 0) if green else (0, 0, 230)
        if emergency and _phase_serves(phase, direction):
            text = "EMERG"
            color = (0, 0, 255)
        x, y = positions[direction]
        cv2.rectangle(image, (x, y - 16), (x + 88, y + 8), (30, 30, 30), -1)
        cv2.rectangle(image, (x, y - 16), (x + 88, y + 8), color, 2)
        cv2.putText(image, f"{direction[0].upper()}:{text}", (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def _draw_zones(image: np.ndarray) -> None:
    cv2.polylines(image, [NORTH_ZONE], True, ZONE_COLORS["north"], 2)
    cv2.polylines(image, [SOUTH_ZONE], True, ZONE_COLORS["south"], 2)
    cv2.polylines(image, [EAST_ZONE], True, ZONE_COLORS["east"], 2)
    cv2.polylines(image, [WEST_ZONE], True, ZONE_COLORS["west"], 2)


def annotate_frame(frame: np.ndarray, detection_result: dict, zone_pce: dict | None = None, decision: dict | None = None) -> bytes:
    zone_pce = zone_pce or {}
    image = frame.copy()

    _draw_zones(image)

    for box in detection_result.get("bboxes", []):
        x1, y1, x2, y2 = map(int, box["xyxy"])
        color = ZONE_COLORS.get(box["direction"], (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{box['class_id']}:{box['direction']}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    y = 20
    for direction in ("north", "south", "east", "west"):
        pce = zone_pce.get(direction, {}).get("total_pce", 0)
        cv2.putText(image, f"{direction.upper()} PCE={pce}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ZONE_COLORS[direction], 2)
        y += 24

    if decision:
        cv2.putText(
            image,
            f"PHASE={decision.get('phase')} DURATION={decision.get('duration_s')}s",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        _draw_signal_badges(image, decision)

    if detection_result.get("ambulance_detected"):
        cv2.rectangle(image, (0, 0), (image.shape[1] - 1, image.shape[0] - 1), (0, 0, 255), 8)
        cv2.putText(image, "CORRIDOR ACTIVE", (240, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()


def to_base64_jpeg(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")
