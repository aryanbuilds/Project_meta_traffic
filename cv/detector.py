"""YOLO + simple zone assignment and ambulance color heuristic."""

from functools import lru_cache

import cv2
import numpy as np
from ultralytics import YOLO

from cv.zone_config import CENTER_MAX, CENTER_MIN

VALID_CLASSES = {2, 3, 5, 7}


@lru_cache(maxsize=1)
def _model() -> YOLO:
    return YOLO("yolov8n.pt")


def _center_of_box(xyxy: np.ndarray) -> tuple[int, int]:
    x1, y1, x2, y2 = xyxy.astype(int)
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def _direction_for_point(x: int, y: int) -> str:
    if y < CENTER_MIN:
        return "north"
    if y > CENTER_MAX:
        return "south"
    if x < CENTER_MIN:
        return "west"
    if x > CENTER_MAX:
        return "east"
    # center box fallback
    return "north"


def _is_red_region(frame: np.ndarray, box: np.ndarray) -> bool:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return False

    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 100, 50), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 100, 50), (180, 255, 255))
    red_ratio = float(np.count_nonzero(mask1 | mask2)) / float(mask1.size)
    return red_ratio > 0.2


def detect_zones(frame: np.ndarray) -> dict:
    pred = _model()(frame, verbose=False)[0]
    zone_class_ids = {"north": [], "south": [], "east": [], "west": []}
    bboxes = []
    ambulance_detected = False
    ambulance_direction = None

    if pred.boxes is None:
        return {
            **zone_class_ids,
            "ambulance_detected": False,
            "ambulance_direction": None,
            "bboxes": [],
        }

    boxes = pred.boxes.xyxy.cpu().numpy() if pred.boxes.xyxy is not None else np.empty((0, 4))
    cls = pred.boxes.cls.cpu().numpy().astype(int) if pred.boxes.cls is not None else np.empty((0,), dtype=int)

    for box, cid in zip(boxes, cls):
        if cid not in VALID_CLASSES:
            continue
        cx, cy = _center_of_box(box)
        direction = _direction_for_point(cx, cy)
        zone_class_ids[direction].append(int(cid))
        bboxes.append({"xyxy": box.tolist(), "class_id": int(cid), "direction": direction})
        if _is_red_region(frame, box):
            ambulance_detected = True
            ambulance_direction = direction

    return {
        **zone_class_ids,
        "ambulance_detected": ambulance_detected,
        "ambulance_direction": ambulance_direction,
        "bboxes": bboxes,
    }
