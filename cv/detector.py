"""YOLO + simple zone assignment and ambulance color heuristic."""

from functools import lru_cache

import cv2
import numpy as np
from ultralytics import YOLO

from cv.zone_config import CENTER_MAX, CENTER_MIN, build_polygon_zones

try:
    import supervision as sv
except Exception:  # noqa: BLE001
    sv = None

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
    lower1 = np.array([0, 100, 50], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 100, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_ratio = float(np.count_nonzero(red_mask)) / float(mask1.size)
    return red_ratio > 0.2


@lru_cache(maxsize=8)
def _polygon_zones(width: int, height: int):
    return build_polygon_zones(width, height)


def _assign_by_supervision(frame: np.ndarray, boxes: np.ndarray, cls: np.ndarray) -> tuple[dict[str, list[int]], list[dict]]:
    zone_class_ids = {"north": [], "south": [], "east": [], "west": []}
    bboxes: list[dict] = []
    if sv is None or boxes.size == 0:
        return zone_class_ids, bboxes

    zones = _polygon_zones(int(frame.shape[1]), int(frame.shape[0]))
    if not zones:
        return zone_class_ids, bboxes

    detections = sv.Detections(xyxy=boxes, class_id=cls)
    assigned: set[int] = set()
    for direction, zone in zones.items():
        try:
            mask = zone.trigger(detections=detections)
        except TypeError:
            mask = zone.trigger(detections)
        if mask is None:
            continue
        indices = np.where(mask)[0]
        for idx in indices:
            if int(idx) in assigned:
                continue
            assigned.add(int(idx))
            cid = int(cls[idx])
            zone_class_ids[direction].append(cid)
            bboxes.append({"xyxy": boxes[idx].tolist(), "class_id": cid, "direction": direction})

    return zone_class_ids, bboxes


def detect_zones(frame: np.ndarray) -> dict:
    pred = _model()(frame, verbose=False)[0]
    zone_class_ids = {"north": [], "south": [], "east": [], "west": []}
    bboxes: list[dict] = []
    ambulance_detected = False
    ambulance_direction = None
    ambulance_scores = {"north": 0, "south": 0, "east": 0, "west": 0}

    if pred.boxes is None:
        return {
            **zone_class_ids,
            "ambulance_detected": False,
            "ambulance_direction": None,
            "bboxes": [],
        }

    boxes = pred.boxes.xyxy.cpu().numpy() if pred.boxes.xyxy is not None else np.empty((0, 4))
    cls = pred.boxes.cls.cpu().numpy().astype(int) if pred.boxes.cls is not None else np.empty((0,), dtype=int)
    if cls.size:
        keep_mask = np.isin(cls, list(VALID_CLASSES))
        boxes = boxes[keep_mask]
        cls = cls[keep_mask]

    # Preferred path: zone assignment via supervision PolygonZone.
    if boxes.size > 0 and sv is not None:
        zone_class_ids, bboxes = _assign_by_supervision(frame, boxes, cls)

    # Fallback path: center-point directional assignment.
    if not bboxes:
        for box, cid in zip(boxes, cls):
            if cid not in VALID_CLASSES:
                continue
            cx, cy = _center_of_box(box)
            direction = _direction_for_point(cx, cy)
            zone_class_ids[direction].append(int(cid))
            bboxes.append({"xyxy": box.tolist(), "class_id": int(cid), "direction": direction})

    for box_info in bboxes:
        box = np.asarray(box_info["xyxy"], dtype=float)
        cid = int(box_info["class_id"])
        direction = str(box_info["direction"])
        if cid not in VALID_CLASSES:
            continue
        if _is_red_region(frame, box):
            ambulance_detected = True
            ambulance_scores[direction] += 1

    if ambulance_detected:
        ambulance_direction = max(ambulance_scores.items(), key=lambda item: item[1])[0]

    return {
        **zone_class_ids,
        "ambulance_detected": ambulance_detected,
        "ambulance_direction": ambulance_direction,
        "bboxes": bboxes,
    }
