"""YOLO detection + zone assignment with resilient runtime fallback."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from cv.zone_config import CENTER_MAX, CENTER_MIN, build_polygon_zones

try:
    import supervision as sv
except Exception:  # noqa: BLE001
    sv = None

VALID_CLASSES = {2, 3, 5, 7}
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
STARTUP_WINERROR_THRESHOLD = max(1, int(os.getenv("DETECTOR_WINERROR_THRESHOLD", "2")))
EARLY_FAILURE_THRESHOLD = max(1, int(os.getenv("DETECTOR_EARLY_FAILURE_THRESHOLD", "3")))
EARLY_FAILURE_WINDOW = max(1, int(os.getenv("DETECTOR_EARLY_FAILURE_WINDOW", "20")))
WORKER_TIMEOUT_S = max(0.5, float(os.getenv("DETECTOR_WORKER_TIMEOUT_S", "4.0")))


@lru_cache(maxsize=8)
def _polygon_zones(width: int, height: int):
    return build_polygon_zones(width, height)


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


def _classify_error(exc: Exception) -> tuple[str, str, bool]:
    text = str(exc)
    lowered = text.lower()
    if "winerror 1114" in lowered or "c10.dll" in lowered:
        return "TORCH_DLL_INIT_FAILED", text[:300], True
    if "yolo" in lowered or "model" in lowered:
        return "MODEL_LOAD_FAILED", text[:300], False
    return "INFERENCE_FAILED", text[:300], False


def _empty_detection(
    *,
    detector_ready: bool,
    error_code: str | None,
    error_message: str,
    worker_mode: bool,
    worker_alive: bool,
    use_polygon_zones: bool = True,
) -> dict[str, Any]:
    return {
        "north": [],
        "south": [],
        "east": [],
        "west": [],
        "ambulance_detected": False,
        "ambulance_direction": None,
        "bboxes": [],
        "detector_ready": detector_ready,
        "detector_error_code": error_code,
        "detector_error_message": error_message,
        "detector_worker_mode": worker_mode,
        "detector_worker_alive": worker_alive,
    }


def _predict_with_model(model: Any, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = model(frame, verbose=False)[0]
    if pred.boxes is None:
        return np.empty((0, 4), dtype=float), np.empty((0,), dtype=int)

    boxes = pred.boxes.xyxy.cpu().numpy() if pred.boxes.xyxy is not None else np.empty((0, 4), dtype=float)
    cls = pred.boxes.cls.cpu().numpy().astype(int) if pred.boxes.cls is not None else np.empty((0,), dtype=int)
    if cls.size:
        keep_mask = np.isin(cls, list(VALID_CLASSES))
        boxes = boxes[keep_mask]
        cls = cls[keep_mask]
    return boxes, cls


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


def _fallback_boxes_from_frame(frame: np.ndarray, max_boxes: int = 150) -> np.ndarray:
    """Extract candidate vehicle boxes from stylized top-down frames when YOLO sees nothing."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((sat > 45) & (val > 45)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[list[float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < 20 or area > 1800:
            continue
        if w < 3 or h < 3 or w > 100 or h > 100:
            continue
        boxes.append([float(x), float(y), float(x + w), float(y + h)])
        if len(boxes) >= max_boxes:
            break
    if not boxes:
        return np.empty((0, 4), dtype=float)
    return np.asarray(boxes, dtype=float)

def _build_detection_output(
    frame: np.ndarray,
    boxes: np.ndarray,
    cls: np.ndarray,
    *,
    detector_ready: bool,
    error_code: str | None,
    error_message: str,
    worker_mode: bool,
    worker_alive: bool,
    use_polygon_zones: bool = True,
) -> dict[str, Any]:
    zone_class_ids = {"north": [], "south": [], "east": [], "west": []}
    bboxes: list[dict] = []

    if boxes.size > 0 and sv is not None and use_polygon_zones:
        zone_class_ids, bboxes = _assign_by_supervision(frame, boxes, cls)

    if not bboxes:
        if boxes.size == 0:
            boxes = _fallback_boxes_from_frame(frame)
            cls = np.full((len(boxes),), 2, dtype=int)
        for box, cid in zip(boxes, cls):
            if int(cid) not in VALID_CLASSES:
                continue
            cx, cy = _center_of_box(np.asarray(box, dtype=float))
            direction = _direction_for_point(cx, cy)
            zone_class_ids[direction].append(int(cid))
            bboxes.append({"xyxy": np.asarray(box, dtype=float).tolist(), "class_id": int(cid), "direction": direction})

    ambulance_detected = False
    ambulance_direction = None
    ambulance_scores = {"north": 0, "south": 0, "east": 0, "west": 0}

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
        "detector_ready": detector_ready,
        "detector_error_code": error_code,
        "detector_error_message": error_message,
        "detector_worker_mode": worker_mode,
        "detector_worker_alive": worker_alive,
    }


def _worker_main(in_q: Any, out_q: Any, model_path: str) -> None:
    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        out_q.put({"kind": "startup", "ok": True})
    except Exception as exc:  # noqa: BLE001
        code, message, _ = _classify_error(exc)
        out_q.put({"kind": "startup", "ok": False, "error_code": code, "error_message": message})
        return

    while True:
        job = in_q.get()
        if job is None:
            return
        req_id = int(job.get("req_id", 0))
        frame_bytes = job.get("frame_jpg", b"")
        try:
            arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Failed to decode frame payload in detector worker")
            boxes, cls = _predict_with_model(model, frame)
            out_q.put(
                {
                    "kind": "result",
                    "req_id": req_id,
                    "ok": True,
                    "boxes": boxes.tolist(),
                    "cls": cls.tolist(),
                }
            )
        except Exception as exc:  # noqa: BLE001
            code, message, _ = _classify_error(exc)
            out_q.put(
                {
                    "kind": "result",
                    "req_id": req_id,
                    "ok": False,
                    "error_code": code,
                    "error_message": message,
                }
            )


class _DetectorRuntime:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.detector_ready = False
        self.last_error_code: str | None = None
        self.last_error_message = ""
        self.total_attempts = 0
        self.total_failures = 0
        self.startup_winerror_count = 0
        self.early_failures = 0
        self.worker_mode = False
        self.worker_failover_reason: str | None = None
        self._ctx = mp.get_context("spawn")
        self._worker_process: Any | None = None
        self._in_q: Any | None = None
        self._out_q: Any | None = None
        self._request_id = 0

    def _record_success(self) -> None:
        self.detector_ready = True
        self.last_error_code = None
        self.last_error_message = ""

    def _record_error(self, code: str, message: str, is_winerror_1114: bool) -> None:
        self.detector_ready = False
        self.total_failures += 1
        self.last_error_code = code
        self.last_error_message = message
        if is_winerror_1114:
            self.startup_winerror_count += 1
        if self.total_attempts <= EARLY_FAILURE_WINDOW:
            self.early_failures += 1

    def _should_activate_worker(self) -> bool:
        if self.startup_winerror_count >= STARTUP_WINERROR_THRESHOLD:
            return True
        return self.total_attempts <= EARLY_FAILURE_WINDOW and self.early_failures >= EARLY_FAILURE_THRESHOLD

    def _worker_alive(self) -> bool:
        return bool(self._worker_process is not None and self._worker_process.is_alive())

    def _start_worker(self, reason: str) -> bool:
        if self._worker_alive():
            self.worker_mode = True
            return True

        try:
            self._in_q = self._ctx.Queue(maxsize=2)
            self._out_q = self._ctx.Queue(maxsize=4)
            self._worker_process = self._ctx.Process(target=_worker_main, args=(self._in_q, self._out_q, MODEL_PATH), daemon=True)
            self._worker_process.start()
            startup = self._out_q.get(timeout=WORKER_TIMEOUT_S)
            if not startup.get("ok", False):
                self._record_error(
                    str(startup.get("error_code") or "MODEL_LOAD_FAILED"),
                    str(startup.get("error_message") or "Worker startup failed"),
                    str(startup.get("error_code")) == "TORCH_DLL_INIT_FAILED",
                )
                return False
            self.worker_mode = True
            self.worker_failover_reason = reason
            return True
        except Exception as exc:  # noqa: BLE001
            code, message, is_win = _classify_error(exc)
            self._record_error(code, message, is_win)
            return False

    def _ensure_model(self) -> Any:
        if self.model is not None:
            return self.model
        from ultralytics import YOLO

        self.model = YOLO(MODEL_PATH)
        return self.model

    def _infer_worker(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._worker_alive() and not self._start_worker("worker_restart"):
            raise RuntimeError("Detector worker unavailable")
        if self._in_q is None or self._out_q is None:
            raise RuntimeError("Detector worker channels unavailable")

        self._request_id += 1
        req_id = self._request_id
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError("Failed to encode frame for detector worker")
        self._in_q.put({"req_id": req_id, "frame_jpg": encoded.tobytes()}, timeout=WORKER_TIMEOUT_S)

        deadline = time.monotonic() + WORKER_TIMEOUT_S
        while time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            try:
                msg = self._out_q.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError("Detector worker timed out") from exc
            if msg.get("kind") != "result":
                continue
            if int(msg.get("req_id", -1)) != req_id:
                continue
            if not bool(msg.get("ok", False)):
                raise RuntimeError(f"{msg.get('error_code')}: {msg.get('error_message')}")
            boxes = np.asarray(msg.get("boxes", []), dtype=float).reshape(-1, 4)
            cls = np.asarray(msg.get("cls", []), dtype=int).reshape(-1)
            return boxes, cls

        raise TimeoutError("Detector worker response deadline exceeded")

    def infer(self, frame: np.ndarray, use_polygon_zones: bool = True) -> dict[str, Any]:
        self.total_attempts += 1

        try:
            if self.worker_mode:
                boxes, cls = self._infer_worker(frame)
            else:
                model = self._ensure_model()
                boxes, cls = _predict_with_model(model, frame)
            self._record_success()
            return _build_detection_output(
                frame,
                boxes,
                cls,
                detector_ready=True,
                error_code=None,
                error_message="",
                worker_mode=self.worker_mode,
                worker_alive=self._worker_alive(),
                use_polygon_zones=use_polygon_zones,
            )
        except Exception as exc:  # noqa: BLE001
            code, message, is_win = _classify_error(exc)
            self._record_error(code, message, is_win)

            if not self.worker_mode and self._should_activate_worker() and self._start_worker("error_threshold"):
                try:
                    boxes, cls = self._infer_worker(frame)
                    self._record_success()
                    return _build_detection_output(
                        frame,
                        boxes,
                        cls,
                        detector_ready=True,
                        error_code=None,
                        error_message="",
                        worker_mode=True,
                        worker_alive=self._worker_alive(),
                        use_polygon_zones=use_polygon_zones,
                    )
                except Exception as worker_exc:  # noqa: BLE001
                    worker_code, worker_message, worker_is_win = _classify_error(worker_exc)
                    self._record_error(worker_code, worker_message, worker_is_win)

            return _empty_detection(
                detector_ready=False,
                error_code=self.last_error_code,
                error_message=self.last_error_message,
                worker_mode=self.worker_mode,
                worker_alive=self._worker_alive(),
                use_polygon_zones=use_polygon_zones,
            )

    def health(self) -> dict[str, Any]:
        return {
            "detector_ready": self.detector_ready,
            "last_detector_error": {
                "code": self.last_error_code,
                "message": self.last_error_message,
            },
            "detector_worker_mode": self.worker_mode,
            "detector_worker_alive": self._worker_alive(),
            "detector_worker_failover_reason": self.worker_failover_reason,
            "detector_total_attempts": self.total_attempts,
            "detector_total_failures": self.total_failures,
            "detector_startup_winerror_count": self.startup_winerror_count,
            "detector_early_failures": self.early_failures,
        }

    def shutdown(self) -> None:
        if self._in_q is not None:
            try:
                self._in_q.put_nowait(None)
            except Exception:
                pass
        if self._worker_process is not None and self._worker_process.is_alive():
            self._worker_process.join(timeout=1.0)
            if self._worker_process.is_alive():
                self._worker_process.terminate()
        self._worker_process = None
        self._in_q = None
        self._out_q = None


@lru_cache(maxsize=1)
def _runtime() -> _DetectorRuntime:
    return _DetectorRuntime()


def detect_zones(frame: np.ndarray, use_polygon_zones: bool = True) -> dict[str, Any]:
    return _runtime().infer(frame, use_polygon_zones=use_polygon_zones)


def get_detector_health() -> dict[str, Any]:
    return _runtime().health()


def reset_detector_runtime_for_tests() -> None:
    rt = _runtime()
    rt.shutdown()
    _runtime.cache_clear()




