"""PCE computation utilities for zone detections."""

from collections import Counter
from typing import Iterable

from config.india_vehicles import PCE_WEIGHTS

YOLO_CLASS_TO_PCE = {
    3: PCE_WEIGHTS.TWO_WHEELER,  # motorcycle
    2: PCE_WEIGHTS.CAR,          # car
    5: PCE_WEIGHTS.TRUCK,        # bus
    7: PCE_WEIGHTS.TRUCK,        # truck
}


def compute_pce(class_ids: Iterable[int]) -> dict:
    ids = list(class_ids)
    counts = Counter(ids)
    total = len(ids)
    motorcycle_count = counts.get(3, 0)

    total_pce = 0.0
    for cid in ids:
        total_pce += YOLO_CLASS_TO_PCE.get(cid, 0.0)

    return {
        "total_pce": round(total_pce, 2),
        "count": total,
        "tw_ratio": (motorcycle_count / total) if total else 0.0,
        "breakdown": {
            "car": counts.get(2, 0),
            "motorcycle": counts.get(3, 0),
            "bus": counts.get(5, 0),
            "truck": counts.get(7, 0),
        },
    }


def compute_all_zones(zone_class_ids: dict[str, list[int]], wait_s: dict[str, int] | None = None) -> dict[str, dict]:
    wait_s = wait_s or {}
    out: dict[str, dict] = {}
    for direction in ("north", "south", "east", "west"):
        base = compute_pce(zone_class_ids.get(direction, []))
        base["wait_s"] = int(wait_s.get(direction, 0))
        out[direction] = base
    return out
