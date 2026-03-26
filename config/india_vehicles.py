"""Canonical India-specific vehicle mix and PCE constants."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PCEWeights:
    TWO_WHEELER: float = 0.5
    AUTO: float = 1.2
    CAR: float = 1.0
    MINIBUS: float = 1.5
    TRUCK: float = 2.5


PCE_WEIGHTS = PCEWeights()

BHARAT_VEHICLE_MIX = {
    "two_wheeler": 0.55,
    "auto_rickshaw": 0.20,
    "car": 0.15,
    "minibus": 0.07,
    "truck": 0.03,
}

# Approximate physical presets for simulation tuning.
VEHICLE_DIMENSIONS = {
    "two_wheeler": {"length": 2.0, "width": 0.8, "max_speed": 13.0},
    "auto_rickshaw": {"length": 2.6, "width": 1.3, "max_speed": 10.0},
    "car": {"length": 4.3, "width": 1.8, "max_speed": 16.0},
    "minibus": {"length": 6.2, "width": 2.1, "max_speed": 14.0},
    "truck": {"length": 8.0, "width": 2.3, "max_speed": 12.0},
}

