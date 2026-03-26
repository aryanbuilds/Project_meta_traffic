"""Phase and direction helpers shared by controllers."""

from typing import Literal

Direction = Literal["north", "south", "east", "west"]
Phase = Literal["north", "south", "east", "west", "north_south", "east_west"]


def phase_for_direction(direction: str) -> str:
    if direction in {"north", "south"}:
        return "north_south"
    if direction in {"east", "west"}:
        return "east_west"
    return direction


def phase_serves_direction(phase: str, direction: str) -> bool:
    if phase == "north_south":
        return direction in {"north", "south"}
    if phase == "east_west":
        return direction in {"east", "west"}
    return phase == direction
