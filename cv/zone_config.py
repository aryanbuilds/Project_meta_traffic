"""Static polygon zones for 800x800 top-down frames."""

import numpy as np

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
