from enum import Enum


# setup
DEBUG_INFO: bool = True
MAX_FRAMERATE: int = 512
USE_DEFAULT_CURSOR: bool = True

# screen
SCREEN_WIDTH: int = 1920
SCREEN_HEIGHT: int = 1080

# hex model
class HexOrientation(Enum):
    FLAT = 1
    POINTY = 2
HEX_INIT_SIZE: int = 128
HEX_CHUNK_SIZE: int = 8
HEX_NOF_CHUNKS: tuple[int, int] = (10, 10)
HEX_ORIENTATION: HexOrientation = HexOrientation.POINTY
NR_HEX_OUTSIDE_BOUNDS: int = 2

# hex view
DRAG_MOVE_FACTOR: float = 1.5
HEX_MIN_SIZE: int = 8
HEX_MAX_SIZE: int = 256
ZOOM_STEP_FACTOR: float = 1.5  # > 1
ZOOM_MOVE_FACTOR: float = .5
