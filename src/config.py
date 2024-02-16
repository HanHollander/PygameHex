from enum import Enum

from util import V2


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
HEX_INIT_SPRITE_SIZE: int = 256
HEX_INIT_NOF_HEXES_IN_CHUNK: int = 2
HEX_INIT_CHUNK_OVERFLOW = -1
HEX_MIN_CHUNK_SIZE: int = 4
HEX_MAX_CHUNK_SIZE: int = 16
HEX_INIT_NOF_CHUNKS_IN_STORE: V2[int] = V2(128, 80)
HEX_ORIENTATION: HexOrientation = HexOrientation.POINTY

# hex view
DRAG_MOVE_FACTOR: float = 1.5
HEX_MIN_SIZE: int = 4
HEX_MAX_SIZE: int = 128
ZOOM_STEP_FACTOR: float = 2
ZOOM_MOVE_FACTOR: float = .5
CHUNKS_PER_FRAME: int = 1
