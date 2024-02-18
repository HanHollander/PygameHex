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
HEX_INIT_CHUNK_SIZE: int = 2
HEX_INIT_CHUNK_OVERFLOW = -1
HEX_MIN_CHUNK_SIZE: int = 2
HEX_MAX_CHUNK_SIZE: int = 16
HEX_INIT_STORE_SIZE: V2[int] = V2(256, 160)
HEX_ORIENTATION: HexOrientation = HexOrientation.POINTY
HEX_NOF_HEXES: V2[int] = V2(HEX_INIT_CHUNK_SIZE, HEX_INIT_CHUNK_SIZE) * HEX_INIT_STORE_SIZE

# hex view
DRAG_MOVE_FACTOR: float = 1.5
HEX_MIN_SIZE: int = 2
HEX_MAX_SIZE: int = HEX_INIT_SPRITE_SIZE
ZOOM_STEP_FACTOR: float = 2
ZOOM_MOVE_FACTOR: float = .5
CHUNKS_PER_FRAME: int = 1

# terrain
CONTINENT_NOISE_FREQUENCY: V2[int] = V2(2, 2)
CONTINENT_NOISE_WEIGHT: float = 0.55
FEATURE_NOISE_FREQUENCY: V2[int] = V2(4, 4)
FEATURE_NOISE_WEIGHT: float = 0.15
MOUNTAIN_NOISE_FREQUENCY: V2[int] = V2(8, 8)
MOUNTAIN2_NOISE_FREQUENCY: V2[int] = V2(32, 32)
MOUNTAIN_NOISE_WEIGHT: float = 0.15
TERRAIN_NOISE_FREQUENCY: V2[int] = HEX_INIT_STORE_SIZE.scalar_floordiv(8)
TERRAIN_NOISE_WEIGHT: float = 0.075
TERRAIN1_NOISE_FREQUENCY: V2[int] = HEX_INIT_STORE_SIZE.scalar_floordiv(4)
TERRAIN1_NOISE_WEIGHT: float = 0.05
TERRAIN2_NOISE_FREQUENCY: V2[int] = HEX_INIT_STORE_SIZE.scalar_floordiv(2)
TERRAIN2_NOISE_WEIGHT: float = 0.025
