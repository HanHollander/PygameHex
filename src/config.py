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
HEX_STORE_SIZE: tuple[int, int] = (100, 100)
HEX_ORIENTATION: HexOrientation = HexOrientation.POINTY

# hex view
DRAG_MOVE_FACTOR: float = 1.5
MIN_SCALE: float = 0.5  # 2x zoomed in
MAX_SCALE: float = 5.0  # 5x zoomed out
ZOOM_STEP_SIZE: float = 0.1
ZOOM_MOVE_FACTOR: float = 0.1
