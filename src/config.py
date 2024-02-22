from configparser import ConfigParser, SectionProxy
from enum import Enum
from math import isclose
from pathlib import Path

from util import V2


class HexOrientation(Enum):
    FLAT = 1
    POINTY = 2

class TerrainLayerShown(Enum):
    CONTINENTS = 1
    MOUNTAINS = 2
    HILLS = 3
    HEIGHTMAP = 4
    CONTINENTS_MOUNTAINS = 5
    CONTINENTS_HILLS = 6
    MOUNTAINS_HILLS = 7

class Cfg:

    @staticmethod
    def init() -> None:
        Cfg.read_general_config()
        Cfg.read_hex_config()
        Cfg.read_terrain_config()

    # ==== GENERAL ==== #
    DEBUG_INFO: bool
    MAX_FRAMERATE: int
    USE_DEFAULT_CURSOR: bool

    SCREEN_WIDTH: int
    SCREEN_HEIGHT: int

    @staticmethod
    def read_general_config() -> None:
        config: ConfigParser = ConfigParser()
        config.read(Path(__file__).parent / "../config/general.cfg")

        general: SectionProxy = config["general"]
        Cfg.DEBUG_INFO = general.getboolean("DEBUG_INFO")
        Cfg.MAX_FRAMERATE = general.getint("MAX_FRAMERATE")
        Cfg.USE_DEFAULT_CURSOR = general.getboolean("USE_DEFAULT_CURSOR")

        screen: SectionProxy = config["screen"]
        Cfg.SCREEN_WIDTH = screen.getint("SCREEN_WIDTH")
        Cfg.SCREEN_HEIGHT = screen.getint("SCREEN_HEIGHT")

    # ==== HEX ==== #
    HEX_INIT_SPRITE_SIZE: int
    HEX_INIT_CHUNK_SIZE: int
    HEX_INIT_CHUNK_OVERFLOW: int
    HEX_MIN_CHUNK_SIZE: int
    HEX_MAX_CHUNK_SIZE: int
    HEX_INIT_STORE_SIZE: V2[int]  # TODO
    HEX_ORIENTATION: HexOrientation  # TODO
    HEX_NOF_HEXES: V2[int]

    DRAG_MOVE_FACTOR: float
    HEX_MIN_SIZE: int
    HEX_MAX_SIZE: int
    ZOOM_STEP_FACTOR: float
    ZOOM_MOVE_FACTOR: float
    CHUNKS_PER_FRAME: int

    @staticmethod
    def read_hex_config() -> None:
        config: ConfigParser = ConfigParser()
        config.read(Path(__file__).parent / "../config/hex.cfg")

        model: SectionProxy = config["model"]
        Cfg.HEX_INIT_SPRITE_SIZE = model.getint("HEX_INIT_SPRITE_SIZE")
        Cfg.HEX_INIT_CHUNK_SIZE = model.getint("HEX_INIT_CHUNK_SIZE")
        Cfg.HEX_INIT_CHUNK_OVERFLOW = model.getint("HEX_INIT_CHUNK_OVERFLOW")
        Cfg.HEX_MIN_CHUNK_SIZE = model.getint("HEX_MIN_CHUNK_SIZE")
        Cfg.HEX_MAX_CHUNK_SIZE = model.getint("HEX_MAX_CHUNK_SIZE")
        Cfg.HEX_INIT_STORE_SIZE = V2(128, 80)
        Cfg.HEX_ORIENTATION = HexOrientation.POINTY
        Cfg.HEX_NOF_HEXES = V2(Cfg.HEX_INIT_CHUNK_SIZE, Cfg.HEX_INIT_CHUNK_SIZE) * Cfg.HEX_INIT_STORE_SIZE

        view: SectionProxy = config["view"]
        Cfg.DRAG_MOVE_FACTOR = view.getfloat("DRAG_MOVE_FACTOR")
        Cfg.HEX_MIN_SIZE = view.getint("HEX_MIN_SIZE")
        Cfg.HEX_MAX_SIZE = Cfg.HEX_INIT_SPRITE_SIZE
        Cfg.ZOOM_STEP_FACTOR = view.getfloat("ZOOM_STEP_FACTOR")
        Cfg.ZOOM_MOVE_FACTOR = view.getfloat("ZOOM_MOVE_FACTOR")
        Cfg.CHUNKS_PER_FRAME = view.getint("CHUNKS_PER_FRAME")

    # ==== HEIGHTMAP ==== #
    CONTINENT_NOISE_WEIGHT: float
    TERRAIN0_NOISE_WEIGHT: float
    TERRAIN1_NOISE_WEIGHT: float
    TERRAIN2_NOISE_WEIGHT: float
    CONTINENT_NOISE_FREQUENCY: V2[int]  # TODO
    TERRAIN0_NOISE_FREQUENCY: V2[int]  # TODO
    TERRAIN1_NOISE_FREQUENCY: V2[int]  # TODO
    TERRAIN2_NOISE_FREQUENCY: V2[int]  # TODO
    CONTINENT_NOISE_PEAK_FLATTENING_RESOLUTION: float                   # lower is higher resolution
    CONTINENT_NOISE_SIZE_MODIF: float                                   # higher is larger continents
    CONTINENT_MAX_HEIGHT_DIV: float                                     # higher is higher continents

    MOUNTAIN0_NOISE_WEIGHT: float
    MOUNTAIN1_NOISE_WEIGHT: float
    MOUNTAIN2_NOISE_WEIGHT: float
    MOUNTAIN_RANGE_NOISE_FREQUENCY: V2[int]  # TODO
    MOUNTAIN0_NOISE_FREQUENCY: V2[int]  # TODO
    MOUNTAIN1_NOISE_FREQUENCY: V2[int]  # TODO
    MOUNTAIN2_NOISE_FREQUENCY: V2[int]  # TODO
    MOUNTAIN_RANGE_NOISE_WIDTH_MODIF: float                             # lower is wider ranges
    MOUNTAIN_MASK_SIZE_MODIF: float                                     # higher is larger mask
    MOUNTAIN_MASK_STRENGTH_MODIF: float                                 # higher is stronger mask

    HILL0_NOISE_WEIGHT: float
    HILL1_NOISE_WEIGHT: float
    HILL2_NOISE_WEIGHT: float
    HILL0_NOISE_FREQUENCY: V2[int]  # TODO
    HILL1_NOISE_FREQUENCY: V2[int]  # TODO 
    HILL2_NOISE_FREQUENCY: V2[int]  # TODO
    HILL_RANGE_NOISE_WIDTH_MODIF: float                                 # lower is wider ranges
    HILL_MASK_SIZE_MODIF: float                                         # higher is larger mask
    HILL_MASK_STRENGTH_MODIF: float                                     # higher is stronger mask
    HILL_NOISE_PEAK_FLATTENING_RESOLUTION: float                        # lower is higher resolution
    HILL_MAX_HEIGHT_DIV: float                                          # higher is higher hills

    TERRAIN_LAYER_SHOWN: TerrainLayerShown

    SHADING_MULT: float

    @staticmethod
    def read_terrain_config() -> None:
        config: ConfigParser = ConfigParser()
        config.read(Path(__file__).parent / "../config/terrain.cfg")

        continents: SectionProxy = config["heightmap.continents"]
        Cfg.CONTINENT_NOISE_WEIGHT = continents.getfloat("CONTINENT_NOISE_WEIGHT")
        Cfg.TERRAIN0_NOISE_WEIGHT = continents.getfloat("TERRAIN0_NOISE_WEIGHT")
        Cfg.TERRAIN1_NOISE_WEIGHT = continents.getfloat("TERRAIN1_NOISE_WEIGHT")
        Cfg.TERRAIN2_NOISE_WEIGHT = continents.getfloat("TERRAIN2_NOISE_WEIGHT")
        assert isclose(Cfg.CONTINENT_NOISE_WEIGHT + Cfg.TERRAIN0_NOISE_WEIGHT + Cfg.TERRAIN1_NOISE_WEIGHT + Cfg.TERRAIN2_NOISE_WEIGHT, 1.0)
        Cfg.CONTINENT_NOISE_FREQUENCY = V2(4, 2)
        Cfg.TERRAIN0_NOISE_FREQUENCY = Cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(8)
        Cfg.TERRAIN1_NOISE_FREQUENCY = Cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(4)
        Cfg.TERRAIN2_NOISE_FREQUENCY = Cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(2)
        Cfg.CONTINENT_NOISE_PEAK_FLATTENING_RESOLUTION = continents.getfloat("CONTINENT_NOISE_PEAK_FLATTENING_RESOLUTION")
        Cfg.CONTINENT_NOISE_SIZE_MODIF = continents.getfloat("CONTINENT_NOISE_SIZE_MODIF")
        Cfg.CONTINENT_MAX_HEIGHT_DIV = continents.getfloat("CONTINENT_MAX_HEIGHT_DIV")

        mountains: SectionProxy = config["heightmap.mountains"]
        Cfg.MOUNTAIN0_NOISE_WEIGHT = mountains.getfloat("MOUNTAIN0_NOISE_WEIGHT")
        Cfg.MOUNTAIN1_NOISE_WEIGHT = mountains.getfloat("MOUNTAIN1_NOISE_WEIGHT")
        Cfg.MOUNTAIN2_NOISE_WEIGHT = mountains.getfloat("MOUNTAIN2_NOISE_WEIGHT")
        assert isclose(Cfg.MOUNTAIN0_NOISE_WEIGHT + Cfg.MOUNTAIN1_NOISE_WEIGHT + Cfg.MOUNTAIN2_NOISE_WEIGHT, 1.0)
        Cfg.MOUNTAIN_RANGE_NOISE_FREQUENCY = V2(8, 4)
        Cfg.MOUNTAIN0_NOISE_FREQUENCY = Cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(8)
        Cfg.MOUNTAIN1_NOISE_FREQUENCY = Cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(4)
        Cfg.MOUNTAIN2_NOISE_FREQUENCY = Cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(2)
        Cfg.MOUNTAIN_RANGE_NOISE_WIDTH_MODIF = mountains.getfloat("MOUNTAIN_RANGE_NOISE_WIDTH_MODIF")
        Cfg.MOUNTAIN_MASK_SIZE_MODIF = mountains.getfloat("MOUNTAIN_MASK_SIZE_MODIF")
        Cfg.MOUNTAIN_MASK_STRENGTH_MODIF = mountains.getfloat("MOUNTAIN_MASK_STRENGTH_MODIF")

        hills: SectionProxy = config["heightmap.hills"]
        Cfg.HILL0_NOISE_WEIGHT = hills.getfloat("HILL0_NOISE_WEIGHT")
        Cfg.HILL1_NOISE_WEIGHT = hills.getfloat("HILL1_NOISE_WEIGHT")
        Cfg.HILL2_NOISE_WEIGHT = hills.getfloat("HILL2_NOISE_WEIGHT")
        assert isclose(Cfg.HILL0_NOISE_WEIGHT + Cfg.HILL1_NOISE_WEIGHT + Cfg.HILL2_NOISE_WEIGHT, 1.0)
        Cfg.HILL0_NOISE_FREQUENCY = Cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(8)
        Cfg.HILL1_NOISE_FREQUENCY = Cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(2)
        Cfg.HILL2_NOISE_FREQUENCY = Cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(1)
        Cfg.HILL_RANGE_NOISE_WIDTH_MODIF = hills.getfloat("HILL_RANGE_NOISE_WIDTH_MODIF")
        Cfg.HILL_NOISE_PEAK_FLATTENING_RESOLUTION = hills.getfloat("HILL_NOISE_PEAK_FLATTENING_RESOLUTION")
        Cfg.HILL_MASK_SIZE_MODIF = hills.getfloat("HILL_MASK_SIZE_MODIF")
        Cfg.HILL_MASK_STRENGTH_MODIF = hills.getfloat("HILL_MASK_STRENGTH_MODIF")
        Cfg.HILL_MAX_HEIGHT_DIV = hills.getfloat("HILL_MAX_HEIGHT_DIV")

        terrain: SectionProxy = config["heightmap.terrain"]
        Cfg.TERRAIN_LAYER_SHOWN = TerrainLayerShown(terrain.getint("TERRAIN_LAYER_SHOWN"))

        colours: SectionProxy = config["colours"]
        Cfg.SHADING_MULT = colours.getfloat("SHADING_MULT")
