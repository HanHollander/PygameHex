from configparser import ConfigParser, SectionProxy
from enum import Enum
from math import isclose
from pathlib import Path
import pygame as pg

from util import V2


class HexOrientation(Enum):
    FLAT = 0
    POINTY = 1

class TerrainLayerShown(Enum):
    CONTINENTS = 1
    MOUNTAINS = 2
    HILLS = 3
    HEIGHTMAP = 4
    CONTINENTS_MOUNTAINS = 5
    CONTINENTS_HILLS = 6
    MOUNTAINS_HILLS = 7
    CONTINENT_MASK = 8
    MOUNTAIN_RANGE_NOISE = 9

class ColourScheme(Enum):
    TERRAIN = 0
    HEIGHTMAP = 1
    HUMIDITYMAP = 2
    TEMPERATUREMAP = 3


def v2_converter(s: str) -> V2[int]:
    t = tuple(int(k.strip()) for k in s[1:-1].split(','))
    return V2(t[0], t[1])

def clrtuple_converter(s: str) -> tuple[pg.Color, ...]:
    return tuple(pg.Color(str(k.strip())) for k in s[1:-1].split(','))


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
    HEX_INIT_STORE_SIZE: V2[int]
    HEX_ORIENTATION: HexOrientation
    HEX_NOF_HEXES: V2[int]

    DRAG_MOVE_FACTOR: float
    HEX_MIN_SIZE: int
    HEX_MAX_SIZE: int
    ZOOM_STEP_FACTOR: float
    ZOOM_MOVE_FACTOR: float
    CHUNKS_PER_FRAME: int

    @staticmethod
    def read_hex_config() -> None:
        config: ConfigParser = ConfigParser(converters={"v2": v2_converter})
        config.read(Path(__file__).parent / "../config/hex.cfg")

        model: SectionProxy = config["model"]
        Cfg.HEX_INIT_SPRITE_SIZE = model.getint("HEX_INIT_SPRITE_SIZE")
        Cfg.HEX_INIT_CHUNK_SIZE = model.getint("HEX_INIT_CHUNK_SIZE")
        Cfg.HEX_INIT_CHUNK_OVERFLOW = model.getint("HEX_INIT_CHUNK_OVERFLOW")
        Cfg.HEX_MIN_CHUNK_SIZE = model.getint("HEX_MIN_CHUNK_SIZE")
        Cfg.HEX_MAX_CHUNK_SIZE = model.getint("HEX_MAX_CHUNK_SIZE")
        Cfg.HEX_INIT_STORE_SIZE = model.getv2("HEX_INIT_STORE_SIZE")
        Cfg.HEX_ORIENTATION = HexOrientation(model.getint("HEX_ORIENTATION"))
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
    CONTINENT_NOISE_FREQUENCY: V2[int]
    TERRAIN0_NOISE_FREQUENCY: V2[int]
    TERRAIN1_NOISE_FREQUENCY: V2[int]
    TERRAIN2_NOISE_FREQUENCY: V2[int]
    CONTINENT_NOISE_PEAK_FLATTENING_RESOLUTION: float                   # lower is higher resolution
    CONTINENT_NOISE_SIZE_MODIF: float                                   # higher is larger continents
    CONTINENT_MAX_HEIGHT_DIV: float                                     # higher is higher continents

    MOUNTAIN0_NOISE_WEIGHT: float
    MOUNTAIN1_NOISE_WEIGHT: float
    MOUNTAIN2_NOISE_WEIGHT: float
    MOUNTAIN_RANGE_NOISE_FREQUENCY: V2[int]
    MOUNTAIN0_NOISE_FREQUENCY: V2[int]
    MOUNTAIN1_NOISE_FREQUENCY: V2[int]
    MOUNTAIN2_NOISE_FREQUENCY: V2[int]
    MOUNTAIN_RANGE_NOISE_WIDTH_MODIF: float                             # lower is wider ranges
    MOUNTAIN_MASK_SIZE_MODIF: float                                     # higher is larger mask
    MOUNTAIN_MASK_STRENGTH_MODIF: float                                 # higher is stronger mask

    HILL0_NOISE_WEIGHT: float
    HILL1_NOISE_WEIGHT: float
    HILL2_NOISE_WEIGHT: float
    HILL0_NOISE_FREQUENCY: V2[int]
    HILL1_NOISE_FREQUENCY: V2[int]
    HILL2_NOISE_FREQUENCY: V2[int]
    HILL_RANGE_NOISE_WIDTH_MODIF: float                                 # lower is wider ranges
    HILL_MASK_SIZE_MODIF: float                                         # higher is larger mask
    HILL_MASK_STRENGTH_MODIF: float                                     # higher is stronger mask
    HILL_NOISE_PEAK_FLATTENING_RESOLUTION: float                        # lower is higher resolution
    HILL_MAX_HEIGHT_DIV: float                                          # higher is higher hills

    TERRAIN_LAYER_SHOWN: TerrainLayerShown
    NS_MASK0_DIV: int
    NS_MASK1_DIV: int
    NS_MASK2_DIV: int
    NS_MASK0_VAL: float
    NS_MASK1_VAL: float
    NS_MASK2_VAL: float
    NS_MASK_GAUSSIAN_FILTER_SIZE: int
    NS_MASK_NOISE_WEIGHT: float

    A_HIGH: float
    A_MEDIUM: float
    A_LOW: float
    A_SHALLOW_WATER: float
    A_DEEP_WATER: float

    H_ARID: float
    H_AVERAGE: float
    H_HUMID: float
    H_0: float
    H_1: float
    H_2: float
    H_3: float
    H_4: float
    H_CONTINENT_MASK_NOISE_WEIGHT: float
    H_CONTINENT_MASK_THRESHOLD: float
    H_MOUNTAIN_MASK_NOISE_WEIGHT: float
    H_MOUNTAIN_MASK_THRESHOLD: float

    T_FREEZING: float
    T_COLD: float
    T_AVERAGE: float
    T_WARM: float
    T_HOT: float
    T_0: float
    T_1: float
    T_2: float
    T_3: float
    T_4: float
    T_CONTINENT_MASK_NOISE_WEIGHT: float

    COLOUR_SCHEME: ColourScheme

    SHADING_MULT: float
    C_ARCTIC: tuple[pg.Color, pg.Color]
    C_TUNDRA: tuple[pg.Color, pg.Color]
    C_STEPPE: tuple[pg.Color, pg.Color]
    C_SAVANNA: tuple[pg.Color, pg.Color]
    C_DESERT: tuple[pg.Color, pg.Color]
    C_BOREAL: tuple[pg.Color, pg.Color]
    C_TEMPERATE: tuple[pg.Color, pg.Color]
    C_MEDITERRANEAN: tuple[pg.Color, pg.Color]
    C_TROPIC: tuple[pg.Color, pg.Color]
    C_SHALLOW_WATER: tuple[pg.Color, pg.Color]
    C_DEEP_WATER: tuple[pg.Color, pg.Color]
    C_VOID: tuple[pg.Color, pg.Color]

    C_LOW: pg.Color
    C_HIGH: pg.Color

    H_H_LOW: int
    H_H_HIGH: int
    H_S: float
    H_L: float

    T_H_LOW: int
    T_H_HIGH: int
    T_S: float
    T_L: float

    @staticmethod
    def read_terrain_config() -> None:
        config: ConfigParser = ConfigParser(converters={"v2": v2_converter, "clrtuple": clrtuple_converter})
        config.read(Path(__file__).parent / "../config/terrain.cfg")

        continents: SectionProxy = config["heightmap.continents"]
        Cfg.CONTINENT_NOISE_WEIGHT = continents.getfloat("CONTINENT_NOISE_WEIGHT")
        Cfg.TERRAIN0_NOISE_WEIGHT = continents.getfloat("TERRAIN0_NOISE_WEIGHT")
        Cfg.TERRAIN1_NOISE_WEIGHT = continents.getfloat("TERRAIN1_NOISE_WEIGHT")
        Cfg.TERRAIN2_NOISE_WEIGHT = continents.getfloat("TERRAIN2_NOISE_WEIGHT")
        assert isclose(Cfg.CONTINENT_NOISE_WEIGHT + Cfg.TERRAIN0_NOISE_WEIGHT + Cfg.TERRAIN1_NOISE_WEIGHT + Cfg.TERRAIN2_NOISE_WEIGHT, 1.0)
        Cfg.CONTINENT_NOISE_FREQUENCY = continents.getv2("CONTINENT_NOISE_FREQUENCY")
        TERRAIN0_NOISE_FREQUENCY_DIV: int = continents.getint("TERRAIN0_NOISE_FREQUENCY_DIV")
        Cfg.TERRAIN0_NOISE_FREQUENCY = Cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(TERRAIN0_NOISE_FREQUENCY_DIV)
        TERRAIN1_NOISE_FREQUENCY_DIV: int = continents.getint("TERRAIN1_NOISE_FREQUENCY_DIV")
        Cfg.TERRAIN1_NOISE_FREQUENCY = Cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(TERRAIN1_NOISE_FREQUENCY_DIV)
        TERRAIN2_NOISE_FREQUENCY_DIV: int = continents.getint("TERRAIN2_NOISE_FREQUENCY_DIV")
        Cfg.TERRAIN2_NOISE_FREQUENCY = Cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(TERRAIN2_NOISE_FREQUENCY_DIV)
        Cfg.CONTINENT_NOISE_PEAK_FLATTENING_RESOLUTION = continents.getfloat("CONTINENT_NOISE_PEAK_FLATTENING_RESOLUTION")
        Cfg.CONTINENT_NOISE_SIZE_MODIF = continents.getfloat("CONTINENT_NOISE_SIZE_MODIF")
        Cfg.CONTINENT_MAX_HEIGHT_DIV = continents.getfloat("CONTINENT_MAX_HEIGHT_DIV")

        mountains: SectionProxy = config["heightmap.mountains"]
        Cfg.MOUNTAIN0_NOISE_WEIGHT = mountains.getfloat("MOUNTAIN0_NOISE_WEIGHT")
        Cfg.MOUNTAIN1_NOISE_WEIGHT = mountains.getfloat("MOUNTAIN1_NOISE_WEIGHT")
        Cfg.MOUNTAIN2_NOISE_WEIGHT = mountains.getfloat("MOUNTAIN2_NOISE_WEIGHT")
        assert isclose(Cfg.MOUNTAIN0_NOISE_WEIGHT + Cfg.MOUNTAIN1_NOISE_WEIGHT + Cfg.MOUNTAIN2_NOISE_WEIGHT, 1.0)
        Cfg.MOUNTAIN_RANGE_NOISE_FREQUENCY = mountains.getv2("MOUNTAIN_RANGE_NOISE_FREQUENCY")
        MOUNTAIN0_NOISE_FREQUENCY_DIV: int = mountains.getint("MOUNTAIN0_NOISE_FREQUENCY_DIV")
        Cfg.MOUNTAIN0_NOISE_FREQUENCY = Cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(MOUNTAIN0_NOISE_FREQUENCY_DIV)
        MOUNTAIN1_NOISE_FREQUENCY_DIV: int = mountains.getint("MOUNTAIN1_NOISE_FREQUENCY_DIV")
        Cfg.MOUNTAIN1_NOISE_FREQUENCY = Cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(MOUNTAIN1_NOISE_FREQUENCY_DIV)
        MOUNTAIN2_NOISE_FREQUENCY_DIV: int = mountains.getint("MOUNTAIN2_NOISE_FREQUENCY_DIV")
        Cfg.MOUNTAIN2_NOISE_FREQUENCY = Cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(MOUNTAIN2_NOISE_FREQUENCY_DIV)
        Cfg.MOUNTAIN_RANGE_NOISE_WIDTH_MODIF = mountains.getfloat("MOUNTAIN_RANGE_NOISE_WIDTH_MODIF")
        Cfg.MOUNTAIN_MASK_SIZE_MODIF = mountains.getfloat("MOUNTAIN_MASK_SIZE_MODIF")
        Cfg.MOUNTAIN_MASK_STRENGTH_MODIF = mountains.getfloat("MOUNTAIN_MASK_STRENGTH_MODIF")

        hills: SectionProxy = config["heightmap.hills"]
        Cfg.HILL0_NOISE_WEIGHT = hills.getfloat("HILL0_NOISE_WEIGHT")
        Cfg.HILL1_NOISE_WEIGHT = hills.getfloat("HILL1_NOISE_WEIGHT")
        Cfg.HILL2_NOISE_WEIGHT = hills.getfloat("HILL2_NOISE_WEIGHT")
        assert isclose(Cfg.HILL0_NOISE_WEIGHT + Cfg.HILL1_NOISE_WEIGHT + Cfg.HILL2_NOISE_WEIGHT, 1.0)
        HILL0_NOISE_FREQUENCY_DIV: int = hills.getint("HILL0_NOISE_FREQUENCY_DIV")
        Cfg.HILL0_NOISE_FREQUENCY = Cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(HILL0_NOISE_FREQUENCY_DIV)
        HILL1_NOISE_FREQUENCY_DIV: int = hills.getint("HILL1_NOISE_FREQUENCY_DIV")
        Cfg.HILL1_NOISE_FREQUENCY = Cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(HILL1_NOISE_FREQUENCY_DIV)
        HILL2_NOISE_FREQUENCY_DIV: int = hills.getint("HILL2_NOISE_FREQUENCY_DIV")
        Cfg.HILL2_NOISE_FREQUENCY = Cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(HILL2_NOISE_FREQUENCY_DIV)
        Cfg.HILL_RANGE_NOISE_WIDTH_MODIF = hills.getfloat("HILL_RANGE_NOISE_WIDTH_MODIF")
        Cfg.HILL_NOISE_PEAK_FLATTENING_RESOLUTION = hills.getfloat("HILL_NOISE_PEAK_FLATTENING_RESOLUTION")
        Cfg.HILL_MASK_SIZE_MODIF = hills.getfloat("HILL_MASK_SIZE_MODIF")
        Cfg.HILL_MASK_STRENGTH_MODIF = hills.getfloat("HILL_MASK_STRENGTH_MODIF")
        Cfg.HILL_MAX_HEIGHT_DIV = hills.getfloat("HILL_MAX_HEIGHT_DIV")

        terrain: SectionProxy = config["heightmap.terrain"]
        Cfg.TERRAIN_LAYER_SHOWN = TerrainLayerShown(terrain.getint("TERRAIN_LAYER_SHOWN"))
        Cfg.NS_MASK0_DIV = terrain.getint("NS_MASK0_DIV")
        Cfg.NS_MASK1_DIV = terrain.getint("NS_MASK1_DIV")
        Cfg.NS_MASK2_DIV = terrain.getint("NS_MASK2_DIV")
        Cfg.NS_MASK0_VAL = terrain.getfloat("NS_MASK0_VAL")
        Cfg.NS_MASK1_VAL = terrain.getfloat("NS_MASK1_VAL")
        Cfg.NS_MASK2_VAL = terrain.getfloat("NS_MASK2_VAL")
        Cfg.NS_MASK_GAUSSIAN_FILTER_SIZE = terrain.getint("NS_MASK_GAUSSIAN_FILTER_SIZE")
        Cfg.NS_MASK_NOISE_WEIGHT = terrain.getfloat("NS_MASK_NOISE_WEIGTH")

        altitude: SectionProxy = config["altitude"]
        Cfg.A_HIGH = altitude.getfloat("A_HIGH")
        Cfg.A_MEDIUM = altitude.getfloat("A_MEDIUM")
        Cfg.A_LOW = altitude.getfloat("A_LOW")
        Cfg.A_SHALLOW_WATER = altitude.getfloat("A_SHALLOW_WATER")
        Cfg.A_DEEP_WATER = altitude.getfloat("A_DEEP_WATER")

        humidity: SectionProxy = config["humidity"]
        Cfg.H_ARID = humidity.getfloat("H_ARID")
        Cfg.H_AVERAGE = humidity.getfloat("H_AVERAGE")
        Cfg.H_HUMID = humidity.getfloat("H_HUMID")
        Cfg.H_0 = humidity.getfloat("H_0")
        Cfg.H_1 = humidity.getfloat("H_1")
        Cfg.H_2 = humidity.getfloat("H_2")
        Cfg.H_3 = humidity.getfloat("H_3")
        Cfg.H_4 = humidity.getfloat("H_4")
        Cfg.H_CONTINENT_MASK_NOISE_WEIGHT = humidity.getfloat("H_CONTINENT_MASK_NOISE_WEIGHT")
        Cfg.H_CONTINENT_MASK_THRESHOLD = humidity.getfloat("H_CONTINENT_MASK_THRESHOLD")
        Cfg.H_MOUNTAIN_MASK_NOISE_WEIGHT = humidity.getfloat("H_MOUNTAIN_MASK_NOISE_WEIGHT")
        Cfg.H_MOUNTAIN_MASK_THRESHOLD = humidity.getfloat("H_MOUNTAIN_MASK_THRESHOLD")

        temperature: SectionProxy = config["temperature"]
        Cfg.T_FREEZING = temperature.getfloat("T_FREEZING")
        Cfg.T_COLD = temperature.getfloat("T_COLD")
        Cfg.T_AVERAGE = temperature.getfloat("T_AVERAGE")
        Cfg.T_WARM = temperature.getfloat("T_WARM")
        Cfg.T_HOT = temperature.getfloat("T_HOT")
        Cfg.T_0 = temperature.getfloat("T_0")
        Cfg.T_1 = temperature.getfloat("T_1")
        Cfg.T_2 = temperature.getfloat("T_2")
        Cfg.T_3 = temperature.getfloat("T_3")
        Cfg.T_4 = temperature.getfloat("T_4")
        Cfg.T_CONTINENT_MASK_NOISE_WEIGHT = temperature.getfloat("T_CONTINENT_MASK_NOISE_WEIGHT")

        colours: SectionProxy = config["colours"]
        Cfg.COLOUR_SCHEME = ColourScheme(colours.getint("COLOUR_SCHEME"))

        terrainmap: SectionProxy = config["colours.terrainmap"]
        Cfg.SHADING_MULT = terrainmap.getfloat("SHADING_MULT")
        Cfg.C_ARCTIC = terrainmap.getclrtuple("C_ARCTIC")
        Cfg.C_TUNDRA = terrainmap.getclrtuple("C_TUNDRA")
        Cfg.C_STEPPE = terrainmap.getclrtuple("C_STEPPE")
        Cfg.C_SAVANNA = terrainmap.getclrtuple("C_SAVANNA")
        Cfg.C_DESERT = terrainmap.getclrtuple("C_DESERT")
        Cfg.C_BOREAL = terrainmap.getclrtuple("C_BOREAL")
        Cfg.C_TEMPERATE = terrainmap.getclrtuple("C_TEMPERATE")
        Cfg.C_MEDITERRANEAN = terrainmap.getclrtuple("C_MEDITERRANEAN")
        Cfg.C_TROPIC = terrainmap.getclrtuple("C_TROPIC")
        Cfg.C_SHALLOW_WATER = terrainmap.getclrtuple("C_SHALLOW_WATER")
        Cfg.C_DEEP_WATER = terrainmap.getclrtuple("C_DEEP_WATER")
        Cfg.C_VOID = terrainmap.getclrtuple("C_VOID")

        heightmap: SectionProxy = config["colours.heightmap"]
        Cfg.C_LOW = pg.Color(heightmap.get("C_LOW"))
        Cfg.C_HIGH = pg.Color(heightmap.get("C_HIGH"))

        humiditymap: SectionProxy = config["colours.humiditymap"]
        Cfg.H_H_LOW = humiditymap.getint("H_H_LOW")
        Cfg.H_H_HIGH = humiditymap.getint("H_H_HIGH")
        Cfg.H_S = humiditymap.getfloat("H_S")
        Cfg.H_L = humiditymap.getfloat("H_L")

        temperaturemap: SectionProxy = config["colours.temperaturemap"]
        Cfg.T_H_LOW = temperaturemap.getint("T_H_LOW")
        Cfg.T_H_HIGH = temperaturemap.getint("T_H_HIGH")
        Cfg.T_S = temperaturemap.getfloat("T_S")
        Cfg.T_L = temperaturemap.getfloat("T_L")
