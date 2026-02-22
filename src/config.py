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
    GRADIENT = 4


def v2_converter(s: str) -> V2[int]:
    t = tuple(int(k.strip()) for k in s[1:-1].split(','))
    return V2(t[0], t[1])

def clrtuple_converter(s: str) -> tuple[pg.Color, ...]:
    return tuple(pg.Color(str(k.strip())) for k in s[1:-1].split(','))


class cfg:

    @staticmethod
    def init() -> None:
        cfg.read_general_config()
        cfg.read_hex_config()
        cfg.read_terrain_config()

    # ==== GENERAL ==== #
    DEBUG_INFO: bool
    MAX_FRAMERATE: int
    USE_DEFAULT_CURSOR: bool
    PROFILE: bool

    SCREEN_WIDTH: int
    SCREEN_HEIGHT: int

    @staticmethod
    def read_general_config() -> None:
        config: ConfigParser = ConfigParser()
        config.read(Path(__file__).parent / "../config/general.cfg")

        general: SectionProxy = config["general"]
        cfg.DEBUG_INFO = general.getboolean("DEBUG_INFO")
        cfg.MAX_FRAMERATE = general.getint("MAX_FRAMERATE")
        cfg.USE_DEFAULT_CURSOR = general.getboolean("USE_DEFAULT_CURSOR")
        cfg.PROFILE = general.getboolean("PROFILE")

        screen: SectionProxy = config["screen"]
        cfg.SCREEN_WIDTH = screen.getint("SCREEN_WIDTH")
        cfg.SCREEN_HEIGHT = screen.getint("SCREEN_HEIGHT")

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
        cfg.HEX_INIT_SPRITE_SIZE = model.getint("HEX_INIT_SPRITE_SIZE")
        cfg.HEX_INIT_CHUNK_SIZE = model.getint("HEX_INIT_CHUNK_SIZE")
        cfg.HEX_INIT_CHUNK_OVERFLOW = model.getint("HEX_INIT_CHUNK_OVERFLOW")
        cfg.HEX_MIN_CHUNK_SIZE = model.getint("HEX_MIN_CHUNK_SIZE")
        cfg.HEX_MAX_CHUNK_SIZE = model.getint("HEX_MAX_CHUNK_SIZE")
        cfg.HEX_INIT_STORE_SIZE = model.getv2("HEX_INIT_STORE_SIZE")
        cfg.HEX_ORIENTATION = HexOrientation(model.getint("HEX_ORIENTATION"))
        cfg.HEX_NOF_HEXES = V2(cfg.HEX_INIT_CHUNK_SIZE, cfg.HEX_INIT_CHUNK_SIZE) * cfg.HEX_INIT_STORE_SIZE

        view: SectionProxy = config["view"]
        cfg.DRAG_MOVE_FACTOR = view.getfloat("DRAG_MOVE_FACTOR")
        cfg.HEX_MIN_SIZE = view.getint("HEX_MIN_SIZE")
        cfg.HEX_MAX_SIZE = cfg.HEX_INIT_SPRITE_SIZE
        cfg.ZOOM_STEP_FACTOR = view.getfloat("ZOOM_STEP_FACTOR")
        cfg.ZOOM_MOVE_FACTOR = view.getfloat("ZOOM_MOVE_FACTOR")
        cfg.CHUNKS_PER_FRAME = view.getint("CHUNKS_PER_FRAME")

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
    H_HEIGHTMAP_MASK_NOISE_WEIGHT: float

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
    SEA_SHADING_MULT_MODIF: float
    C_ARCTIC: pg.Color
    C_TUNDRA: pg.Color
    C_STEPPE: pg.Color
    C_SAVANNA: pg.Color
    C_DESERT: pg.Color
    C_BOREAL: pg.Color
    C_TEMPERATE: pg.Color
    C_MEDITERRANEAN: pg.Color
    C_TROPIC: pg.Color
    C_SHALLOW_WATER: pg.Color
    C_DEEP_WATER: pg.Color
    C_VOID: pg.Color

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

    C_G_1: pg.Color
    C_G_2: pg.Color
    C_G_3: pg.Color
    C_G_4: pg.Color

    @staticmethod
    def read_terrain_config() -> None:
        config: ConfigParser = ConfigParser(converters={"v2": v2_converter, "clrtuple": clrtuple_converter})
        config.read(Path(__file__).parent / "../config/terrain.cfg")

        continents: SectionProxy = config["heightmap.continents"]
        cfg.CONTINENT_NOISE_WEIGHT = continents.getfloat("CONTINENT_NOISE_WEIGHT")
        cfg.TERRAIN0_NOISE_WEIGHT = continents.getfloat("TERRAIN0_NOISE_WEIGHT")
        cfg.TERRAIN1_NOISE_WEIGHT = continents.getfloat("TERRAIN1_NOISE_WEIGHT")
        cfg.TERRAIN2_NOISE_WEIGHT = continents.getfloat("TERRAIN2_NOISE_WEIGHT")
        assert isclose(cfg.CONTINENT_NOISE_WEIGHT + cfg.TERRAIN0_NOISE_WEIGHT + cfg.TERRAIN1_NOISE_WEIGHT + cfg.TERRAIN2_NOISE_WEIGHT, 1.0)
        cfg.CONTINENT_NOISE_FREQUENCY = continents.getv2("CONTINENT_NOISE_FREQUENCY")
        TERRAIN0_NOISE_FREQUENCY_DIV: int = continents.getint("TERRAIN0_NOISE_FREQUENCY_DIV")
        cfg.TERRAIN0_NOISE_FREQUENCY = cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(TERRAIN0_NOISE_FREQUENCY_DIV)
        TERRAIN1_NOISE_FREQUENCY_DIV: int = continents.getint("TERRAIN1_NOISE_FREQUENCY_DIV")
        cfg.TERRAIN1_NOISE_FREQUENCY = cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(TERRAIN1_NOISE_FREQUENCY_DIV)
        TERRAIN2_NOISE_FREQUENCY_DIV: int = continents.getint("TERRAIN2_NOISE_FREQUENCY_DIV")
        cfg.TERRAIN2_NOISE_FREQUENCY = cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(TERRAIN2_NOISE_FREQUENCY_DIV)
        cfg.CONTINENT_NOISE_PEAK_FLATTENING_RESOLUTION = continents.getfloat("CONTINENT_NOISE_PEAK_FLATTENING_RESOLUTION")
        cfg.CONTINENT_NOISE_SIZE_MODIF = continents.getfloat("CONTINENT_NOISE_SIZE_MODIF")
        cfg.CONTINENT_MAX_HEIGHT_DIV = continents.getfloat("CONTINENT_MAX_HEIGHT_DIV")

        mountains: SectionProxy = config["heightmap.mountains"]
        cfg.MOUNTAIN0_NOISE_WEIGHT = mountains.getfloat("MOUNTAIN0_NOISE_WEIGHT")
        cfg.MOUNTAIN1_NOISE_WEIGHT = mountains.getfloat("MOUNTAIN1_NOISE_WEIGHT")
        cfg.MOUNTAIN2_NOISE_WEIGHT = mountains.getfloat("MOUNTAIN2_NOISE_WEIGHT")
        assert isclose(cfg.MOUNTAIN0_NOISE_WEIGHT + cfg.MOUNTAIN1_NOISE_WEIGHT + cfg.MOUNTAIN2_NOISE_WEIGHT, 1.0)
        cfg.MOUNTAIN_RANGE_NOISE_FREQUENCY = mountains.getv2("MOUNTAIN_RANGE_NOISE_FREQUENCY")
        MOUNTAIN0_NOISE_FREQUENCY_DIV: int = mountains.getint("MOUNTAIN0_NOISE_FREQUENCY_DIV")
        cfg.MOUNTAIN0_NOISE_FREQUENCY = cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(MOUNTAIN0_NOISE_FREQUENCY_DIV)
        MOUNTAIN1_NOISE_FREQUENCY_DIV: int = mountains.getint("MOUNTAIN1_NOISE_FREQUENCY_DIV")
        cfg.MOUNTAIN1_NOISE_FREQUENCY = cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(MOUNTAIN1_NOISE_FREQUENCY_DIV)
        MOUNTAIN2_NOISE_FREQUENCY_DIV: int = mountains.getint("MOUNTAIN2_NOISE_FREQUENCY_DIV")
        cfg.MOUNTAIN2_NOISE_FREQUENCY = cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(MOUNTAIN2_NOISE_FREQUENCY_DIV)
        cfg.MOUNTAIN_RANGE_NOISE_WIDTH_MODIF = mountains.getfloat("MOUNTAIN_RANGE_NOISE_WIDTH_MODIF")
        cfg.MOUNTAIN_MASK_SIZE_MODIF = mountains.getfloat("MOUNTAIN_MASK_SIZE_MODIF")
        cfg.MOUNTAIN_MASK_STRENGTH_MODIF = mountains.getfloat("MOUNTAIN_MASK_STRENGTH_MODIF")

        hills: SectionProxy = config["heightmap.hills"]
        cfg.HILL0_NOISE_WEIGHT = hills.getfloat("HILL0_NOISE_WEIGHT")
        cfg.HILL1_NOISE_WEIGHT = hills.getfloat("HILL1_NOISE_WEIGHT")
        cfg.HILL2_NOISE_WEIGHT = hills.getfloat("HILL2_NOISE_WEIGHT")
        assert isclose(cfg.HILL0_NOISE_WEIGHT + cfg.HILL1_NOISE_WEIGHT + cfg.HILL2_NOISE_WEIGHT, 1.0)
        HILL0_NOISE_FREQUENCY_DIV: int = hills.getint("HILL0_NOISE_FREQUENCY_DIV")
        cfg.HILL0_NOISE_FREQUENCY = cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(HILL0_NOISE_FREQUENCY_DIV)
        HILL1_NOISE_FREQUENCY_DIV: int = hills.getint("HILL1_NOISE_FREQUENCY_DIV")
        cfg.HILL1_NOISE_FREQUENCY = cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(HILL1_NOISE_FREQUENCY_DIV)
        HILL2_NOISE_FREQUENCY_DIV: int = hills.getint("HILL2_NOISE_FREQUENCY_DIV")
        cfg.HILL2_NOISE_FREQUENCY = cfg.HEX_INIT_STORE_SIZE.scalar_floordiv(HILL2_NOISE_FREQUENCY_DIV)
        cfg.HILL_RANGE_NOISE_WIDTH_MODIF = hills.getfloat("HILL_RANGE_NOISE_WIDTH_MODIF")
        cfg.HILL_NOISE_PEAK_FLATTENING_RESOLUTION = hills.getfloat("HILL_NOISE_PEAK_FLATTENING_RESOLUTION")
        cfg.HILL_MASK_SIZE_MODIF = hills.getfloat("HILL_MASK_SIZE_MODIF")
        cfg.HILL_MASK_STRENGTH_MODIF = hills.getfloat("HILL_MASK_STRENGTH_MODIF")
        cfg.HILL_MAX_HEIGHT_DIV = hills.getfloat("HILL_MAX_HEIGHT_DIV")

        terrain: SectionProxy = config["heightmap.terrain"]
        cfg.TERRAIN_LAYER_SHOWN = TerrainLayerShown(terrain.getint("TERRAIN_LAYER_SHOWN"))
        cfg.NS_MASK0_DIV = terrain.getint("NS_MASK0_DIV")
        cfg.NS_MASK1_DIV = terrain.getint("NS_MASK1_DIV")
        cfg.NS_MASK2_DIV = terrain.getint("NS_MASK2_DIV")
        cfg.NS_MASK0_VAL = terrain.getfloat("NS_MASK0_VAL")
        cfg.NS_MASK1_VAL = terrain.getfloat("NS_MASK1_VAL")
        cfg.NS_MASK2_VAL = terrain.getfloat("NS_MASK2_VAL")
        cfg.NS_MASK_GAUSSIAN_FILTER_SIZE = terrain.getint("NS_MASK_GAUSSIAN_FILTER_SIZE")
        cfg.NS_MASK_NOISE_WEIGHT = terrain.getfloat("NS_MASK_NOISE_WEIGTH")

        altitude: SectionProxy = config["altitude"]
        cfg.A_HIGH = altitude.getfloat("A_HIGH")
        cfg.A_MEDIUM = altitude.getfloat("A_MEDIUM")
        cfg.A_LOW = altitude.getfloat("A_LOW")
        cfg.A_SHALLOW_WATER = altitude.getfloat("A_SHALLOW_WATER")
        cfg.A_DEEP_WATER = altitude.getfloat("A_DEEP_WATER")

        humidity: SectionProxy = config["humidity"]
        cfg.H_ARID = humidity.getfloat("H_ARID")
        cfg.H_AVERAGE = humidity.getfloat("H_AVERAGE")
        cfg.H_HUMID = humidity.getfloat("H_HUMID")
        cfg.H_0 = humidity.getfloat("H_0")
        cfg.H_1 = humidity.getfloat("H_1")
        cfg.H_2 = humidity.getfloat("H_2")
        cfg.H_3 = humidity.getfloat("H_3")
        cfg.H_4 = humidity.getfloat("H_4")
        cfg.H_CONTINENT_MASK_NOISE_WEIGHT = humidity.getfloat("H_CONTINENT_MASK_NOISE_WEIGHT")
        cfg.H_CONTINENT_MASK_THRESHOLD = humidity.getfloat("H_CONTINENT_MASK_THRESHOLD")
        cfg.H_MOUNTAIN_MASK_NOISE_WEIGHT = humidity.getfloat("H_MOUNTAIN_MASK_NOISE_WEIGHT")
        cfg.H_MOUNTAIN_MASK_THRESHOLD = humidity.getfloat("H_MOUNTAIN_MASK_THRESHOLD")
        cfg.H_HEIGHTMAP_MASK_NOISE_WEIGHT = humidity.getfloat("H_HEIGHTMAP_MASK_NOISE_WEIGHT")

        temperature: SectionProxy = config["temperature"]
        cfg.T_FREEZING = temperature.getfloat("T_FREEZING")
        cfg.T_COLD = temperature.getfloat("T_COLD")
        cfg.T_AVERAGE = temperature.getfloat("T_AVERAGE")
        cfg.T_WARM = temperature.getfloat("T_WARM")
        cfg.T_HOT = temperature.getfloat("T_HOT")
        cfg.T_0 = temperature.getfloat("T_0")
        cfg.T_1 = temperature.getfloat("T_1")
        cfg.T_2 = temperature.getfloat("T_2")
        cfg.T_3 = temperature.getfloat("T_3")
        cfg.T_4 = temperature.getfloat("T_4")
        cfg.T_CONTINENT_MASK_NOISE_WEIGHT = temperature.getfloat("T_CONTINENT_MASK_NOISE_WEIGHT")

        colours: SectionProxy = config["colours"]
        cfg.COLOUR_SCHEME = ColourScheme(colours.getint("COLOUR_SCHEME"))

        terrainmap: SectionProxy = config["colours.terrainmap"]
        cfg.SHADING_MULT = terrainmap.getfloat("SHADING_MULT")
        cfg.SEA_SHADING_MULT_MODIF = terrainmap.getfloat("SEA_SHADING_MULT_MODIF")
        cfg.C_ARCTIC = pg.Color(terrainmap.get("C_ARCTIC"))
        cfg.C_TUNDRA = pg.Color(terrainmap.get("C_TUNDRA"))
        cfg.C_STEPPE = pg.Color(terrainmap.get("C_STEPPE"))
        cfg.C_SAVANNA = pg.Color(terrainmap.get("C_SAVANNA"))
        cfg.C_DESERT = pg.Color(terrainmap.get("C_DESERT"))
        cfg.C_BOREAL = pg.Color(terrainmap.get("C_BOREAL"))
        cfg.C_TEMPERATE = pg.Color(terrainmap.get("C_TEMPERATE"))
        cfg.C_MEDITERRANEAN = pg.Color(terrainmap.get("C_MEDITERRANEAN"))
        cfg.C_TROPIC = pg.Color(terrainmap.get("C_TROPIC"))
        cfg.C_SHALLOW_WATER = pg.Color(terrainmap.get("C_SHALLOW_WATER"))
        cfg.C_DEEP_WATER = pg.Color(terrainmap.get("C_DEEP_WATER"))
        cfg.C_VOID = pg.Color(terrainmap.get("C_VOID"))

        heightmap: SectionProxy = config["colours.heightmap"]
        cfg.C_LOW = pg.Color(heightmap.get("C_LOW"))
        cfg.C_HIGH = pg.Color(heightmap.get("C_HIGH"))

        humiditymap: SectionProxy = config["colours.humiditymap"]
        cfg.H_H_LOW = humiditymap.getint("H_H_LOW")
        cfg.H_H_HIGH = humiditymap.getint("H_H_HIGH")
        cfg.H_S = humiditymap.getfloat("H_S")
        cfg.H_L = humiditymap.getfloat("H_L")

        temperaturemap: SectionProxy = config["colours.temperaturemap"]
        cfg.T_H_LOW = temperaturemap.getint("T_H_LOW")
        cfg.T_H_HIGH = temperaturemap.getint("T_H_HIGH")
        cfg.T_S = temperaturemap.getfloat("T_S")
        cfg.T_L = temperaturemap.getfloat("T_L")

        gradient: SectionProxy = config["colours.gradient"]
        cfg.C_G_1 = pg.Color(gradient.get("C_G_1"))
        cfg.C_G_2 = pg.Color(gradient.get("C_G_2"))
        cfg.C_G_3 = pg.Color(gradient.get("C_G_3"))
        cfg.C_G_4 = pg.Color(gradient.get("C_G_4"))


    @staticmethod
    def set_colour_scheme(i: int) -> None:
        config: ConfigParser = ConfigParser(converters={"v2": v2_converter, "clrtuple": clrtuple_converter})
        config.read(Path(__file__).parent / "../config/terrain.cfg")
        config.set("colours", "COLOUR_SCHEME", str(i))
        with open(Path(__file__).parent / "../config/terrain.cfg", "w") as configfile:
            config.write(configfile)

    @staticmethod
    def get_map_type_string() -> str:
        if cfg.COLOUR_SCHEME == ColourScheme.TERRAIN:
            return "Terrain"
        elif cfg.COLOUR_SCHEME == ColourScheme.HEIGHTMAP:
            return "Height"
        elif cfg.COLOUR_SCHEME == ColourScheme.HUMIDITYMAP:
            return "Humidity"
        elif cfg.COLOUR_SCHEME == ColourScheme.TEMPERATUREMAP:
            return "Temperature"
        else:
            return "ERROR"
        
