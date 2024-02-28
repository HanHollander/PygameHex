from enum import Enum
from math import ceil, sqrt
from typing import Any, Callable, TypeVar
from nptyping import NDArray
import numpy as np
import pygame as pg
import perlin_numpy as pnp
from config import *
from scipy.ndimage import gaussian_filter # type: ignore
from util import V2, V3, FArray, normalise

# https://www.redblobgames.com/maps/terrain-from-noise/ 


# ==== ENUMS === #

class TerrainType(Enum):
    ARCTIC = 0
    TUNDRA = 1
    STEPPE = 2
    GRASSLANDS = 3
    SAVANNA = 4
    DESERT = 5
    BOREAL = 6
    TEMPERATE = 7
    MEDITERRANEAN = 8
    TROPIC = 9
    SHALLOW_WATER = 10
    DEEP_WATER = 11
    VOID = 12

class TerrainAltitude(int, Enum):
    HIGH = 0
    MEDIUM = 1
    LOW = 2
    SHALLOW_WATER = 3
    DEEP_WATER = 4

class TerrainHumidity(int, Enum):
    ARID = 0
    AVERAGE = 1
    HUMID = 2

class TerrainTemperature(int, Enum):
    FREEZING = 0
    COLD = 1
    AVERAGE = 2
    WARM = 3
    HOT = 4


# ==== HELPER FUNCTIONS ==== #

noise_dim: V2[int] = V2(Cfg.HEX_NOF_HEXES.x(), Cfg.HEX_NOF_HEXES.y())

def normal_noise(res: tuple[int, int]) -> NDArray: # type: ignore
    noise: FArray = pnp.generate_perlin_noise_2d( # type: ignore
        shape=(noise_dim.x(), noise_dim.y()),
        res=res,
        tileable=(True, False)
    )  # [-1, 1]
    return normalise(noise)  # [0, 1]

def ridge_noise(res: tuple[int, int]) -> NDArray: # type: ignore
    noise: FArray = pnp.generate_perlin_noise_2d( # type: ignore
        shape=(noise_dim.x(), noise_dim.y()),
        res=res,
        tileable=(True, False)
    )  # [-1, 1]
    # make ridges
    noise = -1 * np.abs(noise)  # [-1, 0]
    noise += 0.5  # [-0.5, 0.5]
    noise *= 2  # [-1, 1]
    return normalise(noise)  # [0, 1]

def flatten(noise: Any, min_height: float, res: float) -> Any:
    return noise - ((noise - min_height) * res)


# ==== MAPPINGS ==== #

class TerrainMapping:

    def __init__(self) -> None:
        ARCTIC = TerrainType.ARCTIC
        TUNDRA = TerrainType.TUNDRA
        STEPPE = TerrainType.STEPPE
        GRASSLANDS = TerrainType.GRASSLANDS
        SAVANNA = TerrainType.SAVANNA
        DESERT = TerrainType.DESERT
        BOREAL = TerrainType.BOREAL
        TEMPERATE = TerrainType.TEMPERATE
        MEDITERRANEAN = TerrainType.MEDITERRANEAN
        TROPIC = TerrainType.TROPIC
        SHALLOW_WATER = TerrainType.SHALLOW_WATER
        DEEP_WATER = TerrainType.DEEP_WATER

        self._mapping: list[list[list[TerrainType]]] = [[[] for _ in range(len(TerrainAltitude))] for _ in range(len(TerrainHumidity))]
        # humidity -> altitude -> temperature
        # TODO magic
        self._mapping[TerrainHumidity.ARID][TerrainAltitude.LOW] = [ARCTIC, TUNDRA, STEPPE, SAVANNA, DESERT]
        self._mapping[TerrainHumidity.ARID][TerrainAltitude.MEDIUM] = [ARCTIC, ARCTIC, STEPPE, SAVANNA, DESERT]
        self._mapping[TerrainHumidity.ARID][TerrainAltitude.HIGH] = [ARCTIC, ARCTIC, ARCTIC, DESERT, DESERT]
        self._mapping[TerrainHumidity.ARID][TerrainAltitude.SHALLOW_WATER] = [ARCTIC, SHALLOW_WATER, SHALLOW_WATER, SHALLOW_WATER, SHALLOW_WATER]
        self._mapping[TerrainHumidity.ARID][TerrainAltitude.DEEP_WATER] = [ARCTIC, DEEP_WATER, DEEP_WATER, DEEP_WATER, DEEP_WATER]
        self._mapping[TerrainHumidity.AVERAGE][TerrainAltitude.LOW] = [ARCTIC, BOREAL, TEMPERATE, MEDITERRANEAN, SAVANNA]
        self._mapping[TerrainHumidity.AVERAGE][TerrainAltitude.MEDIUM] = [ARCTIC, TUNDRA, GRASSLANDS, MEDITERRANEAN, SAVANNA]
        self._mapping[TerrainHumidity.AVERAGE][TerrainAltitude.HIGH] = [ARCTIC, ARCTIC, ARCTIC, ARCTIC, GRASSLANDS]
        self._mapping[TerrainHumidity.AVERAGE][TerrainAltitude.SHALLOW_WATER] = [ARCTIC, SHALLOW_WATER, SHALLOW_WATER, SHALLOW_WATER, SHALLOW_WATER]
        self._mapping[TerrainHumidity.AVERAGE][TerrainAltitude.DEEP_WATER] = [ARCTIC, DEEP_WATER, DEEP_WATER, DEEP_WATER, DEEP_WATER]
        self._mapping[TerrainHumidity.HUMID][TerrainAltitude.LOW] = [ARCTIC, BOREAL, TEMPERATE, TROPIC, TROPIC]
        self._mapping[TerrainHumidity.HUMID][TerrainAltitude.MEDIUM] = [ARCTIC, BOREAL, TEMPERATE, TROPIC, TROPIC]
        self._mapping[TerrainHumidity.HUMID][TerrainAltitude.HIGH] = [ARCTIC, ARCTIC, ARCTIC, ARCTIC, GRASSLANDS]
        self._mapping[TerrainHumidity.HUMID][TerrainAltitude.SHALLOW_WATER] = [ARCTIC, SHALLOW_WATER, SHALLOW_WATER, SHALLOW_WATER, SHALLOW_WATER]
        self._mapping[TerrainHumidity.HUMID][TerrainAltitude.DEEP_WATER] = [ARCTIC, DEEP_WATER, DEEP_WATER, DEEP_WATER, DEEP_WATER]

    def get_terrain_type(self, humidity: TerrainHumidity, altitude: TerrainAltitude, temperature: TerrainTemperature) -> TerrainType:
        return self._mapping[humidity][altitude][temperature]


class TerrainAltitudeMapping:

    def __init__(self) -> None:
        H_HIGH: V2[float] = V2(Cfg.A_HIGH, 999.0)
        H_MEDIUM: V2[float] = V2(Cfg.A_MEDIUM, H_HIGH[0])
        H_LOW: V2[float] = V2(Cfg.A_LOW, H_MEDIUM[0])
        H_SHALLOW_WATER: V2[float] = V2(Cfg.A_SHALLOW_WATER, H_LOW[0])
        H_DEEP_WATER: V2[float] = V2(Cfg.A_DEEP_WATER, H_SHALLOW_WATER[0])
        H_VOID: V2[float] = V2(-999.0, H_DEEP_WATER[0])

        self._mapping: dict[TerrainAltitude, V2[float]] = {
            TerrainAltitude.HIGH: H_HIGH, 
            TerrainAltitude.MEDIUM: H_MEDIUM, 
            TerrainAltitude.LOW: H_LOW, 
            TerrainAltitude.SHALLOW_WATER: H_SHALLOW_WATER,
            TerrainAltitude.DEEP_WATER: H_DEEP_WATER
        }

    def get_terrain_altitude(self, altitude: float) -> TerrainAltitude:
        for kv in self._mapping.items():
            if altitude >= kv[1][0] and altitude < kv[1][1]:
                return kv[0]
        return TerrainAltitude.DEEP_WATER
    
    def get_altitude(self, terrain_altitude: TerrainAltitude) -> V2[float]:
        return self._mapping[terrain_altitude]
    

class TerrainHumidityMapping:
    def __init__(self) -> None:
        H_ARID: V2[float] = V2(Cfg.H_ARID, 999.0)
        H_AVERAGE: V2[float] = V2(Cfg.H_AVERAGE, H_ARID[0])
        H_HUMID: V2[float] = V2(Cfg.H_HUMID, H_AVERAGE[0])

        self._mapping: dict[TerrainHumidity, V2[float]] = {
            TerrainHumidity.ARID: H_ARID, 
            TerrainHumidity.AVERAGE: H_AVERAGE, 
            TerrainHumidity.HUMID: H_HUMID
        }

    def get_terrain_humidity(self, humidity: float) -> TerrainHumidity:
        for kv in self._mapping.items():
            if humidity >= kv[1][0] and humidity < kv[1][1]:
                return kv[0]
        return TerrainHumidity.AVERAGE
    
    def get_humidity(self, terrain_humidity: TerrainHumidity) -> V2[float]:
        return self._mapping[terrain_humidity]  


class TerrainTemperatureMapping:
    
    def __init__(self) -> None:
        T_FREEZING: V2[float] = V2(Cfg.T_FREEZING, 999.0)
        T_COLD: V2[float] = V2(Cfg.T_COLD, T_FREEZING[0])
        T_AVERAGE: V2[float] = V2(Cfg.T_AVERAGE, T_COLD[0])
        T_WARM: V2[float] = V2(Cfg.T_WARM, T_AVERAGE[0])
        T_HOT: V2[float] = V2(Cfg.T_HOT, T_WARM[0])

        self._mapping: dict[TerrainTemperature, V2[float]] = {
            TerrainTemperature.FREEZING: T_FREEZING, 
            TerrainTemperature.COLD: T_COLD, 
            TerrainTemperature.AVERAGE: T_AVERAGE, 
            TerrainTemperature.WARM: T_WARM,
            TerrainTemperature.HOT: T_HOT
        }

    def get_terrain_temperature(self, temperature: float) -> TerrainTemperature:
        for kv in self._mapping.items():
            if temperature >= kv[1][0] and temperature < kv[1][1]:
                return kv[0]
        return TerrainTemperature.HOT
    
    def get_temperature(self, terrain_temperature: TerrainTemperature) -> V2[float]:
        return self._mapping[terrain_temperature]  


class TerrainColourMapping:

    def __init__(self) -> None:

        self._mapping: dict[TerrainType, tuple[pg.Color, pg.Color]] = {
            TerrainType.ARCTIC: Cfg.C_ARCTIC,
            TerrainType.TUNDRA: Cfg.C_TUNDRA,
            TerrainType.STEPPE: Cfg.C_STEPPE,
            TerrainType.GRASSLANDS: Cfg.C_GRASSLANDS,
            TerrainType.SAVANNA: Cfg.C_SAVANNA,
            TerrainType.DESERT: Cfg.C_DESERT,
            TerrainType.BOREAL: Cfg.C_BOREAL,
            TerrainType.TEMPERATE: Cfg.C_TEMPERATE,
            TerrainType.MEDITERRANEAN: Cfg.C_MEDITERRANEAN,
            TerrainType.TROPIC: Cfg.C_TROPIC,
            TerrainType.SHALLOW_WATER: Cfg.C_SHALLOW_WATER,
            TerrainType.DEEP_WATER: Cfg.C_DEEP_WATER,
            TerrainType.VOID: Cfg.C_VOID
        }

    def get_colour(self, type: TerrainType) -> tuple[pg.Color, pg.Color]:
        return self._mapping[type]


# ==== MAPS === #

class TerrainHeightmap:
    
    def __init__(self) -> None:

        h_low: V2[float] = V2(Cfg.A_LOW, Cfg.A_HIGH)

        def ns_mask(mask_noise: FArray) -> FArray:
            mask: FArray = np.ones(noise_dim.get())
            mask[:, :noise_dim.y() // Cfg.NS_MASK0_DIV] = Cfg.NS_MASK0_VAL
            mask[:, :noise_dim.y() // Cfg.NS_MASK1_DIV] = Cfg.NS_MASK1_VAL
            mask[:, :noise_dim.y() // Cfg.NS_MASK2_DIV] = Cfg.NS_MASK2_VAL
            mask[:, noise_dim.y() - noise_dim.y() // Cfg.NS_MASK0_DIV:] = Cfg.NS_MASK0_VAL
            mask[:, noise_dim.y() - noise_dim.y() // Cfg.NS_MASK1_DIV:] = Cfg.NS_MASK1_VAL
            mask[:, noise_dim.y() - noise_dim.y() // Cfg.NS_MASK2_DIV:] = Cfg.NS_MASK2_VAL
            mask = gaussian_filter(mask, noise_dim.scalar_truediv(Cfg.NS_MASK_GAUSSIAN_FILTER_SIZE).to_int().get()) # type: ignore
            assert isinstance(mask, NDArray)
            terrain0_noise: FArray = normal_noise(Cfg.TERRAIN0_NOISE_FREQUENCY.get())
            terrain1_noise: FArray = normal_noise(Cfg.TERRAIN1_NOISE_FREQUENCY.get())
            terrain2_noise: FArray = normal_noise(Cfg.TERRAIN2_NOISE_FREQUENCY.get())
            mask += \
                (Cfg.TERRAIN0_NOISE_WEIGHT * terrain0_noise) + \
                (Cfg.TERRAIN1_NOISE_WEIGHT * terrain1_noise) + \
                (Cfg.TERRAIN2_NOISE_WEIGHT * terrain2_noise)
            mask += Cfg.NS_MASK_NOISE_WEIGHT * mask_noise
            mask = np.clip(mask, 0.0, 1.0)
            return mask

        # ==== CONTINENTS ==== #
        continent0_noise: FArray = normal_noise(Cfg.CONTINENT_NOISE_FREQUENCY.get())
        continent1_noise: FArray = normal_noise(Cfg.CONTINENT_NOISE_FREQUENCY.get())

        # combine two noise patterns
        continent_noise: FArray = normalise(continent0_noise * continent1_noise)

        # "raise" continents (skew distribution higher)
        continent_noise **= Cfg.CONTINENT_NOISE_SIZE_MODIF

        # ==== TERRAIN ==== #
        terrain0_noise: FArray = normal_noise(Cfg.TERRAIN0_NOISE_FREQUENCY.get())
        terrain1_noise: FArray = normal_noise(Cfg.TERRAIN1_NOISE_FREQUENCY.get())
        terrain2_noise: FArray = normal_noise(Cfg.TERRAIN2_NOISE_FREQUENCY.get())

        # ==== CONTINENTS ==== #
        # combine weighted noise patterns (total weight == 1)
        continents: FArray = \
            (Cfg.CONTINENT_NOISE_WEIGHT * continent_noise) + \
            (Cfg.TERRAIN0_NOISE_WEIGHT * terrain0_noise) + \
            (Cfg.TERRAIN1_NOISE_WEIGHT * terrain1_noise) + \
            (Cfg.TERRAIN2_NOISE_WEIGHT * terrain2_noise)

        # flatten peaks until max height is less than certain value
        mask: NDArray[bool, Any] = continents > h_low[0]
        max_height: float = h_low[1] - ((h_low[1] - h_low[0]) / Cfg.CONTINENT_MAX_HEIGHT_DIV)
        while np.max(continents) > max_height:
            continents: FArray = np.where(mask, flatten(continents, h_low[0], Cfg.CONTINENT_NOISE_PEAK_FLATTENING_RESOLUTION), continents)

        continents *= ns_mask(continents.copy())
    
        # ==== MOUNTAIN RANGES ==== #
        mountain_range_noise: FArray = ridge_noise(Cfg.MOUNTAIN_RANGE_NOISE_FREQUENCY.get())

        # make ridges thicker
        mountain_range_noise **= Cfg.MOUNTAIN_RANGE_NOISE_WIDTH_MODIF  # [0, 1] 

        # mask with heightmap
        mountain_range_noise *= ((continents + Cfg.MOUNTAIN_MASK_SIZE_MODIF) ** Cfg.MOUNTAIN_MASK_STRENGTH_MODIF)  # [0, ??]

        # ==== MOUNTAIN TERRAIN ==== #
        mountain0_noise: FArray = ridge_noise(Cfg.MOUNTAIN0_NOISE_FREQUENCY.get())
        mountain1_noise: FArray = ridge_noise(Cfg.MOUNTAIN1_NOISE_FREQUENCY.get())
        mountain2_noise: FArray = ridge_noise(Cfg.MOUNTAIN2_NOISE_FREQUENCY.get())

        # ==== MOUNTAINS ==== #
        # combine weighted noise patterns (total weight == 1)
        mountains: FArray = \
            (Cfg.MOUNTAIN0_NOISE_WEIGHT * mountain0_noise) + \
            (Cfg.MOUNTAIN1_NOISE_WEIGHT * mountain1_noise) + \
            (Cfg.MOUNTAIN2_NOISE_WEIGHT * mountain2_noise)
        
        # mask with mountain range pattern
        mountains *= mountain_range_noise
        mountains = normalise(mountains)

        # ==== HILLS RANGES ==== #
        # make hill ranges with the same noise as mountains
        hill_range_noise: FArray = mountain_range_noise.copy()

        # increase or decrease width ("amount") of hills
        hill_range_noise **= Cfg.HILL_RANGE_NOISE_WIDTH_MODIF

        # mask with heightmap
        hill_range_noise *= ((continents + Cfg.HILL_MASK_SIZE_MODIF) ** Cfg.HILL_MASK_STRENGTH_MODIF)  # [0, ??]

        # ==== HILLS TERRAIN ==== #
        hill0_noise: FArray = normal_noise(Cfg.HILL0_NOISE_FREQUENCY.get())
        hill1_noise: FArray = normal_noise(Cfg.HILL1_NOISE_FREQUENCY.get())
        hill2_noise: FArray = normal_noise(Cfg.HILL2_NOISE_FREQUENCY.get())

        # ==== HILLS ==== #
        # combine weighted noise patterns (total weight == 1)
        hills: FArray = \
            (Cfg.HILL0_NOISE_WEIGHT * hill0_noise) + \
            (Cfg.HILL1_NOISE_WEIGHT * hill1_noise) + \
            (Cfg.HILL2_NOISE_WEIGHT * hill2_noise)
        
        # mask with hills range pattern
        hills *= hill_range_noise
        hills = normalise(hills)

        # flatten peaks until max height is less than certain value
        mask: NDArray[bool, Any] = hills > h_low[0]
        max_height: float = h_low[1] - ((h_low[1] - h_low[0]) / Cfg.HILL_MAX_HEIGHT_DIV)
        while np.max(hills) > max_height:
            hills: FArray = np.where(mask, flatten(hills, h_low[0], Cfg.HILL_NOISE_PEAK_FLATTENING_RESOLUTION), hills)

        # ==== HEIGHTMAP COMBINED ==== #
        # take the maximum height of heightmap or mountain_noise
        heightmap: FArray
        match Cfg.TERRAIN_LAYER_SHOWN:
            case TerrainLayerShown.CONTINENTS:
                heightmap = continents
            case TerrainLayerShown.MOUNTAINS:
                heightmap = mountains
            case TerrainLayerShown.HILLS:
                heightmap = hills
            case TerrainLayerShown.HEIGHTMAP:
                heightmap = np.maximum(np.maximum(continents, mountains), hills)
            case TerrainLayerShown.CONTINENTS_MOUNTAINS:
                heightmap = np.maximum(continents, mountains)
            case TerrainLayerShown.CONTINENTS_HILLS:
                heightmap = np.maximum(continents, hills)
            case TerrainLayerShown.MOUNTAINS_HILLS:
                heightmap = np.maximum(mountains, hills)
            case TerrainLayerShown.CONTINENT_MASK:
                heightmap = ns_mask(continents.copy())
           
        # ==== SET MAP AND GRADIENTS ==== #
        self._map: FArray = heightmap
        gradient: list[FArray] = np.gradient(heightmap)
        self._gradient_x: FArray = gradient[0]
        self._gradient_y: FArray = gradient[1]
        self._gradient_x_std: float = float(np.std(self._gradient_x))
        self._gradient_y_std: float = float(np.std(self._gradient_y))

        self.continents = continents
        self.mountains = mountains
        self.hills = hills

    def get_altitude_from_of(self, of: V2[int]) -> float:
        return  self._map[of.x()][of.y()]
    
    def get_x_gradient_from_of(self, of: V2[int]) -> float:   
        return self._gradient_x[of.x()][of.y()]
    def get_y_gradient_from_of(self, of: V2[int]) -> float:   
        return self._gradient_y[of.x()][of.y()]
    def get_gradient_std(self) -> V2[float]:
        return V2(2.0 * self._gradient_x_std, 2.0 * self._gradient_y_std)
    

class TerrainHumiditymap:

    def __init__(self, continents: FArray) -> None:

        def ns_mask() -> FArray:  # TODO magic
            mask: FArray = np.ones(noise_dim.get())
            mask[:, :int(noise_dim.y() * Cfg.H_0)] = Cfg.H_HUMID
            mask[:, :int(noise_dim.y() * Cfg.H_1)] = Cfg.H_ARID
            mask[:, :int(noise_dim.y() * Cfg.H_2)] = Cfg.H_AVERAGE
            mask[:, :int(noise_dim.y() * Cfg.H_3)] = Cfg.H_HUMID
            mask[:, :int(noise_dim.y() * Cfg.H_4)] = Cfg.H_AVERAGE
            mask[:, int(noise_dim.y() - noise_dim.y() * Cfg.H_0):] = Cfg.H_HUMID
            mask[:, int(noise_dim.y() - noise_dim.y() * Cfg.H_1):] = Cfg.H_ARID
            mask[:, int(noise_dim.y() - noise_dim.y() * Cfg.H_2):] = Cfg.H_AVERAGE
            mask[:, int(noise_dim.y() - noise_dim.y() * Cfg.H_3):] = Cfg.H_HUMID
            mask[:, int(noise_dim.y() - noise_dim.y() * Cfg.H_4):] = Cfg.H_AVERAGE
            mask = gaussian_filter(mask, noise_dim.scalar_truediv(Cfg.NS_MASK_GAUSSIAN_FILTER_SIZE).to_int().get()) # type: ignore
            assert isinstance(mask, NDArray)
            terrain0_noise: FArray = normal_noise(Cfg.TERRAIN0_NOISE_FREQUENCY.get())
            terrain1_noise: FArray = normal_noise(Cfg.TERRAIN1_NOISE_FREQUENCY.get())
            terrain2_noise: FArray = normal_noise(Cfg.TERRAIN2_NOISE_FREQUENCY.get())
            mask += \
                (0.25 * terrain0_noise) + \
                (Cfg.TERRAIN1_NOISE_WEIGHT * terrain1_noise) + \
                (Cfg.TERRAIN2_NOISE_WEIGHT * terrain2_noise)
            mask += Cfg.H_MASK_NOISE_WEIGHT * continents
            mask = normalise(mask)
            return mask

        self._map: FArray = ns_mask()
        
    def get_humidity_from_of(self, of: V2[int]) -> float:
        return  self._map[of.x()][of.y()]


class TerrainTemperaturemap:

    def __init__(self) -> None:

        def ns_mask() -> FArray:  # TODO magic
            mask: FArray = np.ones(noise_dim.get())
            mask[:, :int(noise_dim.y() * Cfg.T_0)] = Cfg.T_HOT
            mask[:, :int(noise_dim.y() * Cfg.T_1)] = Cfg.T_WARM
            mask[:, :int(noise_dim.y() * Cfg.T_2)] = Cfg.T_AVERAGE
            mask[:, :int(noise_dim.y() * Cfg.T_3)] = Cfg.T_COLD
            mask[:, :int(noise_dim.y() * Cfg.T_4)] = Cfg.T_FREEZING
            mask[:, int(noise_dim.y() - noise_dim.y() * Cfg.T_0):] = Cfg.T_HOT
            mask[:, int(noise_dim.y() - noise_dim.y() * Cfg.T_1):] = Cfg.T_WARM
            mask[:, int(noise_dim.y() - noise_dim.y() * Cfg.T_2):] = Cfg.T_AVERAGE
            mask[:, int(noise_dim.y() - noise_dim.y() * Cfg.T_3):] = Cfg.T_COLD
            mask[:, int(noise_dim.y() - noise_dim.y() * Cfg.T_4):] = Cfg.T_FREEZING
            mask = gaussian_filter(mask, noise_dim.scalar_truediv(Cfg.NS_MASK_GAUSSIAN_FILTER_SIZE).to_int().get()) # type: ignore
            assert isinstance(mask, NDArray)
            terrain0_noise: FArray = normal_noise(Cfg.TERRAIN0_NOISE_FREQUENCY.get())
            terrain1_noise: FArray = normal_noise(Cfg.TERRAIN1_NOISE_FREQUENCY.get())
            terrain2_noise: FArray = normal_noise(Cfg.TERRAIN2_NOISE_FREQUENCY.get())
            mask += \
                (Cfg.TERRAIN0_NOISE_WEIGHT * terrain0_noise) + \
                (Cfg.TERRAIN1_NOISE_WEIGHT * terrain1_noise) + \
                (Cfg.TERRAIN2_NOISE_WEIGHT * terrain2_noise)
            mask = normalise(mask)
            return mask

        self._map: FArray = ns_mask()
        
    def get_temperature_from_of(self, of: V2[int]) -> float:
        return  self._map[of.x()][of.y()]