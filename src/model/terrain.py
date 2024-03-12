from bisect import bisect
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
    SAVANNA = 3
    DESERT = 4
    BOREAL = 5
    TEMPERATE = 6
    MEDITERRANEAN = 7
    TROPIC = 8
    SHALLOW_WATER = 9
    DEEP_WATER = 10
    VOID = 11

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

noise_dim: V2[int] = V2(cfg.HEX_NOF_HEXES.x(), cfg.HEX_NOF_HEXES.y())

def normal_noise(res: tuple[int, int]) -> NDArray: # type: ignore
    noise: FArray = pnp.generate_perlin_noise_2d( # type: ignore
        shape=(noise_dim.x(), noise_dim.y()),
        res=res,
        tileable=(True, False)
    )  # [-1, 1]
    return normalise(noise)  # [0, 1]

def ridge_noise(res: tuple[int, int]) -> tuple[FArray, FArray]: 
    noise: FArray = pnp.generate_perlin_noise_2d( # type: ignore
        shape=(noise_dim.x(), noise_dim.y()),
        res=res,
        tileable=(True, False)
    )  # [-1, 1]
    unmod_noise: FArray = noise.copy()
    # make ridges
    noise = -1.0 * np.abs(noise)  # [-1, 0]
    noise += 0.5  # [-0.5, 0.5]
    noise *= 2.0  # [-1, 1]
    return normalise(unmod_noise), normalise(noise)  # [0, 1]

def flatten(noise: Any, min_height: float, res: float) -> Any:
    return noise - ((noise - min_height) * res)


# ==== MAPPINGS ==== #

class TerrainMapping:

    def __init__(self) -> None:
        self.mapping: list[list[TerrainType]] = [[TerrainType.VOID for _ in range(len(TerrainTemperature))]  for _ in range(len(TerrainHumidity))]
        # humidity -> temperature
        self.mapping[TerrainHumidity.ARID][TerrainTemperature.FREEZING] = TerrainType.ARCTIC
        self.mapping[TerrainHumidity.ARID][TerrainTemperature.COLD] = TerrainType.TUNDRA
        self.mapping[TerrainHumidity.ARID][TerrainTemperature.AVERAGE] = TerrainType.STEPPE
        self.mapping[TerrainHumidity.ARID][TerrainTemperature.WARM] = TerrainType.SAVANNA
        self.mapping[TerrainHumidity.ARID][TerrainTemperature.HOT] = TerrainType.DESERT
        self.mapping[TerrainHumidity.AVERAGE][TerrainTemperature.FREEZING] = TerrainType.ARCTIC
        self.mapping[TerrainHumidity.AVERAGE][TerrainTemperature.COLD] = TerrainType.BOREAL
        self.mapping[TerrainHumidity.AVERAGE][TerrainTemperature.AVERAGE] = TerrainType.TEMPERATE
        self.mapping[TerrainHumidity.AVERAGE][TerrainTemperature.WARM] = TerrainType.MEDITERRANEAN
        self.mapping[TerrainHumidity.AVERAGE][TerrainTemperature.HOT] = TerrainType.SAVANNA
        self.mapping[TerrainHumidity.HUMID][TerrainTemperature.FREEZING] = TerrainType.ARCTIC
        self.mapping[TerrainHumidity.HUMID][TerrainTemperature.COLD] = TerrainType.BOREAL
        self.mapping[TerrainHumidity.HUMID][TerrainTemperature.AVERAGE] = TerrainType.TEMPERATE
        self.mapping[TerrainHumidity.HUMID][TerrainTemperature.WARM] = TerrainType.MEDITERRANEAN
        self.mapping[TerrainHumidity.HUMID][TerrainTemperature.HOT] = TerrainType.TROPIC

    def get_terrain_type(self, altitude: TerrainAltitude, humidity: TerrainHumidity, temperature: TerrainTemperature) -> TerrainType:
        if altitude in [TerrainAltitude.DEEP_WATER, TerrainAltitude.SHALLOW_WATER]:
            return TerrainType.DEEP_WATER if altitude == TerrainAltitude.DEEP_WATER else TerrainType.SHALLOW_WATER
        return self.mapping[humidity][temperature]


class TerrainAltitudeMapping:

    def __init__(self) -> None:
        H_HIGH: V2[float] = V2(cfg.A_HIGH, 999.0)
        H_MEDIUM: V2[float] = V2(cfg.A_MEDIUM, H_HIGH[0])
        H_LOW: V2[float] = V2(cfg.A_LOW, H_MEDIUM[0])
        H_SHALLOW_WATER: V2[float] = V2(cfg.A_SHALLOW_WATER, H_LOW[0])
        H_DEEP_WATER: V2[float] = V2(cfg.A_DEEP_WATER, H_SHALLOW_WATER[0])
        H_VOID: V2[float] = V2(-999.0, H_DEEP_WATER[0])

        self.mapping: dict[TerrainAltitude, V2[float]] = {
            TerrainAltitude.DEEP_WATER: H_DEEP_WATER,
            TerrainAltitude.SHALLOW_WATER: H_SHALLOW_WATER,
            TerrainAltitude.LOW: H_LOW, 
            TerrainAltitude.MEDIUM: H_MEDIUM, 
            TerrainAltitude.HIGH: H_HIGH 
        }

        self._keys: list[float] = [k[0] for k in self.mapping.values()]

    def get_terrain_altitude(self, altitude: float) -> TerrainAltitude:
        # assume that _keys is sorted (low to high)
        i: int = bisect(self._keys, altitude)
        i = 0 if i == 0 else i - 1
        return list(self.mapping)[i]
    
    def get_altitude(self, terrain_altitude: TerrainAltitude) -> V2[float]:
        return self.mapping[terrain_altitude]
    

class TerrainHumidityMapping:
    def __init__(self) -> None:
        H_HUMID: V2[float] = V2(cfg.H_HUMID, 1.0)
        H_AVERAGE: V2[float] = V2(cfg.H_AVERAGE, H_HUMID[0])
        H_ARID: V2[float] = V2(cfg.H_ARID, H_AVERAGE[0])

        self.mapping: dict[TerrainHumidity, V2[float]] = {
            TerrainHumidity.ARID: H_ARID, 
            TerrainHumidity.AVERAGE: H_AVERAGE, 
            TerrainHumidity.HUMID: H_HUMID
        }

        self._keys: list[float] = [k[0] for k in self.mapping.values()]

    def get_terrain_humidity(self, humidity: float) -> TerrainHumidity:
        # assume that _keys is sorted (low to high)
        i: int = bisect(self._keys, humidity)
        i = 0 if i == 0 else i - 1
        return list(self.mapping)[i]
    
    def get_humidity(self, terrain_humidity: TerrainHumidity) -> V2[float]:
        return self.mapping[terrain_humidity]  


class TerrainTemperatureMapping:
    
    def __init__(self) -> None:
        T_HOT: V2[float] = V2(cfg.T_HOT, 1.0)
        T_WARM: V2[float] = V2(cfg.T_WARM, T_HOT[0])
        T_AVERAGE: V2[float] = V2(cfg.T_AVERAGE, T_WARM[0])
        T_COLD: V2[float] = V2(cfg.T_COLD, T_AVERAGE[0])
        T_FREEZING: V2[float] = V2(cfg.T_FREEZING, T_COLD[0])

        self.mapping: dict[TerrainTemperature, V2[float]] = {
            TerrainTemperature.FREEZING: T_FREEZING, 
            TerrainTemperature.COLD: T_COLD, 
            TerrainTemperature.AVERAGE: T_AVERAGE, 
            TerrainTemperature.WARM: T_WARM,
            TerrainTemperature.HOT: T_HOT
        }

        self._keys: list[float] = [k[0] for k in self.mapping.values()]

    def get_terrain_temperature(self, temperature: float) -> TerrainTemperature:
        # assume that _keys is sorted (low to high)
        i: int = bisect(self._keys, temperature)
        i = 0 if i == 0 else i - 1
        return list(self.mapping)[i]
    
    def get_temperature(self, terrain_temperature: TerrainTemperature) -> V2[float]:
        return self.mapping[terrain_temperature]  


class TerrainColourMapping:

    def __init__(self) -> None:

        self._mapping: dict[TerrainType, pg.Color] = {
            TerrainType.ARCTIC: cfg.C_ARCTIC,
            TerrainType.TUNDRA: cfg.C_TUNDRA,
            TerrainType.STEPPE: cfg.C_STEPPE,
            TerrainType.SAVANNA: cfg.C_SAVANNA,
            TerrainType.DESERT: cfg.C_DESERT,
            TerrainType.BOREAL: cfg.C_BOREAL,
            TerrainType.TEMPERATE: cfg.C_TEMPERATE,
            TerrainType.MEDITERRANEAN: cfg.C_MEDITERRANEAN,
            TerrainType.TROPIC: cfg.C_TROPIC,
            TerrainType.SHALLOW_WATER: cfg.C_SHALLOW_WATER,
            TerrainType.DEEP_WATER: cfg.C_DEEP_WATER,
            TerrainType.VOID: cfg.C_VOID
        }

    def get_colour(self, terrain_type: TerrainType) -> pg.Color:
        return self._mapping[terrain_type]


# ==== MAPS === #

class TerrainHeightmap:
    
    def __init__(self) -> None:

        h_low: V2[float] = V2(cfg.A_LOW, cfg.A_HIGH)

        def ns_mask(mask_noise: FArray) -> FArray:
            mask: FArray = np.ones(noise_dim.get())
            mask[:, :noise_dim.y() // cfg.NS_MASK0_DIV] = cfg.NS_MASK0_VAL
            mask[:, :noise_dim.y() // cfg.NS_MASK1_DIV] = cfg.NS_MASK1_VAL
            mask[:, :noise_dim.y() // cfg.NS_MASK2_DIV] = cfg.NS_MASK2_VAL
            mask[:, noise_dim.y() - noise_dim.y() // cfg.NS_MASK0_DIV:] = cfg.NS_MASK0_VAL
            mask[:, noise_dim.y() - noise_dim.y() // cfg.NS_MASK1_DIV:] = cfg.NS_MASK1_VAL
            mask[:, noise_dim.y() - noise_dim.y() // cfg.NS_MASK2_DIV:] = cfg.NS_MASK2_VAL
            mask = gaussian_filter(mask, noise_dim.scalar_truediv(cfg.NS_MASK_GAUSSIAN_FILTER_SIZE).to_int().get()) # type: ignore
            assert isinstance(mask, NDArray)
            terrain0_noise: FArray = normal_noise(cfg.TERRAIN0_NOISE_FREQUENCY.get())
            terrain1_noise: FArray = normal_noise(cfg.TERRAIN1_NOISE_FREQUENCY.get())
            terrain2_noise: FArray = normal_noise(cfg.TERRAIN2_NOISE_FREQUENCY.get())
            mask += \
                (cfg.TERRAIN0_NOISE_WEIGHT * terrain0_noise) + \
                (cfg.TERRAIN1_NOISE_WEIGHT * terrain1_noise) + \
                (cfg.TERRAIN2_NOISE_WEIGHT * terrain2_noise)
            mask += cfg.NS_MASK_NOISE_WEIGHT * mask_noise
            mask = np.clip(mask, 0.0, 1.0)
            return mask

        # ==== CONTINENTS ==== #
        continent0_noise: FArray = normal_noise(cfg.CONTINENT_NOISE_FREQUENCY.get())
        continent1_noise: FArray = normal_noise(cfg.CONTINENT_NOISE_FREQUENCY.get())

        # combine two noise patterns
        continent_noise: FArray = normalise(continent0_noise * continent1_noise)

        # "raise" continents (skew distribution higher)
        continent_noise **= cfg.CONTINENT_NOISE_SIZE_MODIF

        # ==== TERRAIN ==== #
        terrain0_noise: FArray = normal_noise(cfg.TERRAIN0_NOISE_FREQUENCY.get())
        terrain1_noise: FArray = normal_noise(cfg.TERRAIN1_NOISE_FREQUENCY.get())
        terrain2_noise: FArray = normal_noise(cfg.TERRAIN2_NOISE_FREQUENCY.get())

        # ==== CONTINENTS ==== #
        # combine weighted noise patterns (total weight == 1)
        continents: FArray = \
            (cfg.CONTINENT_NOISE_WEIGHT * continent_noise) + \
            (cfg.TERRAIN0_NOISE_WEIGHT * terrain0_noise) + \
            (cfg.TERRAIN1_NOISE_WEIGHT * terrain1_noise) + \
            (cfg.TERRAIN2_NOISE_WEIGHT * terrain2_noise)

        # flatten peaks until max height is less than certain value
        mask: NDArray[bool, Any] = continents > h_low[0]
        max_height: float = h_low[1] - ((h_low[1] - h_low[0]) / cfg.CONTINENT_MAX_HEIGHT_DIV)
        while np.max(continents) > max_height:
            continents: FArray = np.where(mask, flatten(continents, h_low[0], cfg.CONTINENT_NOISE_PEAK_FLATTENING_RESOLUTION), continents)

        continents *= ns_mask(continents.copy())
    
        # ==== MOUNTAIN RANGES ==== #
        unmod_mountain_range_noise, mountain_range_noise = \
            ridge_noise(cfg.MOUNTAIN_RANGE_NOISE_FREQUENCY.get())

        # make ridges thicker
        mountain_range_noise **= cfg.MOUNTAIN_RANGE_NOISE_WIDTH_MODIF  # [0, 1] 

        # mask with heightmap
        mountain_range_noise *= ((continents + cfg.MOUNTAIN_MASK_SIZE_MODIF) ** cfg.MOUNTAIN_MASK_STRENGTH_MODIF)  # [0, ??]

        # ==== MOUNTAIN TERRAIN ==== #
        _, mountain0_noise = ridge_noise(cfg.MOUNTAIN0_NOISE_FREQUENCY.get())
        _, mountain1_noise = ridge_noise(cfg.MOUNTAIN1_NOISE_FREQUENCY.get())
        _, mountain2_noise = ridge_noise(cfg.MOUNTAIN2_NOISE_FREQUENCY.get())

        # ==== MOUNTAINS ==== #
        # combine weighted noise patterns (total weight == 1)
        mountains: FArray = \
            (cfg.MOUNTAIN0_NOISE_WEIGHT * mountain0_noise) + \
            (cfg.MOUNTAIN1_NOISE_WEIGHT * mountain1_noise) + \
            (cfg.MOUNTAIN2_NOISE_WEIGHT * mountain2_noise)
        
        # mask with mountain range pattern
        mountains *= mountain_range_noise
        mountains = normalise(mountains)

        # ==== HILLS RANGES ==== #
        # make hill ranges with the same noise as mountains
        hill_range_noise: FArray = mountain_range_noise.copy()

        # increase or decrease width ("amount") of hills
        hill_range_noise **= cfg.HILL_RANGE_NOISE_WIDTH_MODIF

        # mask with heightmap
        hill_range_noise *= ((continents + cfg.HILL_MASK_SIZE_MODIF) ** cfg.HILL_MASK_STRENGTH_MODIF)  # [0, ??]

        # ==== HILLS TERRAIN ==== #
        hill0_noise: FArray = normal_noise(cfg.HILL0_NOISE_FREQUENCY.get())
        hill1_noise: FArray = normal_noise(cfg.HILL1_NOISE_FREQUENCY.get())
        hill2_noise: FArray = normal_noise(cfg.HILL2_NOISE_FREQUENCY.get())

        # ==== HILLS ==== #
        # combine weighted noise patterns (total weight == 1)
        hills: FArray = \
            (cfg.HILL0_NOISE_WEIGHT * hill0_noise) + \
            (cfg.HILL1_NOISE_WEIGHT * hill1_noise) + \
            (cfg.HILL2_NOISE_WEIGHT * hill2_noise)
        
        # mask with hills range pattern
        hills *= hill_range_noise
        hills = normalise(hills)

        # flatten peaks until max height is less than certain value
        mask: NDArray[bool, Any] = hills > h_low[0]
        max_height: float = h_low[1] - ((h_low[1] - h_low[0]) / cfg.HILL_MAX_HEIGHT_DIV)
        while np.max(hills) > max_height:
            hills: FArray = np.where(mask, flatten(hills, h_low[0], cfg.HILL_NOISE_PEAK_FLATTENING_RESOLUTION), hills)

        # ==== HEIGHTMAP COMBINED ==== #
        # take the maximum height of heightmap or mountain_noise
        heightmap: FArray
        match cfg.TERRAIN_LAYER_SHOWN:
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
            case TerrainLayerShown.MOUNTAIN_RANGE_NOISE:
                heightmap = unmod_mountain_range_noise
           
        # ==== SET MAP AND GRADIENTS ==== #
        self._map: FArray = heightmap
        gradient: list[FArray] = np.gradient(heightmap)
        self._gradient_x: FArray = gradient[0]
        self._gradient_y: FArray = gradient[1]
        self._gradient_x_std: float = float(np.std(self._gradient_x))
        self._gradient_y_std: float = float(np.std(self._gradient_y))

        self.heightmap = heightmap
        self.continents = continents
        self.unmod_mountain_range_noise = unmod_mountain_range_noise
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

    def __init__(self, continents: FArray, mountains: FArray, heightmap: FArray) -> None:

        def ns_mask() -> FArray: 
            mask: FArray = np.ones(noise_dim.get())
            H_HUMID: float = cfg.H_HUMID + (1.0 - cfg.H_HUMID) / 2.0
            H_AVERAGE: float = cfg.H_AVERAGE + (cfg.H_HUMID - cfg.H_AVERAGE) / 2.0
            H_ARID: float = cfg.H_ARID + (cfg.H_AVERAGE - cfg.H_ARID) / 2.0

            mask[:, :int(noise_dim.y() * cfg.H_0)] = H_HUMID
            mask[:, :int(noise_dim.y() * cfg.H_1)] = H_ARID
            mask[:, :int(noise_dim.y() * cfg.H_2)] = H_AVERAGE
            mask[:, :int(noise_dim.y() * cfg.H_3)] = H_HUMID
            mask[:, :int(noise_dim.y() * cfg.H_4)] = H_AVERAGE
            mask[:, int(noise_dim.y() - noise_dim.y() * cfg.H_0):] = 1.0
            mask[:, int(noise_dim.y() - noise_dim.y() * cfg.H_1):] = H_ARID
            mask[:, int(noise_dim.y() - noise_dim.y() * cfg.H_2):] = H_AVERAGE
            mask[:, int(noise_dim.y() - noise_dim.y() * cfg.H_3):] = H_HUMID
            mask[:, int(noise_dim.y() - noise_dim.y() * cfg.H_4):] = H_AVERAGE

            # arid continent interiors, humid continent edges
            interior_mask: FArray = continents.copy()
            interior_mask *= -1.0  # [-1, 0] (invert: -1 = high, 0 = low)
            interior_mask += cfg.H_CONTINENT_MASK_THRESHOLD  # [-1 + T, T]
            mask += cfg.H_CONTINENT_MASK_NOISE_WEIGHT * interior_mask
            mask = normalise(mask)

            # blur
            mask = gaussian_filter(mask, noise_dim.scalar_truediv(cfg.NS_MASK_GAUSSIAN_FILTER_SIZE).to_int().get()) # type: ignore
            assert isinstance(mask, NDArray)

            # follow mountain noise (different sides of ridges)
            mountain_mask: FArray = mountains.copy()
            mountain_mask -= cfg.H_MOUNTAIN_MASK_THRESHOLD
            mask += cfg.H_MOUNTAIN_MASK_NOISE_WEIGHT * mountain_mask
            mask = normalise(mask)

            # dryer (lower mask) at higher altitudes (above midway A_LOW/A_MEDIUM)
            interior_mask: FArray = heightmap.copy()
            interior_mask -= cfg.A_LOW + ((cfg.A_MEDIUM - cfg.A_LOW) / 2.0)
            interior_mask = np.clip(interior_mask, 0.0, 1.0)
            mask -= cfg.H_HEIGHTMAP_MASK_NOISE_WEIGHT * interior_mask
            mask = np.clip(mask, 0.0, 1.0)

            # terrain noise
            terrain0_noise: FArray = normal_noise(cfg.TERRAIN0_NOISE_FREQUENCY.get())
            terrain1_noise: FArray = normal_noise(cfg.TERRAIN1_NOISE_FREQUENCY.get())
            terrain2_noise: FArray = normal_noise(cfg.TERRAIN2_NOISE_FREQUENCY.get())
            mask = (cfg.CONTINENT_NOISE_WEIGHT * mask) + \
                (cfg.TERRAIN1_NOISE_WEIGHT * terrain0_noise) + \
                (cfg.TERRAIN1_NOISE_WEIGHT * terrain1_noise) + \
                (cfg.TERRAIN2_NOISE_WEIGHT * terrain2_noise)
            
            return mask

        self._map: FArray = ns_mask()
        
    def get_humidity_from_of(self, of: V2[int]) -> float:
        return  self._map[of.x()][of.y()]


class TerrainTemperaturemap:

    def __init__(self, heightmap: FArray,) -> None:

        def ns_mask() -> FArray:  # TODO magic
            mask: FArray = np.ones(noise_dim.get())
            mask[:, :int(noise_dim.y() * cfg.T_0)] = cfg.T_HOT
            mask[:, :int(noise_dim.y() * cfg.T_1)] = cfg.T_WARM
            mask[:, :int(noise_dim.y() * cfg.T_2)] = cfg.T_AVERAGE
            mask[:, :int(noise_dim.y() * cfg.T_3)] = cfg.T_COLD
            mask[:, :int(noise_dim.y() * cfg.T_4)] = cfg.T_FREEZING
            mask[:, int(noise_dim.y() - noise_dim.y() * cfg.T_0):] = cfg.T_HOT
            mask[:, int(noise_dim.y() - noise_dim.y() * cfg.T_1):] = cfg.T_WARM
            mask[:, int(noise_dim.y() - noise_dim.y() * cfg.T_2):] = cfg.T_AVERAGE
            mask[:, int(noise_dim.y() - noise_dim.y() * cfg.T_3):] = cfg.T_COLD
            mask[:, int(noise_dim.y() - noise_dim.y() * cfg.T_4):] = cfg.T_FREEZING

            # blur
            mask = gaussian_filter(mask, noise_dim.scalar_truediv(cfg.NS_MASK_GAUSSIAN_FILTER_SIZE).to_int().get()) # type: ignore
            assert isinstance(mask, NDArray)

            # colder (lower mask) at higher altitudes (above A_LOW)
            interior_mask: FArray = heightmap.copy()
            interior_mask -= cfg.A_LOW
            interior_mask = np.clip(interior_mask, 0.0, 1.0)
            mask -= cfg.T_CONTINENT_MASK_NOISE_WEIGHT * interior_mask
            mask = np.clip(mask, 0.0, 1.0)

            # terrain noise
            terrain0_noise: FArray = normal_noise(cfg.TERRAIN0_NOISE_FREQUENCY.get())
            terrain1_noise: FArray = normal_noise(cfg.TERRAIN1_NOISE_FREQUENCY.get())
            terrain2_noise: FArray = normal_noise(cfg.TERRAIN2_NOISE_FREQUENCY.get())
            mask = (cfg.CONTINENT_NOISE_WEIGHT * mask) + \
                (cfg.TERRAIN1_NOISE_WEIGHT * terrain0_noise) + \
                (cfg.TERRAIN1_NOISE_WEIGHT * terrain1_noise) + \
                (cfg.TERRAIN2_NOISE_WEIGHT * terrain2_noise)
            mask = normalise(mask)

            return mask

        self._map: FArray = ns_mask()
        
    def get_temperature_from_of(self, of: V2[int]) -> float:
        return  self._map[of.x()][of.y()]