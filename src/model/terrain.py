from enum import Enum
from math import ceil, sqrt
from typing import Any, Callable
from nptyping import NDArray
import numpy as np
import pygame as pg
import perlin_numpy as pnp
from config import *
from util import V2, V3, normalise

# https://www.redblobgames.com/maps/terrain-from-noise/ 


class TerrainType(str, Enum):
    SNOW = "SNOW"
    MOUNTAIN = "MOUNTAIN"
    HILL = "HILL"
    FOREST = "FOREST"
    BEACH = "BEACH"
    SHALLOW_OCEAN = "SHALLOW_OCEAN"
    DEEP_OCEAN = "DEEP_OCEAN"
    VOID = "VOID"


class TerrainColourMapping:

    def __init__(self) -> None:
        clr = pg.Color
        C_SNOW: tuple[pg.Color, pg.Color] = (clr("#fffafa"), clr("#fffafa"))        
        # C_SNOW: tuple[pg.Color, pg.Color] = (clr("#353535"), clr("#fffafa"))
        C_MOUNTAIN: tuple[pg.Color, pg.Color] = (clr("#7f867e"), clr("#d8d8d8"))
        C_HILL: tuple[pg.Color, pg.Color] = (clr("#293b21"), clr("#718873"))
        C_FOREST: tuple[pg.Color, pg.Color] = (clr("#17270f"), clr("#5e7a47"))
        C_BEACH: tuple[pg.Color, pg.Color] = (clr("#ebeec3"), clr("#ebeec3"))
        C_SHALLOW_OCEAN: tuple[pg.Color, pg.Color] = (clr("#6281b9"), clr("#9ac2dd"))
        C_DEEP_OCEAN: tuple[pg.Color, pg.Color] = (clr("#172236"), clr("#5471a7"))
        C_VOID: tuple[pg.Color, pg.Color] = (clr("#000000"), clr("#000000"))

        self._mapping: dict[TerrainType, tuple[pg.Color, pg.Color]] = {
            TerrainType.SNOW: C_SNOW,
            TerrainType.MOUNTAIN: C_MOUNTAIN,
            TerrainType.HILL: C_HILL,
            TerrainType.FOREST: C_FOREST,
            TerrainType.BEACH: C_BEACH,
            TerrainType.SHALLOW_OCEAN: C_SHALLOW_OCEAN,
            TerrainType.DEEP_OCEAN: C_DEEP_OCEAN,
            TerrainType.VOID: C_VOID
        }

    def get_colour(self, type: TerrainType) -> tuple[pg.Color, pg.Color]:
        return self._mapping[type]
    


class TerrainHeightmap:
    
    def __init__(self) -> None:
        H_SNOW = V2(0.95, 999.0)
        H_MOUNTAIN = V2(0.79, H_SNOW[0])
        H_HILL = V2(0.50, H_MOUNTAIN[0])
        H_SHALLOW_OCEAN = V2(0.45, H_HILL[0])
        H_DEEP_OCEAN = V2(0.0, H_SHALLOW_OCEAN[0])
        H_VOID = V2(-999.0, H_DEEP_OCEAN[0])

        # H_SNOW = V2(0.0, 1.0)
        # H_MOUNTAIN = V2(-995.0, H_SNOW[0])
        # H_HILL = V2(-996.0, H_MOUNTAIN[0])
        # H_SHALLOW_OCEAN = V2(-997.0, H_HILL[0])
        # H_DEEP_OCEAN = V2(-998.0, H_SHALLOW_OCEAN[0])
        # H_VOID = V2(-999.0, H_DEEP_OCEAN[0])

        self._mapping: dict[TerrainType, V2[float]] = {
            TerrainType.SNOW: H_SNOW,
            TerrainType.MOUNTAIN: H_MOUNTAIN,
            TerrainType.HILL: H_HILL,
            TerrainType.SHALLOW_OCEAN: H_SHALLOW_OCEAN,
            TerrainType.DEEP_OCEAN: H_DEEP_OCEAN,
            TerrainType.VOID: H_VOID
        }

        noise_dim: V2[int] = V2(HEX_NOF_HEXES.x(), HEX_NOF_HEXES.y())

        def normal_noise(res: tuple[int, int]) -> NDArray: # type: ignore
            noise: NDArray[float, Any] = pnp.generate_perlin_noise_2d( # type: ignore
                shape=(noise_dim.x(), noise_dim.y()),
                res=res,
                tileable=(True, False)
            )  # [-1, 1]
            return normalise(noise)  # [0, 1]
        
        def ridge_noise(res: tuple[int, int]) -> NDArray: # type: ignore
            noise: NDArray[float, Any] = pnp.generate_perlin_noise_2d( # type: ignore
                shape=(noise_dim.x(), noise_dim.y()),
                res=res,
                tileable=(True, False)
            )  # [-1, 1]
            # make ridges
            noise = -1 * np.abs(noise)  # [-1, 0]
            noise += 0.5  # [-0.5, 0.5]
            noise *= 2  # [-1, 1]
            return normalise(noise)  # [0, 1]

        # ==== CONTINENTS ====
        continent0_noise: NDArray[float, Any] = normal_noise(CONTINENT_NOISE_FREQUENCY.get())
        continent1_noise: NDArray[float, Any] = normal_noise(CONTINENT_NOISE_FREQUENCY.get())

        # combine two noise patterns
        continent_noise = normalise(continent0_noise * continent1_noise)

        # "raise" continents (skew distribution higher)
        continent_noise **= 0.8  # TODO MAGIC VALUE

        # flatten peaks until max height is less than middle of H_HILL range
        mask: NDArray[bool, Any] = continent_noise > H_HILL[0]
        max_height: float = H_HILL[0] + ((H_HILL[1] - H_HILL[0]) / 2)
        f: Callable[[NDArray[float, Any]], NDArray[float, Any]] = lambda x: x - ((x - H_HILL[0]) * 0.5)
        while np.max(continent_noise) > max_height:
            continent_noise: NDArray[float, Any] = np.where(mask, f(continent_noise), continent_noise)

        # ==== TERRAIN ====
        terrain0_noise: NDArray[float, Any] = normal_noise(TERRAIN0_NOISE_FREQUENCY.get())
        terrain1_noise: NDArray[float, Any] = normal_noise(TERRAIN1_NOISE_FREQUENCY.get())
        terrain2_noise: NDArray[float, Any] = normal_noise(TERRAIN2_NOISE_FREQUENCY.get())

        # ==== HEIGHTMAP WITHOUT MOUNTAINS ====
        # combine weighted noise patterns (total weight == 1)
        heightmap: NDArray[float, Any] = \
            (CONTINENT_NOISE_WEIGHT * continent_noise) + \
            (TERRAIN0_NOISE_WEIGHT * terrain0_noise) + \
            (TERRAIN1_NOISE_WEIGHT * terrain1_noise) + \
            (TERRAIN2_NOISE_WEIGHT * terrain2_noise)
    
        # ==== MOUNTAIN RANGES ====
        mountain_range_noise: NDArray[float, Any] = ridge_noise(MOUNTAIN_RANGE_NOISE_FREQUENCY.get())

        # make ridges thicker
        mountain_range_noise **= 0.8  # [0, 1]  # TODO magic value

        mountain_range_noise *= ((heightmap + 0.35) ** 2.5)  # TODO magic value

        # ==== MOUNTAIN TERRAIN ====
        mountain0_noise: NDArray[float, Any] = ridge_noise(MOUNTAIN0_NOISE_FREQUENCY.get())
        mountain1_noise: NDArray[float, Any] = ridge_noise(MOUNTAIN1_NOISE_FREQUENCY.get())
        mountain2_noise: NDArray[float, Any] = ridge_noise(MOUNTAIN2_NOISE_FREQUENCY.get())

        # ==== MOUNTAINS ====
        # combine weighted noise patterns (total weight == 1)
        mountains: NDArray[float, Any] = \
            (MOUNTAIN0_NOISE_WEIGHT * mountain0_noise) + \
            (MOUNTAIN1_NOISE_WEIGHT * mountain1_noise) + \
            (MOUNTAIN2_NOISE_WEIGHT * mountain2_noise)
        
        # mask with mountain range pattern
        mountains *= mountain_range_noise
        mountains = normalise(mountains)

        # zero all values below 1/8th of deep ocean range below shallow ocean
        mask = mountains < H_DEEP_OCEAN[1] - ((H_DEEP_OCEAN[1] - H_DEEP_OCEAN[0]) / 8)
        mountains = np.where(mask, 0, mountains)

        # ==== HEIGHTMAP COMBINED ====
        # take the maximum height of heightmap or mountain_noise
        heightmap_combined: NDArray[float, Any] = np.maximum(heightmap, mountains)
        # heightmap_combined = mountains

        # ==== SET MAP AND GRADIENTS ====
        self._map: NDArray[float, Any] = heightmap_combined
        gradient: list[NDArray[float, Any]] = np.gradient(heightmap_combined)
        self._gradient_x: NDArray[float, Any] = gradient[0]
        self._gradient_y: NDArray[float, Any] = gradient[1]
        self._gradient_x_std: float = float(np.std(self._gradient_x))
        self._gradient_y_std: float = float(np.std(self._gradient_y))

    def get_height_from_of(self, of: V2[int]) -> float:
        return  self._map[of.x()][of.y()] if of.x() < len(self._map) and of.y() < len(self._map[0]) else 0.0
    
    def get_x_gradient_from_of(self, of: V2[int]) -> float:   
        return self._gradient_x[of.x()][of.y()] if of.x() < len(self._map) and of.y() < len(self._map[0]) else 0.0
    def get_y_gradient_from_of(self, of: V2[int]) -> float:   
        return self._gradient_y[of.x()][of.y()] if of.x() < len(self._map) and of.y() < len(self._map[0]) else 0.0
    def get_gradient_std(self) -> V2[float]:
        return V2(2.0 * self._gradient_x_std, 2.0 * self._gradient_y_std)
    
    def get_height(self, terrain: TerrainType) -> V2[float]:
        return self._mapping[terrain]

    def get_terrain_type(self, height: float) -> TerrainType:
        for terrain in self._mapping.keys():
            terrain_height: V2[float] = self._mapping[terrain]
            if height >= terrain_height[0] and height < terrain_height[1]:
                return terrain
        return TerrainType.VOID