from enum import Enum
from math import ceil, sqrt
from typing import Any
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
        C_MOUNTAIN: tuple[pg.Color, pg.Color] = (clr("#7f867e"), clr("#d8d8d8"))
        C_HILL: tuple[pg.Color, pg.Color] = (clr("#293b21"), clr("#718873"))
        C_FOREST: tuple[pg.Color, pg.Color] = (clr("#17270f"), clr("#5e7a47"))
        C_BEACH: tuple[pg.Color, pg.Color] = (clr("#ebeec3"), clr("#ebeec3"))
        C_SHALLOW_OCEAN: tuple[pg.Color, pg.Color] = (clr("#6281b9"), clr("#9ac2dd"))
        C_DEEP_OCEAN: tuple[pg.Color, pg.Color] = (clr("#435f92"), clr("#5471a7"))
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
        H_HILL = V2(0.61, H_MOUNTAIN[0])
        H_SHALLOW_OCEAN = V2(0.55, H_HILL[0])
        H_DEEP_OCEAN = V2(0.0, H_SHALLOW_OCEAN[0])
        H_VOID = V2(-999.0, H_DEEP_OCEAN[0])

        self._mapping: dict[TerrainType, V2[float]] = {
            TerrainType.SNOW: H_SNOW,
            TerrainType.MOUNTAIN: H_MOUNTAIN,
            TerrainType.HILL: H_HILL,
            TerrainType.SHALLOW_OCEAN: H_SHALLOW_OCEAN,
            TerrainType.DEEP_OCEAN: H_DEEP_OCEAN,
            TerrainType.VOID: H_VOID
        }

        np.random.seed()

        noise_dim: V2[int] = V2(HEX_NOF_HEXES.x(), HEX_NOF_HEXES.y())

        continent_noise: NDArray[float, Any] = pnp.generate_perlin_noise_2d(
            shape=(noise_dim.x(), noise_dim.y()),
            res=CONTINENT_NOISE_FREQUENCY.get(),
            tileable=(True, False)
        )
        normalise(continent_noise)

        mountain0_noise: NDArray[float, Any] = pnp.generate_perlin_noise_2d(
            shape=(noise_dim.x(), noise_dim.y()),
            res=MOUNTAIN0_NOISE_FREQUENCY.get(),
            tileable=(True, False)
        )
        mountain1_noise: NDArray[float, Any] = pnp.generate_perlin_noise_2d(
            shape=(noise_dim.x(), noise_dim.y()),
            res=MOUNTAIN1_NOISE_FREQUENCY.get(),
            tileable=(True, False)
        )

        mountain_noise = (0.2 * mountain0_noise ) + (0.8 * mountain1_noise)
        mountain_noise = -1 * np.abs(mountain_noise)
        mountain_noise += 0.5
        mountain_noise *= 2


        terrain_noise: NDArray[float, Any] = pnp.generate_perlin_noise_2d(
            shape=(noise_dim.x(), noise_dim.y()),
            res=TERRAIN0_NOISE_FREQUENCY.get(),
            tileable=(True, False)
        )
        terrain1_noise: NDArray[float, Any] = pnp.generate_perlin_noise_2d(
            shape=(noise_dim.x(), noise_dim.y()),
            res=TERRAIN1_NOISE_FREQUENCY.get(),
            tileable=(True, False)
        )
        terrain2_noise: NDArray[float, Any] = pnp.generate_perlin_noise_2d(
            shape=(noise_dim.x(), noise_dim.y()),
            res=TERRAIN2_NOISE_FREQUENCY.get(),
            tileable=(True, False)
        )

        heightmap = (CONTINENT_NOISE_WEIGHT * continent_noise) \
                + (MOUNTAIN_NOISE_WEIGHT * mountain_noise) \
                    + (TERRAIN0_NOISE_WEIGHT * terrain_noise) \
                    + (TERRAIN1_NOISE_WEIGHT * terrain1_noise) \
                    + (TERRAIN2_NOISE_WEIGHT * terrain2_noise)
        



        normalise(heightmap)
        self._map: list[list[float]] =  heightmap.tolist()

    def get_height_from_of(self, of: V2[int]) -> float:
        return  self._map[of.x()][of.y()] if of.x() < len(self._map) and of.y() < len(self._map[0]) else 0.0
    
    def get_height(self, terrain: TerrainType) -> V2[float]:
        return self._mapping[terrain]

    def get_terrain_type(self, height: float) -> TerrainType:
        for terrain in self._mapping.keys():
            terrain_height: V2[float] = self._mapping[terrain]
            if height >= terrain_height[0] and height < terrain_height[1]:
                return terrain
        return TerrainType.VOID