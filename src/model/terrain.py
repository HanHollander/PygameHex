from enum import Enum
from typing import Any
from nptyping import NDArray
import numpy as np
import perlin_numpy as pnp
from config import *
from util import V2, V3

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
        C_SNOW = V3(0xff, 0xfa, 0xfa)
        C_MOUNTAIN = V3(0x6e, 0x70, 0x6e)
        C_HILL = V3(0x78, 0x91, 0x44)
        C_FOREST = V3(0x23, 0x4a, 0x10)
        C_BEACH = V3(0xff, 0xfc, 0xde)
        C_SHALLOW_OCEAN = V3(0x83, 0xb5, 0xd6)
        C_DEEP_OCEAN = V3(0x1e, 0x53, 0x75)
        C_VOID = V3(0x00, 0x00, 0x00)

        self._mapping: dict[TerrainType, V3[int]] = {
            TerrainType.SNOW: C_SNOW,
            TerrainType.MOUNTAIN: C_MOUNTAIN,
            TerrainType.HILL: C_HILL,
            TerrainType.FOREST: C_FOREST,
            TerrainType.BEACH: C_BEACH,
            TerrainType.SHALLOW_OCEAN: C_SHALLOW_OCEAN,
            TerrainType.DEEP_OCEAN: C_DEEP_OCEAN,
            TerrainType.VOID: C_VOID
        }

    def get_colour(self, type: TerrainType) -> V3[int]:
        return self._mapping[type]
    


class TerrainHeightmap:
    
    def __init__(self) -> None:
        H_SNOW: float = 0.95
        H_MOUNTAIN: float = 0.89
        H_HILL: float = 0.79
        H_FOREST: float = 0.61
        H_BEACH: float = 0.608
        H_SHALLOW_OCEAN: float = 0.55
        H_DEEP_OCEAN: float = 0.0
        H_VOID: float = -999.0


        # H_SNOW: float = 0.00 #0.95
        # H_MOUNTAIN: float = -0.01 # 0.81
        # H_HILL: float = -0.01 # 0.69
        # H_FOREST: float = -0.01 # 0.61
        # H_BEACH: float = -0.01 # 0.608
        # H_SHALLOW_OCEAN: float = -0.01 # 0.55
        # H_DEEP_OCEAN: float = -0.01 # 0.0
        # H_VOID: float = -999.0

        self._mapping: dict[TerrainType, float] = {
            TerrainType.SNOW: H_SNOW,
            TerrainType.MOUNTAIN: H_MOUNTAIN,
            TerrainType.HILL: H_HILL,
            TerrainType.FOREST: H_FOREST,
            TerrainType.BEACH: H_BEACH,
            TerrainType.SHALLOW_OCEAN: H_SHALLOW_OCEAN,
            TerrainType.DEEP_OCEAN: H_DEEP_OCEAN,
            TerrainType.VOID: H_VOID
        }

        np.random.seed()

        continent_noise: NDArray[float, Any] = pnp.generate_perlin_noise_2d(
            shape=(HEX_NOF_HEXES.x(), HEX_NOF_HEXES.y()),
            res=CONTINENT_NOISE_FREQUENCY.get(),
            tileable=(True, False)
        )
        feature_noise: NDArray[float, Any] = pnp.generate_perlin_noise_2d(
            shape=(HEX_NOF_HEXES.x(), HEX_NOF_HEXES.y()),
            res=FEATURE_NOISE_FREQUENCY.get(),
            tileable=(True, False)
        )
        mountain_noise: NDArray[float, Any] = pnp.generate_perlin_noise_2d(
            shape=(HEX_NOF_HEXES.x(), HEX_NOF_HEXES.y()),
            res=MOUNTAIN_NOISE_FREQUENCY.get(),
            tileable=(True, False)
        )
        mountain_noise = -1 * np.abs(mountain_noise)
        mountain_noise += 0.5
        mountain_noise *= 2
        terrain_noise: NDArray[float, Any] = pnp.generate_perlin_noise_2d(
            shape=(HEX_NOF_HEXES.x(), HEX_NOF_HEXES.y()),
            res=TERRAIN_NOISE_FREQUENCY.get(),
            tileable=(True, False)
        )
        terrain1_noise: NDArray[float, Any] = pnp.generate_perlin_noise_2d(
            shape=(HEX_NOF_HEXES.x(), HEX_NOF_HEXES.y()),
            res=TERRAIN1_NOISE_FREQUENCY.get(),
            tileable=(True, False)
        )
        terrain2_noise: NDArray[float, Any] = pnp.generate_perlin_noise_2d(
            shape=(HEX_NOF_HEXES.x(), HEX_NOF_HEXES.y()),
            res=TERRAIN2_NOISE_FREQUENCY.get(),
            tileable=(True, False)
        )
        x_axis = np.linspace(0, 1, min(HEX_NOF_HEXES.x(), HEX_NOF_HEXES.y()))[:, None]
        y_axis = np.linspace(0, 1, min(HEX_NOF_HEXES.x(), HEX_NOF_HEXES.y()))[None, :]
        # edge_gradient = np.sqrt(x_axis ** 2 + y_axis ** 2)
        # edge_gradient = np.interp(edge_gradient, np.linspace(0, HEX_NOF_HEXES.x() - 1, HEX_NOF_HEXES.x()), np.linspace(0, HEX_NOF_HEXES.y() - 1, HEX_NOF_HEXES.y()))


        heightmap = (CONTINENT_NOISE_WEIGHT * continent_noise) \
            + (FEATURE_NOISE_WEIGHT * feature_noise) \
                + (MOUNTAIN_NOISE_WEIGHT * mountain_noise) \
                    + (TERRAIN_NOISE_WEIGHT * terrain_noise) \
                    + (TERRAIN1_NOISE_WEIGHT * terrain1_noise) \
                    + (TERRAIN2_NOISE_WEIGHT * terrain2_noise)
        # heightmap *= edge_gradi/ent

        # "Peretoish?" https://www.reddit.com/r/proceduralgeneration/comments/13qzykm/q_quick_question_about_smoothing_perlin_noise_by/ 
        # heightmap = -1 * np.abs(heightmap)
        # heightmap = heightmap + 0.5
        # heightmap = heightmap * 2
        # heightmap = heightmap ** 1.4



        # normalise from [-1, 1] to [0, 1]
        heightmap += np.abs(np.min(heightmap))
        heightmap /= np.max(heightmap)
        self._map: list[list[float]] =  heightmap.tolist()

    def get_height(self, of: V2[int]) -> float:
        return self._map[of.x()][of.y()]

    def get_terrain_type(self, height: float) -> TerrainType:
        result: TerrainType = TerrainType.VOID
        max_height: float = self._mapping[result]
        for terrain in self._mapping.keys():
            terrain_height: float = self._mapping[terrain]
            if height >= terrain_height and terrain_height > max_height:
                result = terrain
                max_height = terrain_height
        return result