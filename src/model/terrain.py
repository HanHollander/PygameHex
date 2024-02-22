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
        C_MOUNTAIN: tuple[pg.Color, pg.Color] = (clr("#62795f"), clr("#9ba89b"))
        C_HILL: tuple[pg.Color, pg.Color] = (clr("#293b21"), clr("#4a6843"))
        C_FOREST: tuple[pg.Color, pg.Color] = (clr("#17270f"), clr("#5e7a47"))
        C_BEACH: tuple[pg.Color, pg.Color] = (clr("#ebeec3"), clr("#ebeec3"))
        C_SHALLOW_OCEAN: tuple[pg.Color, pg.Color] = (clr("#2e446d"), clr("#4c728b"))
        C_DEEP_OCEAN: tuple[pg.Color, pg.Color] = (clr("#0f192c"), clr("#213150"))
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
        # H_SNOW = V2(0.95, 999.0)
        H_MOUNTAIN = V2(0.75, 999.0)
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
            TerrainType.MOUNTAIN: H_MOUNTAIN,
            TerrainType.HILL: H_HILL,
            TerrainType.SHALLOW_OCEAN: H_SHALLOW_OCEAN,
            TerrainType.DEEP_OCEAN: H_DEEP_OCEAN,
            TerrainType.VOID: H_VOID
        }

        noise_dim: V2[int] = V2(Cfg.HEX_NOF_HEXES.x(), Cfg.HEX_NOF_HEXES.y())

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
        
        def flatten(noise: Any, min_height: float, res: float) -> Any:
            return noise - ((noise - min_height) * res)

        # ==== CONTINENTS ==== #
        continent0_noise: NDArray[float, Any] = normal_noise(Cfg.CONTINENT_NOISE_FREQUENCY.get())
        continent1_noise: NDArray[float, Any] = normal_noise(Cfg.CONTINENT_NOISE_FREQUENCY.get())

        # combine two noise patterns
        continent_noise = normalise(continent0_noise * continent1_noise)

        # "raise" continents (skew distribution higher)
        continent_noise **= Cfg.CONTINENT_NOISE_SIZE_MODIF

        # ==== TERRAIN ==== #
        terrain0_noise: NDArray[float, Any] = normal_noise(Cfg.TERRAIN0_NOISE_FREQUENCY.get())
        terrain1_noise: NDArray[float, Any] = normal_noise(Cfg.TERRAIN1_NOISE_FREQUENCY.get())
        terrain2_noise: NDArray[float, Any] = normal_noise(Cfg.TERRAIN2_NOISE_FREQUENCY.get())

        # ==== CONTINENTS ==== #
        # combine weighted noise patterns (total weight == 1)
        continents: NDArray[float, Any] = \
            (Cfg.CONTINENT_NOISE_WEIGHT * continent_noise) + \
            (Cfg.TERRAIN0_NOISE_WEIGHT * terrain0_noise) + \
            (Cfg.TERRAIN1_NOISE_WEIGHT * terrain1_noise) + \
            (Cfg.TERRAIN2_NOISE_WEIGHT * terrain2_noise)

        # flatten peaks until max height is less than certain value
        mask: NDArray[bool, Any] = continents > H_HILL[0]
        max_height: float = H_HILL[1] - ((H_HILL[1] - H_HILL[0]) / Cfg.CONTINENT_MAX_HEIGHT_DIV)
        while np.max(continents) > max_height:
            continents: NDArray[float, Any] = np.where(mask, flatten(continents, H_HILL[0], Cfg.CONTINENT_NOISE_PEAK_FLATTENING_RESOLUTION), continents)
    
        # ==== MOUNTAIN RANGES ==== #
        mountain_range_noise: NDArray[float, Any] = ridge_noise(Cfg.MOUNTAIN_RANGE_NOISE_FREQUENCY.get())

        # make ridges thicker
        mountain_range_noise **= Cfg.MOUNTAIN_RANGE_NOISE_WIDTH_MODIF  # [0, 1] 

        # mask with heightmap
        mountain_range_noise *= ((continents + Cfg.MOUNTAIN_MASK_SIZE_MODIF) ** Cfg.MOUNTAIN_MASK_STRENGTH_MODIF)  # [0, ??]

        # ==== MOUNTAIN TERRAIN ==== #
        mountain0_noise: NDArray[float, Any] = ridge_noise(Cfg.MOUNTAIN0_NOISE_FREQUENCY.get())
        mountain1_noise: NDArray[float, Any] = ridge_noise(Cfg.MOUNTAIN1_NOISE_FREQUENCY.get())
        mountain2_noise: NDArray[float, Any] = ridge_noise(Cfg.MOUNTAIN2_NOISE_FREQUENCY.get())

        # ==== MOUNTAINS ==== #
        # combine weighted noise patterns (total weight == 1)
        mountains: NDArray[float, Any] = \
            (Cfg.MOUNTAIN0_NOISE_WEIGHT * mountain0_noise) + \
            (Cfg.MOUNTAIN1_NOISE_WEIGHT * mountain1_noise) + \
            (Cfg.MOUNTAIN2_NOISE_WEIGHT * mountain2_noise)
        
        # mask with mountain range pattern
        mountains *= mountain_range_noise
        mountains = normalise(mountains)

        # ==== HILLS RANGES ==== #
        # make hill ranges with the same resolution as continents
        hill_range_noise: NDArray[float, Any] = normal_noise(Cfg.CONTINENT_NOISE_FREQUENCY.get())

        # increase or decrease width ("amount") of hills
        hill_range_noise **= Cfg.HILL_RANGE_NOISE_WIDTH_MODIF

        # mask with heightmap
        hill_range_noise *= ((continents + Cfg.HILL_MASK_SIZE_MODIF) ** Cfg.HILL_MASK_STRENGTH_MODIF)  # [0, ??]

        # ==== HILLS TERRAIN ==== #
        hill0_noise: NDArray[float, Any] = normal_noise(Cfg.HILL0_NOISE_FREQUENCY.get())
        hill1_noise: NDArray[float, Any] = normal_noise(Cfg.HILL1_NOISE_FREQUENCY.get())
        hill2_noise: NDArray[float, Any] = normal_noise(Cfg.HILL2_NOISE_FREQUENCY.get())

        # ==== HILLS ==== #
        # combine weighted noise patterns (total weight == 1)
        hills: NDArray[float, Any] = \
            (Cfg.HILL0_NOISE_WEIGHT * hill0_noise) + \
            (Cfg.HILL1_NOISE_WEIGHT * hill1_noise) + \
            (Cfg.HILL2_NOISE_WEIGHT * hill2_noise)
        
        # mask with hills range pattern
        hills *= hill_range_noise
        hills = normalise(hills)

        # flatten peaks until max height is less than certain value
        mask: NDArray[bool, Any] = hills > H_HILL[0]
        max_height: float = H_HILL[1] - ((H_HILL[1] - H_HILL[0]) / Cfg.HILL_MAX_HEIGHT_DIV)
        while np.max(hills) > max_height:
            hills: NDArray[float, Any] = np.where(mask, flatten(hills, H_HILL[0], Cfg.HILL_NOISE_PEAK_FLATTENING_RESOLUTION), hills)

        # ==== HEIGHTMAP COMBINED ==== #
        # take the maximum height of heightmap or mountain_noise
        heightmap: NDArray[float, Any]
        match Cfg.TERRAIN_LAYER_SHOWN:
            case TerrainLayerShown.CONTINENTS:
                heightmap = continents
            case TerrainLayerShown.MOUNTAINS:
                heightmap = mountains
            case TerrainLayerShown.HILLS:
                heightmap = hills
            case TerrainLayerShown.HEIGHTMAP:
                heightmap =  np.maximum(np.maximum(continents, mountains), hills)
            case TerrainLayerShown.CONTINENTS_MOUNTAINS:
                heightmap =  np.maximum(continents, mountains)
            case TerrainLayerShown.CONTINENTS_HILLS:
                heightmap =  np.maximum(continents, hills)
            case TerrainLayerShown.MOUNTAINS_HILLS:
                heightmap =  np.maximum(mountains, hills)
           
        # ==== SET MAP AND GRADIENTS ==== #
        self._map: NDArray[float, Any] = heightmap
        gradient: list[NDArray[float, Any]] = np.gradient(heightmap)
        self._gradient_x: NDArray[float, Any] = gradient[0]
        self._gradient_y: NDArray[float, Any] = gradient[1]
        self._gradient_x_std: float = float(np.std(self._gradient_x))
        self._gradient_y_std: float = float(np.std(self._gradient_y))

    def get_height_from_of(self, of: V2[int]) -> float:
        return  self._map[of.x()][of.y()]
    
    def get_x_gradient_from_of(self, of: V2[int]) -> float:   
        return self._gradient_x[of.x()][of.y()]
    def get_y_gradient_from_of(self, of: V2[int]) -> float:   
        return self._gradient_y[of.x()][of.y()]
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