import colorsys
import ctypes
from threading import Thread
import time
from bisect import bisect
from typing import Any, Callable, TYPE_CHECKING, Generic, TypeVar
from model.terrain import TerrainAltitude, TerrainAltitudeMapping, TerrainHeightmap, TerrainColourMapping, TerrainHumidity, TerrainHumidityMapping, TerrainHumiditymap, TerrainMapping, TerrainTemperature, TerrainTemperatureMapping, TerrainTemperaturemap, TerrainType
import pygame as pg
import pynkie as pk
import numpy as np

from math import ceil, floor, isclose, sqrt, cos, sin, pi
from util import V2, V3, bilinear_interpolate_hsv, bilinear_interpolate_v3, clip, even, get_v3_from_colour, interpolate_hsv, interpolate_v3
from config import cfg, ColourScheme, HexOrientation
from view.loading import display_message

if TYPE_CHECKING:
    from view.hex import HexView


# https://www.redblobgames.com/grids/hexagons/

class Ax:

    # static class variables
    ax_to_px_flat: list[list[float]] = [[3./2, 0], [sqrt(3)/2, sqrt(3)]]
    px_to_ax_flat: list[list[float]] = [[2./3, 0], [-1./3, sqrt(3)/3]]
    ax_to_px_pointy: list[list[float]] = [[sqrt(3), sqrt(3)/2], [0, 3./2]]
    px_to_ax_pointy: list[list[float]] = [[sqrt(3)/3, -1./3], [0, 2./3]]
    
    @staticmethod
    def ax_round(c_frac: V2[float]) -> V2[int]:
        q: int = round(c_frac.q())
        r: int = round(c_frac.r())
        s: int = round(-c_frac.q() - c_frac.r())

        q_diff: float = abs(q - c_frac.q())
        r_diff: float = abs(r - c_frac.r())
        s_diff: float = abs(s - (-c_frac.q() - c_frac.r()))

        if q_diff > r_diff and q_diff > s_diff:
            q = -r-s
        elif r_diff > s_diff:
            r = -q-s

        return V2(q, r)

    @staticmethod
    def ax_to_px(ax: "Ax") -> V2[int]:
        match Hex.orientation:
            case HexOrientation.FLAT:
                px: V2[Any] = V2(*((Hex.size * np.array(Ax.ax_to_px_flat).dot(ax.get().get())).tolist()))
            case HexOrientation.POINTY:
                px: V2[Any] = V2(*((Hex.size * np.array(Ax.ax_to_px_pointy).dot(ax.get().get())).tolist()))
        return V2(round(px.x()), round(px.y()))
    
    @staticmethod
    def ax_to_px_offset(ax: "Ax", offset: V2[int]) -> V2[int]:
        match Hex.orientation:
            case HexOrientation.FLAT:
                px: V2[Any] = V2(*((Hex.size * np.array(Ax.ax_to_px_flat).dot(ax.get().get())).tolist()))
            case HexOrientation.POINTY:
                px: V2[Any] = V2(*((Hex.size * np.array(Ax.ax_to_px_pointy).dot(ax.get().get())).tolist()))
        return V2(round(px.x()), round(px.y())) + offset
            
    @staticmethod
    def px_to_ax(px: V2[int]) -> "Ax":
        match Hex.orientation:
            case HexOrientation.FLAT:
                return Ax(Ax.ax_round(
                    V2(*(np.array(Ax.px_to_ax_flat).dot(px.get()) / Hex.size).tolist())))
            case HexOrientation.POINTY:
                return Ax(Ax.ax_round(
                     V2(*(np.array(Ax.px_to_ax_pointy).dot(px.get()) / Hex.size).tolist())))
            
    @staticmethod
    def px_to_ax_offset(px: V2[int], offset: V2[int]) -> "Ax":
        px_offset: V2[int] = px + offset
        match Hex.orientation:
            case HexOrientation.FLAT:
                return Ax(Ax.ax_round(
                     V2(*(np.array(Ax.px_to_ax_flat).dot(px_offset.get()) / Hex.size).tolist())))
            case HexOrientation.POINTY:
                return Ax(Ax.ax_round(
                     V2(*(np.array(Ax.px_to_ax_pointy).dot(px_offset.get()) / Hex.size).tolist())))
            
    @staticmethod
    def ax_to_of(ax: "Ax") -> V2[int]:  # odd-r
        # direct access ax.c for speed
        q: int = ax.c[0]
        r: int = ax.c[1]
        match Hex.orientation:
            case HexOrientation.FLAT:
                return V2(q, r + round((q - (q&1)) / 2))
            case HexOrientation.POINTY:
                return V2(q + round((r - (r&1)) / 2), r)
            
    @staticmethod
    def of_to_ax(of: V2[int]) -> "Ax":  # odd-q
        match Hex.orientation:
            case HexOrientation.FLAT:
                return Ax(V2(of.x(), round(of.y() - (of.x() - (of.x()&1)) / 2)))
            case HexOrientation.POINTY:
                return Ax(V2(round(of.x() - (of.y() - (of.y()&1)) / 2), of.y()))

    def __init__(self, c: V2[int]) -> None:
        self.c: V2[int] = c  # axial coords [q, r]; q + r + s = 0

    def __str__(self) -> str:
        return "<q = " + str(self.q()) + ", r = " + str(self.r()) + ">"

    def q(self) -> int:
        return self.c.q()
    
    def r(self) -> int:
        return self.c.r()
    
    def s(self) -> int:
        return -self.q() - self.r()
    
    def get(self) -> V2[int]:
        return self.c
    
    def get_cb(self) -> V3[int]:
        return V3(self.q(), self.r(), self.s())


class HexSpriteElement(pk.elements.Element, pg.sprite.Sprite):

    def __init__(self, pos: tuple[int, int], img: pg.Surface, colour: V3[int]) -> None:
        pk.elements.Element.__init__(self, pos, img.get_size())
        pg.sprite.Sprite.__init__(self)
        self.colour: V3[int] = colour
        self.image: pg.Surface = img

class HexSpriteStore:

    # static class variables
    _store: dict[TerrainType, pg.Surface]  # should not access, use scaled_store instead
    scaled_store: dict[TerrainType, pg.Surface]

    @staticmethod
    def draw_hex(surface: pg.Surface, color: list[int], radius: int, position: V2[int], width: int=0) -> None:
        r: int = radius
        x: int = position.x()
        y: int = position.y()
        match Hex.orientation:
            case HexOrientation.FLAT:
                pg.draw.polygon(surface, color, [(x + r * cos(2 * pi * i / 6), y + r * sin(2 * pi * i / 6)) for i in range(6)], width)
            case HexOrientation.POINTY:
                pg.draw.polygon(surface, color, [(x + r * cos(2 * pi * i / 6 + 2 * pi / 12), y + r * sin(2 * pi * i / 6 + 2 * pi / 12)) for i in range(6)], width)

    @staticmethod
    def make_surface_from_of(of: V2[int], dim: V2[int], draw_border: bool = False, draw_center: bool = False) -> pg.Surface:
        ax: Ax = Ax.of_to_ax(of)
        rgb: list[int] = [abs((63 * ax.q() + 82) % 255), 
                          abs((187 * ax.r() + 43) % 255), 
                          abs((229 * ax.s() + 52) % 255)]
        rgb_inv: list[int] = [255 - rgb[0], 255 - rgb[1], 255 - rgb[2]]
        surface = pg.Surface((dim.x(), dim.y()), pg.SRCALPHA)
        HexSpriteStore.draw_hex(surface, rgb, Hex.size, V2(round(dim.x() / 2), round(dim.y() / 2)), 0)
        if draw_border:
            HexSpriteStore.draw_hex(surface, rgb_inv, Hex.size, V2(round(dim.x() / 2), round(dim.y() / 2)), 5)
        if draw_center:
            pg.draw.circle(surface, rgb_inv, [dim.x() / 2, dim.y() / 2], 20)
        return surface
    
    @staticmethod
    def make_surface_from_colour(colour: V3[int], dim: V2[int], draw_border: bool = False, draw_center: bool = False) -> pg.Surface:
        rgb: list[int] = [*colour.get()]
        rgb_inv: list[int] = [255 - rgb[0], 255 - rgb[1], 255 - rgb[2]]
        surface = pg.Surface((dim.x(), dim.y()), pg.SRCALPHA)
        HexSpriteStore.draw_hex(surface, rgb, Hex.size, V2(round(dim.x() / 2), round(dim.y() / 2)), 0)
        if draw_border:
            HexSpriteStore.draw_hex(surface, rgb_inv, Hex.size, V2(round(dim.x() / 2), round(dim.y() / 2)), 5)
        if draw_center:
            pg.draw.circle(surface, rgb_inv, [dim.x() / 2, dim.y() / 2], 20)
        return surface

    @staticmethod
    def init_store() -> None:
        dim: V2[int] = Hex.calc_dim_from_size(cfg.HEX_INIT_SPRITE_SIZE)[0]
        HexSpriteStore._store = {}
        HexSpriteStore.scaled_store = {}
        for terrain in TerrainType:
            HexSpriteStore._store[terrain] = HexSpriteStore.make_surface_from_colour(V3(0xff, 0xff, 0xff), dim)
            HexSpriteStore.scaled_store[terrain] = HexSpriteStore._store[terrain].copy()

    @staticmethod
    def reset_scaled_store() -> None:
        for terrain in TerrainType:
            HexSpriteStore.scaled_store[terrain] =  pg.transform.scale(HexSpriteStore._store[terrain], (Hex.dim[0], Hex.dim[1]))


T_ENUM = TypeVar("T_ENUM")
class HexAttr():
    
    class GradientBound(Generic[T_ENUM]):
        def __init__(self, b: float, t: T_ENUM) -> None:
            self.b: float = b
            self.t: T_ENUM = t
        
    class GradientBoundList(Generic[T_ENUM]):
        def __init__(self) -> None:
            self.list: list[HexAttr.GradientBound[T_ENUM]] = []
            self.min0: float = 0.0
            self.min1: float = 0.0
            self.max: float = 1.0
        def __getitem__(self, i: int) -> float:
            return self.list[i].b
        def at(self, i: int) -> "HexAttr.GradientBound[T_ENUM]":
            return self.list[i]
        def append(self, v: "HexAttr.GradientBound[T_ENUM]") -> None:
            self.list.append(v)
        def __len__(self) -> int:
            return len(self.list)
        def set_min_max(self) -> None:
            self.min0 = self[0]
            self.min1 = self[1]
            self.max = self[len(self) - 1]
        def find_bounds(self, v: float) -> tuple["HexAttr.GradientBound[T_ENUM]", "HexAttr.GradientBound[T_ENUM]"]:
            i: int = bisect(self, v)
            lower: HexAttr.GradientBound[T_ENUM] = self.at(0) if i == 0 else self.at(i - 1)
            higher: HexAttr.GradientBound[T_ENUM] = self.at(len(self) - 1) if i == len(self) else self.at(i)
            return (lower, higher)

    # static class variables
    terrain_mapping: TerrainMapping
    altitude_mapping: TerrainAltitudeMapping
    humidity_mapping: TerrainHumidityMapping
    temperature_mapping: TerrainTemperatureMapping
    colour_mapping: TerrainColourMapping
    heightmap: TerrainHeightmap
    humiditymap: TerrainHumiditymap
    temperaturemap: TerrainTemperaturemap
    gradient_bounds_h: GradientBoundList[TerrainHumidity]
    gradient_bounds_t: GradientBoundList[TerrainTemperature]
    seed: int = 0

    @staticmethod
    def init(reseed: bool = False) -> None:
        if reseed:
            HexAttr.seed = int(np.random.rand() * (2**32 - 1))
        np.random.seed(HexAttr.seed)
        print("seed: " + str(HexAttr.seed))

        HexAttr.terrain_mapping = TerrainMapping()
        HexAttr.altitude_mapping = TerrainAltitudeMapping()
        HexAttr.humidity_mapping = TerrainHumidityMapping()
        HexAttr.temperature_mapping = TerrainTemperatureMapping()
        HexAttr.colour_mapping = TerrainColourMapping()
        HexAttr.heightmap = TerrainHeightmap()
        HexAttr.humiditymap = TerrainHumiditymap(HexAttr.heightmap.continents.copy(), HexAttr.heightmap.unmod_mountain_range_noise.copy(), HexAttr.heightmap.heightmap.copy())
        HexAttr.temperaturemap = TerrainTemperaturemap(HexAttr.heightmap.heightmap.copy())

        HexAttr.gradient_bounds_h = HexAttr.GradientBoundList()
        for terrain_humidity_idx in TerrainHumidity:
            h0, h1 = HexAttr.humidity_mapping.get_humidity(terrain_humidity_idx).get()
            bound: float = h0 + ((h1 - h0) / 2)
            HexAttr.gradient_bounds_h.append(HexAttr.GradientBound(bound, terrain_humidity_idx))
        HexAttr.gradient_bounds_h.set_min_max()
        HexAttr.gradient_bounds_t = HexAttr.GradientBoundList()
        for terrain_temperature_idx in TerrainTemperature:
            t0, t1 = HexAttr.temperature_mapping.get_temperature(terrain_temperature_idx).get()
            bound: float = t0 + ((t1 - t0) / 2)
            HexAttr.gradient_bounds_t.append(HexAttr.GradientBound(bound, terrain_temperature_idx))
        HexAttr.gradient_bounds_t.set_min_max()

    def __init__(self, hex: "Hex") -> None:
        of: V2[int] = Ax.ax_to_of(hex.ax())
        self.altitude: float = HexAttr.heightmap.get_altitude_from_of(of)
        self.terrain_altitude: TerrainAltitude = HexAttr.altitude_mapping.get_terrain_altitude(self.altitude)
        self.humidity: float = HexAttr.humiditymap.get_humidity_from_of(of)
        self.terrain_humidity: TerrainHumidity = HexAttr.humidity_mapping.get_terrain_humidity(self.humidity)
        self.temperature: float = HexAttr.temperaturemap.get_temperature_from_of(of)
        self.terrain_temperature: TerrainTemperature = HexAttr.temperature_mapping.get_terrain_temperature(self.temperature)
        self.terrain_type: TerrainType = HexAttr.terrain_mapping.get_terrain_type(self.terrain_altitude, self.terrain_humidity, self.terrain_temperature)

    
class Hex(pk.model.Model):
    
    # static class variables
    size: int  # "radius" of the hex (center to vertex)
    dim: V2[int]  # width and height of the hex
    dim_float: V2[float]
    orientation: HexOrientation  # pointy or flat top

    n_created: int = 0

    @staticmethod
    def calc_dim() -> tuple[V2[int], V2[float]]:
        match Hex.orientation:
            case HexOrientation.FLAT:
                return (V2(Hex.size * 2, even(round(sqrt(3) * Hex.size))), V2(Hex.size * 2, sqrt(3) * Hex.size))
            case HexOrientation.POINTY:
                return (V2(even(round(sqrt(3) * Hex.size)), Hex.size * 2), V2(sqrt(3) * Hex.size, Hex.size * 2))
        
    @staticmethod
    def calc_dim_from_size(size: int) -> tuple[V2[int], V2[float]]:
        match Hex.orientation:
            case HexOrientation.FLAT:
                return (V2(size * 2, even(round(sqrt(3) * size))), V2(size * 2, sqrt(3) * size))
            case HexOrientation.POINTY:
                return (V2(even(round(sqrt(3) * size)), size * 2), V2(sqrt(3) * size, size * 2))
            
    @staticmethod
    def set_size(size: int) -> None:
        Hex.size = size
        Hex.dim, Hex.dim_float = Hex.calc_dim()
        Hex.size_updated = True

    def __init__(self, q: int, r: int) -> None:
        self._ax: Ax = Ax(V2(q, r))
        self._attr: HexAttr = HexAttr(self)

        px: V2[int] = Ax.ax_to_px(self._ax)
        pos: tuple[int, int] = (px.x() - round(Hex.dim.x() / 2), px.y() - round(Hex.dim.y() / 2));
        img: pg.Surface = HexSpriteStore.scaled_store[self._attr.terrain_type]
        colour: V3[int] = self.determine_colour()
        self._element: HexSpriteElement = HexSpriteElement(pos, img, colour)

        Hex.n_created += 1  # keep track of how many hexes have been created, and report to main thread
        inc: int = (cfg.HEX_NOF_HEXES.x() * cfg.HEX_NOF_HEXES.y()) // 100
        if not (Hex.n_created % inc) :
            time.sleep(0.00000001)  # microsleep to yield GIL to main thread

    def ax(self) -> Ax:
        return self._ax
    
    def attr(self) -> HexAttr:
        return self._attr
    def set_attr(self, attr: HexAttr) -> None:
        self._attr = attr
    
    def element(self) -> HexSpriteElement:
        return self._element
    def set_colour(self, colour: V3[int]) -> None:
        self._element.colour = colour
    
    def get_shading_mult(self) -> float:
        shading_mult = 1.0
        of: V2[int] = Ax.ax_to_of(self._ax)
        
        gradient_2std: V2[float] = HexAttr.heightmap.get_gradient_std()
        gradient_x: float = clip(HexAttr.heightmap.get_x_gradient_from_of(of), -gradient_2std.x(), gradient_2std.x())
        gradient_y: float = clip(HexAttr.heightmap.get_y_gradient_from_of(of), -gradient_2std.y(), gradient_2std.y())
        
        mult_x: float = 1.0 - (gradient_x / gradient_2std.x() * (1.0 - cfg.SHADING_MULT))
        shading_mult *= 1 / mult_x
        mult_y: float = 1.0 - (gradient_y / gradient_2std.y() * (1.0 - cfg.SHADING_MULT))
        shading_mult *= 1 / mult_y

        if self._attr.terrain_type == TerrainType.DEEP_WATER:
            shading_mult **= cfg.SEA_SHADING_MULT_MODIF
            
        return shading_mult
    
    def get_terrain_colour(self, a: float, h: float, t: float) -> V3[int]:
        terrain_altitude: TerrainAltitude = HexAttr.altitude_mapping.get_terrain_altitude(a)
        terrain_humidity: TerrainHumidity = HexAttr.humidity_mapping.get_terrain_humidity(h)
        terrain_temperature: TerrainTemperature = HexAttr.temperature_mapping.get_terrain_temperature(t)
        tt: TerrainType = HexAttr.terrain_mapping.get_terrain_type(terrain_altitude, terrain_humidity, terrain_temperature)

        # special case: deep water
        if tt == TerrainType.DEEP_WATER:
            return get_v3_from_colour(HexAttr.colour_mapping.get_colour(tt))
        
        # special case: freezing
        if t < HexAttr.temperature_mapping.get_temperature(TerrainTemperature.COLD)[0]:
            return get_v3_from_colour(cfg.C_ARCTIC)
        
        # special case: shallow water
        if tt == TerrainType.SHALLOW_WATER:
            return get_v3_from_colour(HexAttr.colour_mapping.get_colour(tt))
        
        t_min: float = HexAttr.gradient_bounds_t.min1
        t_max: float = HexAttr.gradient_bounds_t.max
        h_min: float = HexAttr.gradient_bounds_h.min0
        h_max: float = HexAttr.gradient_bounds_h.max

        if (t < t_min or t > t_max) and (h < h_min or h > h_max):
            # corner, only one choice
            return get_v3_from_colour(HexAttr.colour_mapping.get_colour(tt))
        elif (t >= t_min or t <= t_max) and (h < h_min or h > h_max):
            # top/bottom edge, temperature gradient
            t0, t1 = HexAttr.gradient_bounds_t.find_bounds(t)

            tt0: TerrainType = HexAttr.terrain_mapping.mapping[terrain_humidity][t0.t]
            tt1: TerrainType = HexAttr.terrain_mapping.mapping[terrain_humidity][t1.t]

            c0: V3[int] = get_v3_from_colour(HexAttr.colour_mapping.mapping[tt0])
            c1: V3[int] = get_v3_from_colour(HexAttr.colour_mapping.mapping[tt1])
            x: float = (t1.b - t) / (t1.b - t0.b)

            return interpolate_hsv(c1, c0, x)
        elif (t < t_min or t > t_max) and (h >= h_min or h <= h_max):
            # left/right edge, humidity gradient
            h0, h1 = HexAttr.gradient_bounds_h.find_bounds(h)

            tt0: TerrainType = HexAttr.terrain_mapping.mapping[h0.t][terrain_temperature]
            tt1: TerrainType = HexAttr.terrain_mapping.mapping[h1.t][terrain_temperature] 

            c0: V3[int] = get_v3_from_colour(HexAttr.colour_mapping.mapping[tt0])
            c1: V3[int] = get_v3_from_colour(HexAttr.colour_mapping.mapping[tt1])
            y: float = (h1.b - h) / (h1.b - h0.b)

            return interpolate_hsv(c1, c0, y)
        else:
            # middle, temperature and humidity gradient
            h0, h1 = HexAttr.gradient_bounds_h.find_bounds(h)
            t0, t1 = HexAttr.gradient_bounds_t.find_bounds(t)
            
            tt0: TerrainType = HexAttr.terrain_mapping.mapping[h0.t][t0.t]
            tt1: TerrainType = HexAttr.terrain_mapping.mapping[h0.t][t1.t]
            tt2: TerrainType = HexAttr.terrain_mapping.mapping[h1.t][t0.t]
            tt3: TerrainType = HexAttr.terrain_mapping.mapping[h1.t][t1.t]

            c0: V3[int] = get_v3_from_colour(HexAttr.colour_mapping.mapping[tt0])
            c1: V3[int] = get_v3_from_colour(HexAttr.colour_mapping.mapping[tt1])
            c2: V3[int] = get_v3_from_colour(HexAttr.colour_mapping.mapping[tt2])
            c3: V3[int] = get_v3_from_colour(HexAttr.colour_mapping.mapping[tt3])
            x = (t1.b - t) / (t1.b - t0.b)
            y = (h1.b - h) / (h1.b - h0.b)

            return bilinear_interpolate_hsv(c3, c2, c1, c0, x, y)
    
    def determine_colour(self) -> V3[int]:
        colour: V3[int] = V3(0, 0, 0)
        match cfg.COLOUR_SCHEME:
            case ColourScheme.TERRAIN:
                colour = self.get_terrain_colour(self._attr.altitude, self._attr.humidity, self._attr.temperature)
                colour = colour.to_float().scalar_mul(self.get_shading_mult()).to_int().min(V3(255, 255, 255)).max(V3(0, 0, 0))
            case ColourScheme.HEIGHTMAP:
                c1: pg.Color = cfg.C_LOW
                c2: pg.Color = cfg.C_HIGH
                colour = interpolate_v3(V3(c1[0], c1[1], c1[2]), V3(c2[1], c2[2], c2[3]), self._attr.altitude)
            case ColourScheme.HUMIDITYMAP:
                h: float = (self._attr.humidity * cfg.H_H_LOW + (1 - self._attr.humidity) * cfg.H_H_HIGH) / 360
                colour = V3(*colorsys.hls_to_rgb(h, cfg.H_L, cfg.H_S)).scalar_mul(255).to_int()
            case ColourScheme.TEMPERATUREMAP:
                h: float = (self._attr.temperature * cfg.T_H_LOW + (1 - self._attr.temperature) * cfg.T_H_HIGH) / 360
                colour = V3(*colorsys.hls_to_rgb(h, cfg.T_L, cfg.T_S)).scalar_mul(255).to_int()
            case ColourScheme.GRADIENT:
                of: V2[int] = Ax.ax_to_of(self._ax)
                x: float = of.x() / cfg.HEX_NOF_HEXES.x()
                y: float = of.y() / cfg.HEX_NOF_HEXES.y()
                colour = self.get_terrain_colour(1.0, x, y)
        return colour
    
    def update_element_rect(self) -> None:
        px: V2[int] = Ax.ax_to_px(self._ax)
        self._element.rect.topleft = (px.x() - round(Hex.dim.x() / 2), px.y() - round(Hex.dim.y() / 2))
        self._element.rect.size = Hex.dim.get()

    def update_element_image(self) -> None:
        self._element.image = HexSpriteStore.scaled_store[self._attr.terrain_type]
        pass

    def __str__(self) -> str:
        return "<Hex: ax = " + str(self._ax) + ", px = " + str(Ax.ax_to_px(self._ax)) + ">"
    

class HexChunk:

    # static class variables:
    size_px: V2[int]
    nof_hexes: int = cfg.HEX_INIT_CHUNK_SIZE
    size_overflow: int = 0

    @staticmethod
    def set_size_px() -> None:
        assert HexChunk.nof_hexes > 1
        match Hex.orientation:
            case HexOrientation.FLAT:
                w: int = round((HexChunk.nof_hexes - ((HexChunk.nof_hexes - 1) * (1/4))) * Hex.dim.x())
                h: int = round((HexChunk.nof_hexes + (1/2)) * Hex.dim.y())
            case HexOrientation.POINTY:
                w: int = round((HexChunk.nof_hexes + (1/2)) * Hex.dim.x())
                h: int = round((HexChunk.nof_hexes - ((HexChunk.nof_hexes - 1) * (1/4))) * Hex.dim.y())
        HexChunk.size_px = V2(w, h)

    @staticmethod
    def set_nof_hexes(nof_hexes: int) -> bool:
        # return true if changed
        # only change if within bounds and no overflow
        smaller: bool = nof_hexes < HexChunk.nof_hexes
        bigger: bool = nof_hexes > HexChunk.nof_hexes

        old_nof_hexes: int = HexChunk.nof_hexes
        if HexChunk.size_overflow == 0:  # only if no overflow attempt to change
            HexChunk.nof_hexes = min(cfg.HEX_MAX_CHUNK_SIZE, max(cfg.HEX_MIN_CHUNK_SIZE, nof_hexes))

        if old_nof_hexes != HexChunk.nof_hexes:
            return True
        else:  # size dit not change
            if smaller: HexChunk.size_overflow -= 1
            if bigger: HexChunk.size_overflow += 1
            return False
        
    @staticmethod
    def set_size_overflow(size_overflow: int) -> None:
        HexChunk.size_overflow: int = size_overflow

    @staticmethod
    def of_to_hex_idx(of: V2[int]) -> V2[int]:
        return V2(of.x() % HexChunk.nof_hexes, of.y() % HexChunk.nof_hexes)

    def __init__(self, idx: V2[int], parent_store: "HexStore") -> None:
        self._parent_store: HexStore = parent_store
        self._idx: V2[int] = idx
        self._hexes: list[list[Hex | None]] = [[None for _ in range(HexChunk.nof_hexes)] for _ in range(HexChunk.nof_hexes)]
        self._updated: bool = False
        self._filled: bool = False
        self._topleft: V2[int] = V2(0, 0)
        self._bottomright: V2[int] = V2(0, 0)

    def x(self) -> int:
        return self._idx.x()
    
    def y(self) -> int: 
        return self._idx.y()
    
    def idx(self) -> V2[int]:
        return self._idx

    def topleft(self) -> V2[int]:
        return self._topleft
    
    def bottomright(self) -> V2[int]:
        return self._bottomright
    
    def reset_topleft(self) -> None:
        self._topleft = self.calc_topleft()

    def reset_bottomright(self) -> None:
        self._bottomright = self.calc_bottomright()

    def hexes(self) -> list[list[Hex | None]]:
        return self._hexes
    
    def updated(self) -> bool:
        return self._updated
    
    def set_updated(self, updated: bool) -> None:
        self._updated = updated

    def filled(self) -> bool:
        return self._filled
    
    def set_filled(self, filled: bool) -> None:
        self._filled = filled
    
    def fill(self) -> None:
        self._parent_store.fill_chunk(self._idx)
    
    def calc_topleft(self) -> V2[int]:
        hex: Hex | None = self._hexes[0][0]
        if not hex:
            hex_idx: V2[int] = self._idx * V2(HexChunk.nof_hexes, HexChunk.nof_hexes)
            hex = self._parent_store.hexes()[hex_idx.x()][hex_idx.y()]
        return Ax.ax_to_px(hex.ax()) - (Hex.dim // V2(2, 2))
            
    def calc_bottomright(self,) -> V2[int]:
        hex: Hex | None = self._hexes[HexChunk.nof_hexes - 1][HexChunk.nof_hexes - 1]
        if not hex:
            hex_idx: V2[int] = (self._idx * V2(HexChunk.nof_hexes, HexChunk.nof_hexes)) + V2(HexChunk.nof_hexes - 1, HexChunk.nof_hexes - 1)
            hex = self._parent_store.hexes()[hex_idx.x()][hex_idx.y()]
        return Ax.ax_to_px(hex.ax()) + (Hex.dim // V2(2, 2))

    def get_hex(self, idx: V2[int]) -> Hex | None:
        return self._hexes[idx.x()][idx.y()]


class HexChunkSet:

    def __init__(self) -> None:
        self._chunks: set[HexChunk] = set()
        self._min_chunk_idx: V2[int] = V2(-1, -1)
        self._max_chunk_idx: V2[int] = V2(-1, -1)
        self._nof_chunks: V2[int] = V2(-1, -1)
    
    def chunks(self) -> set[HexChunk]:
        return self._chunks
    
    def min_chunk_idx(self) -> V2[int]:
        return self._min_chunk_idx
    
    def max_chunk_idx(self) -> V2[int]:
        return self._max_chunk_idx
    
    def nof_chunks(self) -> V2[int]:
        return self._nof_chunks
    
    def clear(self) -> None:
        self._chunks.clear()

    def remove_chunks(self) -> None:
        # remove chunks with x or y smaller than min
        self._chunks = {c for c in self._chunks if c.x() < self._min_chunk_idx.x()}
        self._chunks = {c for c in self._chunks if c.y() < self._min_chunk_idx.y()}
        # remove chunks with x or y bigger than max
        self._chunks = {c for c in self._chunks if c.x() > self._max_chunk_idx.x()}
        self._chunks = {c for c in self._chunks if c.y() > self._max_chunk_idx.y()}

    def add_chunks(self, chunks: list[list[HexChunk]]) -> None:
        for x in range(max(0, self._min_chunk_idx.x()), min(self._max_chunk_idx.x() + 1, HexStore.nof_chunks[0])):
            for y in range(max(0, self._min_chunk_idx.y()), min(self._max_chunk_idx.y() + 1, HexStore.nof_chunks[1])):
                self._chunks.add(chunks[x][y])

    def set_min_chunk_idx(self, min_chunk_idx: V2[int]) -> None:
        self._min_chunk_idx = min_chunk_idx

    def set_max_chunk_idx(self, max_chunk_idx: V2[int]) -> None:
        self._max_chunk_idx = max_chunk_idx

    def set_nof_chunks(self) -> None:
        nof_chunks_x: int = min(self._max_chunk_idx[0] + 1, HexStore.nof_chunks[0]) - max(0, self._min_chunk_idx[0])
        nof_chunks_y: int = min(self._max_chunk_idx[1] + 1, HexStore.nof_chunks[1]) - max(0, self._min_chunk_idx[1])
        self._nof_chunks = V2(nof_chunks_x, nof_chunks_y)
    

class HexStore:

    # static class variables
    nof_chunks: V2[int]

    @staticmethod
    def set_nof_chunks(nof_chunks: V2[int]) -> None:
        HexStore.nof_chunks = nof_chunks

    @staticmethod
    def of_to_chunk_idx(of: V2[int]) -> V2[int]:
        return V2(floor(of.x() / HexChunk.nof_hexes), floor(of.y() / HexChunk.nof_hexes))

    def __init__(self) -> None:
        HexStore.set_nof_chunks(cfg.HEX_INIT_STORE_SIZE)
        HexChunk.nof_hexes = cfg.HEX_INIT_CHUNK_SIZE

        assert HexStore.nof_chunks * V2(HexChunk.nof_hexes, HexChunk.nof_hexes) == cfg.HEX_NOF_HEXES

        HexChunk.set_size_overflow(cfg.HEX_INIT_CHUNK_OVERFLOW)
        HexChunk.set_size_px()

        self._hexes: list[list[Hex]] = [[Hex(Ax.of_to_ax(V2(x, y)).q(), Ax.of_to_ax(V2(x, y)).r()) for y in range(HexStore.nof_chunks.y() * HexChunk.nof_hexes)] for x in range(HexStore.nof_chunks.x() * HexChunk.nof_hexes)]
        self._chunks: list[list[HexChunk]] = [[HexChunk(V2(x, y), self) for y in range(HexStore.nof_chunks.y())] for x in range (HexStore.nof_chunks.x())]
        self._in_camera: HexChunkSet = HexChunkSet()

    def hexes(self) -> list[list[Hex]]:
        return self._hexes

    def chunks(self) -> list[list[HexChunk]]:
        return self._chunks
    
    def in_camera(self) -> HexChunkSet:
        return self._in_camera
    
    def in_camera_topleft(self) -> V2[int]:
        return self._chunks[self._in_camera.min_chunk_idx().x()][self._in_camera.min_chunk_idx().y()].calc_topleft()
    
    def in_camera_bottomright(self) -> V2[int]:
        return self._chunks[self._in_camera.max_chunk_idx().x()][self._in_camera.max_chunk_idx().y()].calc_bottomright()

    def update_in_camera(self, min_max_of: V2[V2[int]]) -> None:
        # get the chunk idx of the topleft (min) and bottomright (max) hexes on screen
        min_chunk_idx: V2[int] = HexStore.of_to_chunk_idx(min_max_of[0])
        max_chunk_idx: V2[int] = HexStore.of_to_chunk_idx(min_max_of[1])
        # if chunks in camera have changed, update the in-camera set
        if min_chunk_idx != self._in_camera.min_chunk_idx() or max_chunk_idx != self._in_camera.max_chunk_idx():
            self._in_camera.set_min_chunk_idx(V2(max(0, min_chunk_idx.x()), max(0, min_chunk_idx.y())))
            self._in_camera.set_max_chunk_idx(V2(min(HexStore.nof_chunks.x() - 1, max_chunk_idx.x()), min(HexStore.nof_chunks.y() - 1, max_chunk_idx.y())))
            # set the nof chunks in camera based on min/max indices
            self._in_camera.set_nof_chunks()
            # remove and add chunks as needed
            self._in_camera.remove_chunks()
            self._in_camera.add_chunks(self._chunks)
                
    def get_hex_at_of(self, of: V2[int]) -> Hex | None:
        if of.x() < 0 or of.x() >= len(self._hexes) or of.y() < 0 or of.y() >= len(self._hexes[0]):
            return None
        return self._hexes[of.x()][of.y()]
    
    def reset_chunks(self) -> None:
        # also resets chunk._updated for every chunk
        self._chunks: list[list[HexChunk]] = [[HexChunk(V2(x, y), self) for y in range(HexStore.nof_chunks.y())] for x in range (HexStore.nof_chunks.x())]
    
    def reset_chunk_update_status(self) -> None:
        for i in range(len(self._chunks)):
            for j in range(len(self._chunks[i])):
                self._chunks[i][j].set_updated(False)

    def fill_chunk(self, chunk_idx: V2[int]) -> None:
        min_hex_range: V2[int] = V2(chunk_idx.x() * HexChunk.nof_hexes, chunk_idx.y() * HexChunk.nof_hexes)
        max_hex_range: V2[int] = min_hex_range + V2(HexChunk.nof_hexes, HexChunk.nof_hexes)
        for i in range(min_hex_range.x(), max_hex_range.x()):
            for j in range(min_hex_range.y(), max_hex_range.y()):
                hex: Hex = self._hexes[i][j]
                hex_idx: V2[int] = HexChunk.of_to_hex_idx(Ax.ax_to_of(hex.ax()))
                self._chunks[chunk_idx.x()][chunk_idx.y()].hexes()[hex_idx.x()][hex_idx.y()] = hex

    def fill_chunks(self) -> None:
        # assume new size of chunk and nof chunks has been set
        for i in range(len(self._chunks)):
            for j in range(len(self._chunks[i])):
                self.fill_chunk(V2(i, j))
    

class HexController(pk.model.Model):
    i: int = 0

    @staticmethod
    def add_hex_to_view(hex_controller: "HexController | None", hex: Hex) -> None:
        assert hex_controller, "hex_controller = None"
        hex_controller._view.add(hex.element())

    @staticmethod
    def update_element(_: "HexController | None", hex: Hex) -> None:
        if V2(hex.element().rect.width, hex.element().rect.height) != Hex.dim:
            hex.update_element_rect()
            hex.update_element_image()

    @staticmethod
    def update_attr_and_colour(_: "HexController | None", hex: Hex) -> None:
        hex.set_attr(HexAttr(hex))
        hex.set_colour(hex.determine_colour())

    @staticmethod
    def update_topleft_and_bottomright(_: "HexController | None", chunk: HexChunk) -> None:
        chunk.reset_topleft()
        chunk.reset_bottomright()

    @staticmethod
    def init_store_job(hex_controller: "HexController") -> None:
        hex_controller._store = HexStore()
        hex_controller._store.fill_chunks()

    def init_store(self, display: pg.Surface) -> None:
        thread = Thread(target=self.init_store_job, args=[self])
        thread.start()
        hex_created: int = 0
        t_start: float = time.time()
        display_message(display, "        > hexes created: " + str(Hex.n_created) + "/" + str(cfg.HEX_NOF_HEXES.x() * cfg.HEX_NOF_HEXES.y()), False)
        while thread.is_alive():
            if hex_created != Hex.n_created:
                display_message(display, "        > hexes created: " + str(Hex.n_created) + "/" + str(cfg.HEX_NOF_HEXES.x() * cfg.HEX_NOF_HEXES.y()), True)
                hex_created = Hex.n_created
        t_end: float = time.time()
        display_message(display, "        > hexes created: " + str(Hex.n_created) + "/" + str(cfg.HEX_NOF_HEXES.x() * cfg.HEX_NOF_HEXES.y()) \
                               + " (took " + "{0:.3}".format(t_end - t_start) + "s)", True)
            
    def __init__(self, view: "HexView", display: pg.Surface) -> None:
        pk.model.Model.__init__(self)
        display_message(display, "    > initialising hex")
        Hex.orientation = cfg.HEX_ORIENTATION
        Hex.set_size(cfg.HEX_MAX_SIZE)
        display_message(display, "    > initialising hex attributes")
        HexAttr.init(True)
        display_message(display, "    > initialising sprite store")
        HexSpriteStore.init_store()
        display_message(display, "    > initialising hex store")
        self._store: HexStore
        self.init_store(display)  # spawns thread
        display_message(display, "    > initialising hex view")
        self._view: "HexView" = view
        self.apply_to_all_hex_in_store(HexController.add_hex_to_view)
        
    def store(self) -> HexStore:
        return self._store

    def update(self, dt: float) -> None:
        pk.model.Model.update(self, dt)

        if self._view.flags.request_reset_chunks:
            if cfg.DEBUG_INFO: print(HexController.i, "request_reset_chunks")
            self._view.flags.request_reset_chunks = False
            self._store.reset_chunks()

        if self._view.flags.request_reset_chunk_update_status:
            if cfg.DEBUG_INFO: print(HexController.i, "request reset chunk update status")
            self._view.flags.request_reset_chunk_update_status = False
            self._store.reset_chunk_update_status()

        if self._view.flags.request_reset_scaled_store:
            if cfg.DEBUG_INFO: print(HexController.i, "request_reset_scaled_store")
            self._view.flags.request_reset_scaled_store = False
            HexSpriteStore.reset_scaled_store()

        if self._view.flags.request_update_in_camera:
            if cfg.DEBUG_INFO: print(HexController.i, "request_update_in_camera")
            self._view.flags.request_update_in_camera = False
            self._store.update_in_camera(self._view.get_min_max_of())

        if self._view.flags.request_update_chunk_surface:
            if cfg.DEBUG_INFO: print(HexController.i, "request_update_chunk_surface")
            self._view.flags.request_update_chunk_surface = False
            self._view.update_chunk_surface(self._store.in_camera(), self._store.in_camera_topleft(), self._store.in_camera_bottomright())

        if self._view.flags.request_update_and_add_single_chunk_to_surface:
            # flag gets reset by view
            self._view.update_and_add_single_chunk_to_surface()

        HexController.i += 1
        
    def apply_to_all_hex_in_store(self, f: Callable[["HexController | None", Hex], None]) -> None:
        for x in range(len(self._store.hexes())):
            for y in range(len(self._store.hexes()[x])):
                f(self, self._store.hexes()[x][y])

    def apply_to_all_hex_in_camera(self, f: Callable[["HexController | None", Hex], None]) -> None:
        for chunk in self._store.in_camera().chunks():
            self.apply_to_all_hex_in_chunk(chunk, f)

    def apply_to_all_hex_in_chunk(self, chunk: HexChunk, f: Callable[["HexController | None", Hex], None]) -> None:
        for x in range(cfg.HEX_INIT_CHUNK_SIZE):
            for y in range(cfg.HEX_INIT_CHUNK_SIZE):
                hex: Hex | None = chunk.get_hex(V2(x, y))
                if isinstance(hex, Hex): f(self, hex)

    def apply_to_all_chunk_in_camera(self, f: Callable[["HexController | None", HexChunk], None]) -> None:
        for chunk in self._store.in_camera().chunks():
            f(self, chunk)

    def get_hex_at_px(self, px: V2[int], offset: V2[int] = V2(0, 0)) -> Hex | None:
        ax: Ax = Ax.px_to_ax_offset(px, offset)
        of: V2[int] = Ax.ax_to_of(ax)
        return self._store.get_hex_at_of(of) 
    
    def reset_map(self) -> None:
        t_start: float = time.time()
        cfg.read_terrain_config()
        HexAttr.init(True)
        self.apply_to_all_hex_in_store(HexController.update_attr_and_colour)
        self._view.flags.init = True
        self._view.update_chunk_surface(self._store.in_camera(), self._store.in_camera_topleft(), self._store.in_camera_bottomright())
        t: float = time.time() - t_start
        if cfg.DEBUG_INFO: print("reset_map took " + "{0:.3}".format(t) + " ms")

    def redraw_map(self) -> None:
        t_start: float = time.time()
        cfg.read_terrain_config()
        HexAttr.init(False)
        self.apply_to_all_hex_in_store(HexController.update_attr_and_colour)
        self._view.flags.init = True
        self._view.update_chunk_surface(self._store.in_camera(), self._store.in_camera_topleft(), self._store.in_camera_bottomright())
        t: float = time.time() - t_start
        if cfg.DEBUG_INFO: print("redraw_map took " + "{0:.3}".format(t) + " ms")
