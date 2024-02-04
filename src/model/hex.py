from typing import Any
import pygame as pg
import pynkie as pk
import numpy as np

from enum import Enum
from math import sqrt, cos, sin, pi

# https://www.redblobgames.com/grids/hexagons/

class HexOrientation(Enum):
    FLAT = 1
    POINTY = 2

class AxialCoordinates:

    ax_to_px_flat: list[list[float]] = [[3./2, 0], [sqrt(3)/2, sqrt(3)]]
    px_to_ax_flat: list[list[float]] = [[2./3, 0], [-1./3, sqrt(3)/3]]
    ax_to_px_pointy: list[list[float]] = [[sqrt(3), sqrt(3)/2], [0, 3./2]]
    px_to_ax_pointy: list[list[float]] = [[sqrt(3)/3, -1./3], [0, 2./3]]

    def __init__(self, c: tuple[int, int]) -> None:
        self.c: tuple[int, int] = c  # axial coords [q, r]; q + r + s = 0

    def __str__(self):
        return "<q = " + str(self.q()) + ", r = " + str(self.r()) + ">"

    def q(self) -> int:
        return self.c[0]
    
    def r(self) -> int:
        return self.c[1]
    
    def s(self) -> int:
        return -self.q() -self.r()
    
    def get(self) -> tuple[int, int]:
        return self.c
    
    def get_cb(self) -> tuple[int, int, int]:
        return (self.q(), self.r(), self.s())
    
    @staticmethod
    def ax_round(c_frac: list[float]) -> tuple[int, int]:
        q: int = round(c_frac[0])
        r: int = round(c_frac[1])
        s: int = round(-c_frac[0] - c_frac[1])

        q_diff: float = abs(q - c_frac[0])
        r_diff: float = abs(r - c_frac[1])
        s_diff: float = abs(s - (-c_frac[0] - c_frac[1]))

        if q_diff > r_diff and q_diff > s_diff:
            q = -r-s
        elif r_diff > s_diff:
            r = -q-s

        return (q, r)

    @staticmethod
    def ax_to_px(ax: "AxialCoordinates") -> tuple[int, int]:
        match Hex.orientation:
            case HexOrientation.FLAT:
                px: list[Any] = (Hex.size * np.array(AxialCoordinates.ax_to_px_flat).dot(ax.get())).tolist()
            case HexOrientation.POINTY:
                px: list[Any] = (Hex.size * np.array(AxialCoordinates.ax_to_px_pointy).dot(ax.get())).tolist()
        return (round(px[0]), round(px[1]))
            
    @staticmethod
    def px_to_ax(px: tuple[int, int]) -> "AxialCoordinates":
        match Hex.orientation:
            case HexOrientation.FLAT:
                return AxialCoordinates(AxialCoordinates.ax_round(
                    (np.array(AxialCoordinates.px_to_ax_flat).dot(px) / Hex.size).tolist()))
            case HexOrientation.POINTY:
                return AxialCoordinates(AxialCoordinates.ax_round(
                    (np.array(AxialCoordinates.px_to_ax_pointy).dot(px) / Hex.size).tolist()))
            
    @staticmethod
    def ax_to_of(ax: "AxialCoordinates") -> tuple[int, int]:  # odd-r
        match Hex.orientation:
            case HexOrientation.FLAT:
                return (ax.q(), round(ax.r() + (ax.q() - (ax.q()&1)) / 2))
            case HexOrientation.POINTY:
                return (round(ax.q() + (ax.r() - (ax.r()&1)) / 2), ax.r())
            
    @staticmethod
    def of_to_ax(of: tuple[int, int]) -> "AxialCoordinates":  # odd-q
        match Hex.orientation:
            case HexOrientation.FLAT:
                return AxialCoordinates((of[0], round(of[1] - (of[0] - (of[0]&1)) / 2)))
            case HexOrientation.POINTY:
                return AxialCoordinates((round(of[0] - (of[1] - (of[1]&1)) / 2), of[1]))
            

class HexDimension:

    def __init__(self) -> None:
        pass


class Hex(pk.model.Model):

    @staticmethod
    def calc_dim() -> tuple[int, int]:
        match Hex.orientation:
            case HexOrientation.FLAT:
                return (Hex.size * 2, round(sqrt(3) * Hex.size))
            case HexOrientation.POINTY:
                return (round(sqrt(3) * Hex.size), Hex.size * 2)
            
    @staticmethod
    def calc_spacing() -> tuple[int, int]:
        match Hex.orientation:
            case HexOrientation.FLAT:
                return (round(Hex.size * (3/2)), round(sqrt(3) * Hex.size))
            case HexOrientation.POINTY:
                return (round(sqrt(3) * Hex.size), round(Hex.size * (3/2)))
            
    @staticmethod
    def set_size(size: int) -> None:
        Hex.size = size
        Hex.dim = Hex.calc_dim()
        Hex.spacing = Hex.calc_spacing()
            
    # static class variables
    size: int  # "radius" of the hex (center to vertex)
    dim: tuple[int, int]  # width and height of the hex
    spacing: tuple[int, int]  # distance between centers of hexes
    orientation: HexOrientation = HexOrientation.FLAT

    def __init__(self, q: int, r: int) -> None:
        self.ax: AxialCoordinates = AxialCoordinates((q, r))
        self.px: tuple[int, int] = AxialCoordinates.ax_to_px(self.ax)

        surface = self.make_surface()
        pos: tuple[int, int] = (self.px[0] - int(Hex.dim[0] / 2), self.px[1] - int(Hex.dim[1] / 2));
        self.element: pk.elements.SpriteElement = pk.elements.SpriteElement(pos=pos, img=surface)

    def __str__(self) -> str:
        return "<Hex: ax = " + str(self.ax) + ", px = " + str(self.px) + ">"

    def make_surface(self) -> pg.Surface:
        rgb: list[int] = [abs((22 * int(self.ax.q()))%255), 
                          abs((22 * int(self.ax.r()))%255), 
                          abs((22 * int(self.ax.s()))%255)]
        rgb_inv: list[int] = [255 - rgb[0], 255 - rgb[1], 255 - rgb[2]]
        surface = pg.Surface([self.dim[0], self.dim[1]], pg.SRCALPHA)
        # surface.fill(rgb_inv)
        draw_hex(surface, 
                             rgb,
                             Hex.size, 
                             (round(self.dim[0] / 2), round(self.dim[1] / 2)))
        pg.draw.circle(surface, 
                       rgb_inv, 
                       [self.dim[0] / 2, self.dim[1] / 2],
                       int(Hex.size) / 10)
        return surface


def draw_hex(surface: pg.Surface, color: list[int], radius: int, position: tuple[int, int], width: int=0) -> None:
    r: int = radius
    x: int = position[0]
    y: int = position[1]
    match Hex.orientation:
        case HexOrientation.FLAT:
            pg.draw.polygon(surface, color, [
                (x + r * cos(2 * pi * i / 6), y + r * sin(2 * pi * i / 6))
                for i in range(6)
            ], width)
        case HexOrientation.POINTY:
            pg.draw.polygon(surface, color, [
                (x + r * cos(2 * pi * i / 6 + 2 * pi / 12), y + r * sin(2 * pi * i / 6 + 2 * pi / 12))
                for i in range(6)
            ], width)