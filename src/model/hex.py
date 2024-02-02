import pygame as pg
import numpy as np
import random as rnd

from enum import Enum
from math import sqrt, cos, sin, pi

from elements import SpriteElement
from view import View
from debug import debug


# https://www.redblobgames.com/grids/hexagons/


class HexOrientation(Enum):
    FLAT = 1
    POINTY = 2


class AxialCoordinates:

    ax_to_px_flat = np.array([[3./2, 0], [sqrt(3)/2, sqrt(3)]])
    px_to_ax_flat = np.array([[2./3, 0], [-1./3, sqrt(3)/3]])
    ax_to_px_pointy = np.array([[sqrt(3), sqrt(3)/2], [0, 3./2]])
    px_to_ax_pointy = np.array([[sqrt(3)/3, -1./3], [0, 2./3]])

    def __init__(self, c: np.ndarray) -> None:
        self.c: np.ndarray = c  # axial coords [q, r]; q + r + s = 0

    def q(self) -> np.int32:
        return self.c[0]
    
    def r(self) -> np.int32:
        return self.c[1]
    
    def s(self) -> np.int64:
        return -self.q() - self.r()
    
    def get(self) -> np.ndarray:
        return self.c
    
    def get_cb(self) -> np.ndarray:
        return np.ndarray([self.q(), self.r(), self.s()])
    
    def ax_to_px(ax: "AxialCoordinates") ->  np.ndarray:
        match Hex.orientation:
            case HexOrientation.FLAT:
                return (Hex.size * AxialCoordinates.ax_to_px_flat).dot(ax.get())
            case HexOrientation.POINTY:
                return (Hex.size * AxialCoordinates.ax_to_px_pointy).dot(ax.get())
            
    def px_to_ax(px: np.ndarray) -> "AxialCoordinates":
        match Hex.orientation:
            case HexOrientation.FLAT:
                return AxialCoordinates((Hex.size * AxialCoordinates.px_to_ax_flat).dot(px))
            case HexOrientation.POINTY:
                return AxialCoordinates((Hex.size * AxialCoordinates.px_to_ax_pointy).dot(px))
            
    def ax_to_of(ax: "AxialCoordinates",) -> np.ndarray:  # odd-r
        match Hex.orientation:
            case HexOrientation.FLAT:
                return np.array([ax.q(), ax.r() + (ax.q() - (ax.q()&1)) / 2])
            case HexOrientation.POINTY:
                return np.array([ax.q() + (ax.r() - (ax.r()&1)) / 2, ax.r()])
            
    def of_to_ax(of: np.ndarray) -> "AxialCoordinates":  # odd-q
        match Hex.orientation:
            case HexOrientation.FLAT:
                return AxialCoordinates([of[0], of[1] - (of[0] - (of[0]&1)) / 2])
            case HexOrientation.POINTY:
                return AxialCoordinates([of[0] - (of[1] - (of[1]&1)) / 2, of[1]])
    

class HexDimension:

    def __init__(self) -> None:
        pass


class Hex:

    # static functions
    def calc_dim() -> np.ndarray:
        match Hex.orientation:
            case HexOrientation.FLAT:
                return np.array([2 * Hex.size, sqrt(3) * Hex.size])
            case HexOrientation.POINTY:
                return np.array([sqrt(3) * Hex.size, 2 * Hex.size])
            
    def set_size(size: float) -> None:
        Hex.size = size
        Hex.dim = Hex.calc_dim()
            
    # static class variables
    size: float
    dim: np.ndarray
    orientation: HexOrientation = HexOrientation.FLAT

    def __init__(self, q: int, r: int) -> None:
        self.ax: AxialCoordinates = AxialCoordinates(np.array([q, r]))
        self.px: np.ndarray =AxialCoordinates.ax_to_px(self.ax)

        surface = pg.Surface([self.dim[0], self.dim[1]], pg.SRCALPHA)
        rgb = [abs((22*self.ax.q())%255), 
               abs((22*self.ax.r())%255), 
               abs((22*self.ax.s())%255)]
        draw_regular_polygon(surface, 
                             rgb, 
                             6, 
                             Hex.size, 
                             [self.dim[0] / 2, self.dim[1] / 2])
        pg.draw.circle(surface, 
                       [255 - rgb[0], 255 - rgb[1], 255 - rgb[2]], 
                       [self.dim[0] / 2, self.dim[1] / 2],
                       Hex.size / 10)
        self.element: SpriteElement = SpriteElement(pos=tuple([self.px[0], self.px[1]]), img=surface)
    

class HexController:

    def __init__(self, view: View, size: float) -> None:
        self.view = view
        self.hexes: list[Hex] = []
        Hex.set_size(size)

    def add_hex(self, q: int, r: int):
        self.hexes.append(Hex(q, r))

    def fill_screen(self, w: int, h: int):
        cols = int(w / Hex.dim[1])
        rows = int(h / Hex.dim[0])
        debug["cols"] = cols
        debug["rows"] = rows
        for col in range(0, cols):
            for row in range(0, rows):
                ax = AxialCoordinates.of_to_ax(np.array([col, row]))
                self.hexes.append(Hex(ax.q(), ax.r()))
        for hex in self.hexes:
            self.view.add(hex.element)


def draw_regular_polygon(surface, color, vertex_count, radius, position, width=0):
    n, r = vertex_count, radius
    x, y = position
    pg.draw.polygon(surface, color, [
        (x + r * cos(2 * pi * i / n), y + r * sin(2 * pi * i / n))
        for i in range(n)
    ], width)