import pygame as pg
import numpy as np
import random as rnd

from enum import Enum
from math import sqrt, cos, sin, pi

from elements import SpriteElement


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
        return -self.q() - self.c.r()
    
    def get(self) -> np.ndarray:
        return self.c
    
    def get_cb(self) -> np.ndarray:
        return np.ndarray([self.q(), self.r(), self.s()])
    
    def ax_to_px(ax: "AxialCoordinates", orientation: HexOrientation, size: float) ->  np.ndarray:
        match orientation:
            case HexOrientation.FLAT:
                return (size * AxialCoordinates.ax_to_px_flat).dot(ax.get())
            case HexOrientation.POINTY:
                return (size * AxialCoordinates.ax_to_px_pointy).dot(ax.get())
            
    def px_to_ax(px: np.ndarray, orientation: HexOrientation, size: float) -> "AxialCoordinates":
        match orientation:
            case HexOrientation.FLAT:
                return AxialCoordinates((size * AxialCoordinates.px_to_ax_flat).dot(px))
            case HexOrientation.POINTY:
                return AxialCoordinates((size * AxialCoordinates.px_to_ax_pointy).dot(px))
    

class HexDimension:

    def __init__(self) -> None:
        pass


class Hex:
    orientation: HexOrientation = HexOrientation.FLAT

    def __init__(self, q: int, r: int, size: float) -> None:
        self.ax: AxialCoordinates = AxialCoordinates(np.array([q, r]))
        self.size: float = size
        self.dim: np.ndarray = self.calc_dim()        
        self.px: np.ndarray =AxialCoordinates.ax_to_px(self.ax, Hex.orientation, self.size)

        surface = pg.Surface([self.dim[0], self.dim[1]], pg.SRCALPHA)
        draw_regular_polygon(surface, 
                             [int(rnd.uniform(0, 1) * 255), int(rnd.uniform(0, 1) * 255), int(rnd.uniform(0, 1) * 255)], 
                             6, 
                             self.size, 
                             [self.dim[0] / 2, self.dim[1] / 2])
        self.element: SpriteElement = SpriteElement(pos=tuple([self.px[0], self.px[1]]), img=surface)
    
    def calc_dim(self) -> np.ndarray:
        match Hex.orientation:
            case HexOrientation.FLAT:
                return np.array([2 * self.size, sqrt(3) * self.size])
            case HexOrientation.POINTY:
                return np.array([sqrt(3) * self.size, 2 * self.size])


def draw_regular_polygon(surface, color, vertex_count, radius, position, width=0):
    n, r = vertex_count, radius
    x, y = position
    pg.draw.polygon(surface, color, [
        (x + r * cos(2 * pi * i / n), y + r * sin(2 * pi * i / n))
        for i in range(n)
    ], width)
    
