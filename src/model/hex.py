from typing import Any, Callable, TYPE_CHECKING
from config import HEX_CHUNK_SIZE, HEX_INIT_SIZE, HEX_ORIENTATION, HEX_NOF_CHUNKS, NR_HEX_OUTSIDE_BOUNDS, HexOrientation
import pygame as pg
import pynkie as pk
import numpy as np

from math import floor, sqrt, cos, sin, pi
from util import add_tuple, f2

if TYPE_CHECKING:
    from view.hex import HexView

# https://www.redblobgames.com/grids/hexagons/


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
    def ax_to_px_offset(ax: "AxialCoordinates", offset: tuple[int, int]) -> tuple[int, int]:
        match Hex.orientation:
            case HexOrientation.FLAT:
                px: list[Any] = (Hex.size * np.array(AxialCoordinates.ax_to_px_flat).dot(ax.get())).tolist()
            case HexOrientation.POINTY:
                px: list[Any] = (Hex.size * np.array(AxialCoordinates.ax_to_px_pointy).dot(ax.get())).tolist()
        return f2(add_tuple((round(px[0]), round(px[1])), offset))
            
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
    def px_to_ax_offset(px: tuple[int, int], offset: tuple[int, int]) -> "AxialCoordinates":
        px_offset: tuple[int, int] = f2(add_tuple(px, offset))
        match Hex.orientation:
            case HexOrientation.FLAT:
                return AxialCoordinates(AxialCoordinates.ax_round(
                    (np.array(AxialCoordinates.px_to_ax_flat).dot(px_offset) / Hex.size).tolist()))
            case HexOrientation.POINTY:
                return AxialCoordinates(AxialCoordinates.ax_round(
                    (np.array(AxialCoordinates.px_to_ax_pointy).dot(px_offset) / Hex.size).tolist()))
            
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
        Hex.size_updated = True
            
    # static class variables
    size: int  # "radius" of the hex (center to vertex)
    dim: tuple[int, int]  # width and height of the hex
    spacing: tuple[int, int]  # distance between centers of hexes
    orientation: HexOrientation  # pointy or flat top

    def __init__(self, q: int, r: int) -> None:
        self.ax: AxialCoordinates = AxialCoordinates((q, r))
        self.px: tuple[int, int] = AxialCoordinates.ax_to_px(self.ax)

        surface: pg.Surface = self.make_surface()
        pos: tuple[int, int] = (self.px[0] - int(Hex.dim[0] / 2), self.px[1] - int(Hex.dim[1] / 2));
        self.element: pk.elements.SpriteElement = pk.elements.SpriteElement(pos=pos, img=surface)

    def __str__(self) -> str:
        return "<Hex: ax = " + str(self.ax) + ", px = " + str(self.px) + ">"
    
    def update_size_and_pos(self) -> None:
        if (self.element.image.get_width() != Hex.dim[0] or self.element.image.get_height() != Hex.dim[1]):
            self.element.image = self.make_surface()
            self.px = AxialCoordinates.ax_to_px(self.ax)
            pos: tuple[int, int] = (self.px[0] - int(Hex.dim[0] / 2), self.px[1] - int(Hex.dim[1] / 2))
            self.element.rect.topleft = pos

    def make_surface(self) -> pg.Surface:
        rgb: list[int] = [abs((5 * int(self.ax.q()))%255), 
                          abs((5 * int(self.ax.r()))%255), 
                          abs((5 * int(self.ax.s()))%255)]
        # rgb_inv: list[int] = [255 - rgb[0], 255 - rgb[1], 255 - rgb[2]]
        surface = pg.Surface([self.dim[0], self.dim[1]], pg.SRCALPHA)
        # surface.fill(rgb_inv)
        draw_hex(surface, 
                             rgb,
                             Hex.size, 
                             (round(self.dim[0] / 2), round(self.dim[1] / 2)))
        # pg.draw.circle(surface, 
        #                rgb_inv, 
        #                [self.dim[0] / 2, self.dim[1] / 2],
        #                int(Hex.size) / 10)
        # surface: pg.Surface = surface.convert()
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


class HexChunk:

    @staticmethod
    def of_to_hex_idx(of: tuple[int, int]) -> tuple[int, int]:
        return (of[0] % HEX_CHUNK_SIZE, of[1] % HEX_CHUNK_SIZE)

    def __init__(self, idx: tuple[int, int]) -> None:
        self.idx: tuple[int, int] = idx  # chunk idx in store 
        self.hexes: list[list[Hex | None]] = [[None for _ in range(HEX_CHUNK_SIZE)] for _ in range(HEX_CHUNK_SIZE)]

    def fill_chunk(self) -> None:
        for x in range(HEX_CHUNK_SIZE):
            for y in range(HEX_CHUNK_SIZE):
                of_x: int = HEX_CHUNK_SIZE * self.idx[0] + x
                of_y: int = HEX_CHUNK_SIZE * self.idx[1] + y
                ax: AxialCoordinates = AxialCoordinates.of_to_ax((of_x, of_y))
                self.hexes[x][y] = Hex(ax.q(), ax.r())


class HexStore:

    @staticmethod
    def of_to_chunk_idx(of: tuple[int, int]) -> tuple[int, int]:
        return (floor(of[0] / HEX_CHUNK_SIZE), floor(of[1] / HEX_CHUNK_SIZE))

    def __init__(self) -> None:
        self.hex_chunks: list[list[HexChunk]] = [[HexChunk((i, j)) for j in range(HEX_NOF_CHUNKS[0])] for i in range (HEX_NOF_CHUNKS[1])]
        self.in_camera: set[HexChunk] = set()

    # store negative coordinates on odd indices
    def fill_store(self) -> None:
        for chunk_row in self.hex_chunks:
            for chunk in chunk_row:
                chunk.fill_chunk()

    def update_in_camera(self, min_max_of: tuple[tuple[int, int], tuple[int, int]]) -> None:
        self.in_camera.clear()
        pk.debug.debug["minmaxof chunk"] = min_max_of
        min_chunk_idx: tuple[int, int] = HexStore.of_to_chunk_idx(min_max_of[0])
        max_chunk_idx: tuple[int, int] = HexStore.of_to_chunk_idx(min_max_of[1])
        pk.debug.debug["minmax chunk"] = [min_chunk_idx, max_chunk_idx]
        for x in range(max(0, min_chunk_idx[0]), min(max_chunk_idx[0] + 1, HEX_NOF_CHUNKS[0])):
            for y in range(max(0, min_chunk_idx[1]), min(max_chunk_idx[1] + 1, HEX_NOF_CHUNKS[1])):
                self.in_camera.add(self.hex_chunks[x][y])
        pk.debug.debug["nr in camera"] = len(self.in_camera)

    def get_hex_at_of(self, of: tuple[int, int]) -> Hex | None:
        chunk_idx: tuple[int, int] = HexStore.of_to_chunk_idx(of)
        if (chunk_idx[0] < 0 or chunk_idx[0] >= HEX_NOF_CHUNKS[0] or chunk_idx[1] < 0 or chunk_idx[1] > HEX_NOF_CHUNKS[1]):
            return None
        chunk: HexChunk = self.hex_chunks[chunk_idx[0]][chunk_idx[1]]
        hex_idx: tuple[int, int] = HexChunk.of_to_hex_idx(of)
        hex: Hex | None = chunk.hexes[hex_idx[0]][hex_idx[1]]
        return hex


class HexController(pk.model.Model):
    i: int = 0

    @staticmethod
    def add_hex_to_view(hex_controller: "HexController", hex: Hex) -> None:
        hex_controller.hex_view.add(hex.element)

    @staticmethod
    def update_size_and_pos(hex_controller: "HexController", hex: Hex) -> None:
        # only update hex coordinates and element rect if: 
        # 1. element rect size is not equal to new hex dimensions (after zoom)
        # afterwards, coordinates are correctly adjusted and element rect has the right size until the next zoom takes place
        if (hex.element.rect.width != Hex.dim[0] or hex.element.rect.height != Hex.dim[1]):
            hex.px = AxialCoordinates.ax_to_px(hex.ax)
            pos: tuple[int, int] = (hex.px[0] - int(Hex.dim[0] / 2), hex.px[1] - int(Hex.dim[1] / 2))
            hex.element.rect.topleft = pos
            hex.element.rect.size = Hex.dim

    @staticmethod
    def update_image(hex_controller: "HexController", hex: Hex) -> None:
        # only update image if:
        # 1. image size is not equal to size of element (meaning a zoom has taken place)
        # 2. image is in frame (for the first time after a zoom)
        # afterwards, image size is equal to rect size until the next zoom takes place
        if hex.element.rect.width != hex.element.image.get_width() or hex.element.rect.height != hex.element.image.get_height():
            hex.element.image = hex.make_surface()

    def __init__(self, view: "HexView") -> None:
        self.hex_view: "HexView" = view
        self.hex_store: HexStore = HexStore()
        Hex.orientation = HEX_ORIENTATION
        Hex.set_size(HEX_INIT_SIZE)

    def handle_event(self, event: pg.event.Event) -> None:
        pk.events.EventListener.handle_event(self, event)
        match event.type:
            case pg.MOUSEWHEEL:
                self.on_mouse_wheel(event)
            case _: pass

    def on_mouse_wheel(self, _: pg.event.Event) -> None:
        pass

    def update(self, dt: float) -> None:
        super().update(dt)
        if self.hex_view.request_in_camera:
            self.hex_store.update_in_camera(self.hex_view.get_min_max_of())
            self.apply_to_all_in_camera(HexController.update_size_and_pos)
            self.apply_to_all_in_camera(HexController.update_image)
            self.hex_view.in_camera = self.hex_store.in_camera
            self.hex_view.request_in_camera = False
        
    def apply_to_all_in_store(self, f: Callable[["HexController", Hex], None]) -> None:
        for chunk_row in self.hex_store.hex_chunks:
            for chunk in chunk_row:
                self.apply_to_all_in_chunk(chunk, f)

    def apply_to_all_in_camera(self, f: Callable[["HexController", Hex], None]) -> None:
        for chunk in self.hex_store.in_camera:
            self.apply_to_all_in_chunk(chunk, f)

    def apply_to_all_in_chunk(self, chunk: HexChunk, f: Callable[["HexController", Hex], None]) -> None:
        for hex_row in chunk.hexes:
            for hex in hex_row:
                if isinstance(hex, Hex): f(self, hex)

    def get_hex_at_px(self, pos: tuple[int, int], offset: tuple[int, int] = (0, 0)) -> Hex | None:
        ax: AxialCoordinates = AxialCoordinates.px_to_ax_offset(pos, offset)
        of: tuple[int, int] = AxialCoordinates.ax_to_of(ax)
        
        return self.hex_store.get_hex_at_of(of)