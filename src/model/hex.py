from typing import Any, Callable, TYPE_CHECKING
import pygame as pg
import pynkie as pk
import numpy as np

from math import ceil, floor, sqrt, cos, sin, pi
from util import V2, V3
from config import HEX_CHUNK_SIZE, HEX_INIT_SIZE, HEX_ORIENTATION, HEX_NOF_CHUNKS, ZOOM_STEP_FACTOR, HexOrientation

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
        match Hex.orientation:
            case HexOrientation.FLAT:
                return V2(ax.q(), round(ax.r() + (ax.q() - (ax.q()&1)) / 2))
            case HexOrientation.POINTY:
                return V2(round(ax.q() + (ax.r() - (ax.r()&1)) / 2), ax.r())
            
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
        return -self.q() -self.r()
    
    def get(self) -> V2[int]:
        return self.c
    
    def get_cb(self) -> V3[int]:
        return V3(self.q(), self.r(), self.s())


class HexSpriteStore:

    # static class variables
    store: list[pg.Surface] = []

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
    def make_surface(of: V2[int], draw_center: bool = True) -> pg.Surface:
        ax: Ax = Ax.of_to_ax(of)
        rgb: list[int] = [abs((63 * ax.q() + 1) % 255), 
                          abs((187 * ax.r() + 1) % 255), 
                          abs((229 * ax.s() + 1) % 255)]
        rgb_inv: list[int] = [255 - rgb[0], 255 - rgb[1], 255 - rgb[2]]
        surface = pg.Surface((Hex.dim.x(), Hex.dim.y()), pg.SRCALPHA)
        HexSpriteStore.draw_hex(surface, rgb, Hex.size, V2(round(Hex.dim.x() / 2), round(Hex.dim.y() / 2)), 35)
        if draw_center:
            pg.draw.circle(surface, rgb_inv, [Hex.dim.x() / 2, Hex.dim.y() / 2], 5)
        return surface

    @staticmethod
    def init_store() -> None:
        HexSpriteStore.store = [HexSpriteStore.make_surface(V2(i, j)) for i in range(10) for j in range(10)]
        HexSpriteStore.store[0] = HexSpriteStore.make_surface(V2(3, 3))  # test hex


class Hex(pk.model.Model):
    
    # static class variables
    size: int  # "radius" of the hex (center to vertex)
    dim: V2[int]  # width and height of the hex
    dim_float: V2[float]
    spacing: V2[int]  # distance between centers of hexes
    spacing_float: V2[float]
    orientation: HexOrientation  # pointy or flat top

    @staticmethod
    def calc_dim() -> tuple[V2[int], V2[float]]:
        match Hex.orientation:
            case HexOrientation.FLAT:
                return (V2(Hex.size * 2, round(sqrt(3) * Hex.size)), V2(Hex.size * 2, sqrt(3) * Hex.size))
            case HexOrientation.POINTY:
                return (V2(round(sqrt(3) * Hex.size), Hex.size * 2), V2(sqrt(3) * Hex.size, Hex.size * 2))
            
    @staticmethod
    def calc_spacing() -> tuple[V2[int], V2[float]]:
        match Hex.orientation:
            case HexOrientation.FLAT:
                return (V2(round(Hex.size * (3/2)), round(sqrt(3) * Hex.size)), V2(Hex.size * (3/2), sqrt(3) * Hex.size))
            case HexOrientation.POINTY:
                return (V2(round(sqrt(3) * Hex.size), round(Hex.size * (3/2))), V2(sqrt(3) * Hex.size, Hex.size * (3/2)))
            
    @staticmethod
    def set_size(size: int) -> None:
        Hex.size = size
        Hex.dim, Hex.dim_float = Hex.calc_dim()
        Hex.spacing, Hex.spacing_float = Hex.calc_spacing()
        Hex.size_updated = True
            
    def __init__(self, q: int, r: int) -> None:
        self._ax: Ax = Ax(V2(q, r))
        self._px: V2[int] = Ax.ax_to_px(self._ax)

        chunk_idx: V2[int] = HexStore.of_to_chunk_idx(Ax.ax_to_of(self._ax))
        self._sprite_idx: int = chunk_idx.x() + chunk_idx.y() % len(HexSpriteStore.store)

        pos: tuple[int, int] = (self._px.x() - int(Hex.dim.x() / 2), self._px.y() - int(Hex.dim.y() / 2));
        self._element: pk.elements.SpriteElement = pk.elements.SpriteElement(pos=pos, img=HexSpriteStore.store[self._sprite_idx])

    def ax(self) -> Ax:
        return self._ax
    
    def px(self) -> V2[int]:
        return self._px
    
    def reset_px(self) -> None:
        self._px = Ax.ax_to_px(self._ax)
    
    def element(self) -> pk.elements.SpriteElement:
        return self._element
    
    def update_element_rect(self) -> None:
        self._element.rect.topleft = (self._px.x() - int(Hex.dim.x() / 2), self._px.y() - int(Hex.dim.y() / 2))
        self._element.rect.size = Hex.dim.get()

    def update_element_image(self) -> None:
        # scale to new size + 1px padding (to avoid rounding gaps)
        self._element.image = pg.transform.scale(HexSpriteStore.store[self._sprite_idx], (self._element.rect.width + 1, self._element.rect.height + 1))

    def __str__(self) -> str:
        return "<Hex: ax = " + str(self._ax) + ", px = " + str(self._px) + ">"
    

class HexChunk:

    # static class variables:
    size: V2[int]

    @staticmethod
    def set_size() -> None:
        assert HEX_CHUNK_SIZE > 1
        match Hex.orientation:
            case HexOrientation.FLAT:
                w: int = round((HEX_CHUNK_SIZE - ((HEX_CHUNK_SIZE - 1)* (1/4))) * Hex.dim.x())
                h: int = round((HEX_CHUNK_SIZE + (1/2)) * Hex.dim.y())
            case HexOrientation.POINTY:
                w: int = round((HEX_CHUNK_SIZE + (1/2)) * Hex.dim.x())
                h: int = round((HEX_CHUNK_SIZE - ((HEX_CHUNK_SIZE - 1) * (1/4))) * Hex.dim.y())
        HexChunk.size = V2(w, h)

    @staticmethod
    def of_to_hex_idx(of: V2[int]) -> V2[int]:
        return V2(of.x() % HEX_CHUNK_SIZE, of.y() % HEX_CHUNK_SIZE)

    def __init__(self, idx: V2[int]) -> None:
        HexChunk.set_size()
        self._idx: V2[int] = idx
        self._topleft: V2[int] = self.calc_topleft()
        self._hexes: list[list[Hex | None]] = [[None for _ in range(HEX_CHUNK_SIZE)] for _ in range(HEX_CHUNK_SIZE)]

    def topleft(self) -> V2[int]:
        return self._topleft
    
    def reset_topleft(self) -> None:
        self._topleft = self.calc_topleft()

    def hexes(self) -> list[list[Hex | None]]:
        return self._hexes
    
    def calc_topleft(self) -> V2[int]:
        match Hex.orientation:
            case HexOrientation.FLAT:
                return V2(round(self._idx.x() * HexChunk.size.x() - Hex.dim.x() / 2 - self._idx.x() * Hex.dim.x() / 4),
                          round(self._idx.y() * HexChunk.size.y() - (self._idx.y() + 1) * Hex.dim.y() / 2))
            case HexOrientation.POINTY:
                return V2(round(self._idx.x() * HexChunk.size.x() - (self._idx.x() + 1) * Hex.dim.x() / 2), 
                        round(self._idx.y() * HexChunk.size.y() - Hex.dim.y() / 2 - self._idx.y() * Hex.dim.y() / 4))

    def fill_chunk(self) -> None:
        for x in range(HEX_CHUNK_SIZE):
            for y in range(HEX_CHUNK_SIZE):
                of_x: int = HEX_CHUNK_SIZE * self._idx.x() + x
                of_y: int = HEX_CHUNK_SIZE * self._idx.y() + y
                ax: Ax = Ax.of_to_ax(V2(of_x, of_y))
                self._hexes[x][y] = Hex(ax.q(), ax.r())

    def get_hex(self, idx: V2[int]) -> Hex | None:
        return self._hexes[idx.x()][idx.y()]


class HexStore:

    @staticmethod
    def of_to_chunk_idx(of: V2[int]) -> V2[int]:
        return V2(floor(of.x() / HEX_CHUNK_SIZE), floor(of.y() / HEX_CHUNK_SIZE))

    def __init__(self) -> None:
        self._chunks: list[list[HexChunk]] = [[HexChunk(V2(x, y)) for y in range(HEX_NOF_CHUNKS.y())] for x in range (HEX_NOF_CHUNKS.x())]
        self._in_camera: set[HexChunk] = set()
        self._min_chunk_idx: V2[int] = V2(-1, -1)
        self._max_chunk_idx: V2[int] = V2(-1, -1)
        self._nof_chunks_in_camera: V2[int] = V2(-1, -1)

    def chunks(self) -> list[list[HexChunk]]:
        return self._chunks
    
    def in_camera(self) -> set[HexChunk]:
        return self._in_camera
    
    def min_chunk_idx(self) -> V2[int]:
        return self._min_chunk_idx
    
    def min_max_chunk_idx(self) -> V2[V2[int]]:
        return V2(self._min_chunk_idx, self._max_chunk_idx)
    
    def nof_chunks_in_camera(self) -> V2[int]:
        return self._nof_chunks_in_camera
        
    def fill_store(self) -> None:
        for i in range(len(self._chunks)):
            for j in range(len(self._chunks[i])):
                self._chunks[i][j].fill_chunk()

    def update_in_camera(self, min_max_of: V2[V2[int]]) -> None:
        min_chunk_idx: V2[int] = HexStore.of_to_chunk_idx(min_max_of[0])
        max_chunk_idx: V2[int] = HexStore.of_to_chunk_idx(min_max_of[1])
        if min_chunk_idx[0] != self._min_chunk_idx[0] or min_chunk_idx[1] != self._min_chunk_idx[1] \
            or max_chunk_idx[0] != self._max_chunk_idx[0] or max_chunk_idx[1] != self._max_chunk_idx[1]:
            self._min_chunk_idx = V2(max(0, min_chunk_idx.x()), max(0, min_chunk_idx.y()))
            self._max_chunk_idx = V2(min(HEX_CHUNK_SIZE, max_chunk_idx.x()), min(HEX_CHUNK_SIZE, max_chunk_idx.y()))
            nof_chunks_x: int = min(max_chunk_idx[0] + 1, HEX_NOF_CHUNKS[0]) - max(0, min_chunk_idx[0])
            nof_chunks_y: int = min(max_chunk_idx[1] + 1, HEX_NOF_CHUNKS[1]) - max(0, min_chunk_idx[1])
            self._nof_chunks_in_camera = V2(nof_chunks_x, nof_chunks_y)
            # TODO not full clear
            self._in_camera.clear()
            for x in range(max(0, min_chunk_idx[0]), min(max_chunk_idx[0] + 1, HEX_NOF_CHUNKS[0])):
                for y in range(max(0, min_chunk_idx[1]), min(max_chunk_idx[1] + 1, HEX_NOF_CHUNKS[1])):
                    self._in_camera.add(self._chunks[x][y])

    def get_hex_at_of(self, of: V2[int]) -> Hex | None:
        chunk_idx: V2[int] = HexStore.of_to_chunk_idx(of)
        if (chunk_idx[0] < 0 or chunk_idx[0] >= HEX_NOF_CHUNKS[0] or chunk_idx[1] < 0 or chunk_idx[1] >= HEX_NOF_CHUNKS[1]):
            return None
        chunk: HexChunk = self._chunks[chunk_idx[0]][chunk_idx[1]]
        hex_idx: V2[int] = HexChunk.of_to_hex_idx(of)
        hex: Hex | None = chunk.get_hex(hex_idx)
        return hex
    

class HexController(pk.model.Model):
    i: int = 0

    @staticmethod
    def add_hex_to_view(hex_controller: "HexController", hex: Hex) -> None:
        hex_controller._view.add(hex.element())

    @staticmethod
    def update_rect(_: "HexController", hex: Hex) -> None:
        # only update hex coordinates and element rect if: 
        # 1. element rect size is not equal to new hex dimensions (after zoom)
        # afterwards, coordinates are correctly adjusted and element rect has the right size until the next zoom takes place
        if (hex.element().rect.width != Hex.dim.x() or hex.element().rect.height != Hex.dim.y()):
            hex.reset_px()
            hex.update_element_rect()

    @staticmethod
    def update_image(_: "HexController", hex: Hex) -> None:
        # only update image if:
        # 1. image size is not equal to size of element (meaning a zoom has taken place)
        # 2. image is in frame (for the first time after a zoom)
        # afterwards, image size is equal to rect size until the next zoom takes place
        # image size is 1px larger because of rounding error gap mitigation (see update_element_image())
        if hex.element().rect.width + 1 != hex.element().image.get_width() or hex.element().rect.height + 1 != hex.element().image.get_height():
            hex.update_element_image()
            HexController.i += 1

    @staticmethod
    def update_topleft(_: "HexController", chunk: HexChunk) -> None:
        chunk.reset_topleft()
            
    def __init__(self, view: "HexView") -> None:
        pk.model.Model.__init__(self)

        Hex.orientation = HEX_ORIENTATION
        Hex.set_size(HEX_INIT_SIZE)

        HexSpriteStore.init_store()
        self._view: "HexView" = view
        self._store: HexStore = HexStore()

        self.apply_to_all_hex_in_store(HexController.add_hex_to_view)
        
    def store(self):
        return self._store

    def handle_event(self, event: pg.event.Event) -> None:
        pk.events.EventListener.handle_event(self, event)
        match event.type:
            case pg.MOUSEWHEEL:
                self.on_mouse_wheel(event)
            case _: pass

    def on_mouse_wheel(self, _: pg.event.Event) -> None:
        pass

    def update(self, dt: float) -> None:
        pk.model.Model.update(self, dt)
        if self._view.request_in_camera:
            self._store.update_in_camera(self._view.get_min_max_of())
            self.apply_to_all_hex_in_camera(HexController.update_rect)
            self.apply_to_all_hex_in_camera(HexController.update_image)
            self.apply_to_all_chunk_in_camera(HexController.update_topleft)

            pk.debug.debug["min chunk idx"] = self._store.min_chunk_idx()
            topleft: V2[int] = self._store.chunks()[self._store.min_chunk_idx().x()][self._store.min_chunk_idx().y()].topleft()
            self._view.update_chunk_surface(topleft, self._store.nof_chunks_in_camera(), self._store.in_camera())
            self._view.request_in_camera = False
        
    def apply_to_all_hex_in_store(self, f: Callable[["HexController", Hex], None]) -> None:
        for x in range(HEX_NOF_CHUNKS.x()):
            for y in range(HEX_NOF_CHUNKS.y()):
                self.apply_to_all_hex_in_chunk(self._store.chunks()[x][y], f)

    def apply_to_all_hex_in_camera(self, f: Callable[["HexController", Hex], None]) -> None:
        for chunk in self._store.in_camera():
            self.apply_to_all_hex_in_chunk(chunk, f)

    def apply_to_all_hex_in_chunk(self, chunk: HexChunk, f: Callable[["HexController", Hex], None]) -> None:
        for x in range(HEX_CHUNK_SIZE):
            for y in range(HEX_CHUNK_SIZE):
                hex: Hex | None = chunk.get_hex(V2(x, y))
                if isinstance(hex, Hex): f(self, hex)

    def apply_to_all_chunk_in_camera(self, f: Callable[["HexController", HexChunk], None]) -> None:
        for chunk in self._store.in_camera():
            f(self, chunk)

    def get_hex_at_px(self, pos: V2[int], offset: V2[int] = V2(0, 0)) -> Hex | None:
        ax: Ax = Ax.px_to_ax_offset(pos, offset)
        of: V2[int] = Ax.ax_to_of(ax)
        
        return self._store.get_hex_at_of(of)