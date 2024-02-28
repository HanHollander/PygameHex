import sys
from config import Cfg
import pygame as pg
import pynkie as pk

from model.hex import Ax, Hex, HexChunk, HexController, HexStore
from util import V2
from view.hex import HexView


class Game(pk.model.Model):

    def __init__(self, hex_view: HexView, hex_controller: HexController) -> None:
        pk.model.Model.__init__(self)
        self.keys_down: set[int] = set()
        self.hex_view: HexView = hex_view
        self.hex_controller: HexController = hex_controller
        self.min_fps: int = Cfg.MAX_FRAMERATE
        self.max_fps: int = 0
        # self.hex_controller.apply_to_all_in_store(HexController.add_hex_to_view)

    def update(self, dt: float) -> None:
        if dt > 0:
            fps: int = round(1 / dt)
            if fps < self.min_fps: self.min_fps = fps
            if fps > self.max_fps: self.max_fps = fps
        pk.debug.debug["FPS min/max"] = (self.min_fps, self.max_fps)
        pk.debug.debug["Hex size"] = Hex.size
        pk.debug.debug["Hex dim (int, float)"] = [Hex.dim, Hex.dim_float]
        pk.debug.debug["Chunk size"] = HexChunk.size_px
        pk.debug.debug["Zoom overflow"] = HexChunk.size_overflow

    def handle_event(self, event: pg.event.Event) -> None:
        pk.model.Model.handle_event(self, event)
        match event.type:
            case pg.KEYDOWN:
                self.on_key_down(event)
            case pg.KEYUP:
                self.on_key_up(event)
            case pg.MOUSEMOTION:
                self.on_mouse_motion(event)
            case _: pass

    def on_key_down(self, event: pg.event.Event) -> None:
        self.keys_down.add(event.key)
        if event.key == pg.K_q:
            pg.quit()
            sys.exit()
        if event.key == pg.K_f:
            self.min_fps = Cfg.MAX_FRAMERATE
            self.max_fps = 0
        if event.key == pg.K_r:
            self.hex_controller.reset_map()
    
    def on_key_up(self, event: pg.event.Event) -> None:
        self.keys_down.remove(event.key)

    def on_mouse_motion(self, event: pg.event.Event) -> None:
        pos: V2[int] = V2(*pg.mouse.get_pos())
        offset: V2[int] = V2(*self.hex_view.viewport.camera.topleft)
        hex: Hex | None = self.hex_controller.get_hex_at_px(pos, offset)
        pk.debug.debug["Camera offset"] = offset
        if hex:
            pk.debug.debug["Hex indices (ax, of, px, px2)"] = \
                [hex.ax().c, Ax.ax_to_of(hex.ax()), Ax.ax_to_px(hex.ax()), Ax.ax_to_px(hex.ax()) + (Hex.dim // V2(2, 2))]
            chunk_idx: V2[int] = HexStore.of_to_chunk_idx(Ax.ax_to_of(hex.ax()))
            pk.debug.debug["Chunk indices (c, h, tl, br)"] = \
                [chunk_idx, HexChunk.of_to_hex_idx(Ax.ax_to_of(hex.ax())), 
                 self.hex_controller.store().chunks()[chunk_idx.x()][chunk_idx.y()].topleft(),
                 self.hex_controller.store().chunks()[chunk_idx.x()][chunk_idx.y()].bottomright()]
            pk.debug.debug["Hex terrain/colour"] = [hex.attr().terrain_type, hex.element().colour]
            pk.debug.debug["Hex altitude"] = [hex.attr().altitude, hex.attr().terrain_altitude]
            pk.debug.debug["Hex humidity"] = [hex.attr().humidity, hex.attr().terrain_humidity]
            pk.debug.debug["Hex temperature"] = [hex.attr().temperature, hex.attr().terrain_temperature]