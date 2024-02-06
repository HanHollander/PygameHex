import sys
import pygame as pg
import pynkie as pk

from model.hex import AxialCoordinates, Hex, HexController, HexStore
from view.hex import HexView


class Game(pk.model.Model):

    def __init__(self, hex_view: HexView, hex_controller: HexController) -> None:
        pk.model.Model.__init__(self)
        self.keys_down: set[int] = set()
        self.hex_view: HexView = hex_view
        self.hex_controller: HexController = hex_controller
        self.hex_controller.hex_store.fill_store()

    def update(self, dt: float) -> None:
        pk.debug.debug["Hex.size"] = Hex.size
        pk.debug.debug["Hex.dim"] = Hex.dim
        pk.debug.debug["Hex.spacing"] = Hex.spacing
        pk.debug.debug["HexStore.size"] = HexStore.size

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
    
    def on_key_up(self, event: pg.event.Event) -> None:
        self.keys_down.remove(event.key)

    def on_mouse_motion(self, event: pg.event.Event) -> None:
        pos: tuple[int, int] = pg.mouse.get_pos()
        offset: tuple[int, int] = self.hex_view.viewport.camera.topleft
        hex: Hex | None = self.hex_controller.get_hex_at_px(pos, offset)
        pk.debug.debug["camera offset"] = offset
        if hex: pk.debug.debug["hex idx (ax, of, px)"] = [hex.ax.c, 
                                                      AxialCoordinates.ax_to_of(hex.ax), 
                                                      AxialCoordinates.ax_to_px(hex.ax)]