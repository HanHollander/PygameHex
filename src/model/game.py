import sys
import pygame as pg
import pynkie as pk

from model.hex import AxialCoordinates, Hex, HexController
from config import SCREEN_WIDTH, SCREEN_HEIGHT


class Game(pk.model.Model):

    def __init__(self, view: pk.view.ScaledView) -> None:
        self.keys_down: set[int] = set()
        self.hex_controller: HexController = HexController(view=view, size=200)
        self.hex_controller.fill_screen(SCREEN_WIDTH, SCREEN_HEIGHT)

    def update(self, dt: float):
        pass

    def handle_event(self, event: pg.event.Event) -> None:
        match event.type:
            case pg.KEYDOWN:
                self.on_key_down(event)
            case pg.KEYUP:
                self.on_key_up(event)
            case pg.MOUSEMOTION:
                self.on_mouse_motion(event)
            case _: pass
    

    def on_key_down(self, event: pg.event.Event):
        self.keys_down.add(event.key)
        if event.key == pg.K_q:
            pg.quit()
            sys.exit()
    
    def on_key_up(self, event: pg.event.Event):
        self.keys_down.remove(event.key)

    def on_mouse_motion(self, event: pg.event.Event):
        pos: tuple[int, int] = pg.mouse.get_pos()
        hex: Hex | None = self.hex_controller.get_hex_at_mouse_px(pos[0], pos[1])
        if hex: pk.debug.debug["hex idx (ax, of)"] = [hex.ax.c, AxialCoordinates.ax_to_of(hex.ax)]
        pk.debug.debug["mouse pos"] = pos