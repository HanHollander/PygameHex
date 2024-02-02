import sys
import pygame as pg
import pynkie as pk

from model.hex import HexController
from config import SCREEN_WIDTH, SCREEN_HEIGHT


class Game:

    def __init__(self, view: pk.view.ScaledView):
        self.keys_down: set[int] = set()
        self.hex_controller: HexController = HexController(view=view, size=100)
        self.hex_controller.fill_screen(SCREEN_WIDTH, SCREEN_HEIGHT)

    def update(self, dt: float):
        pass

    def on_key_down(self, event: pg.event.Event):
        self.keys_down.add(event.key)
        if event.key == pg.K_q:
            pg.quit()
            sys.exit()
    
    def on_key_up(self, event: pg.event.Event):
        self.keys_down.remove(event.key)