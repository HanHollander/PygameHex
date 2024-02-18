from typing import Any
import pygame as pg
import pynkie as pk

x: int = 0
y0: int = 0
i: int = 0


def display_message(surface: pg.Surface, message: str)  -> None:
    global x, y0, i
    print("loading -", message)
    surface.blit(pk.debug.font.render(message, pk.debug.antialias, (255, 255, 255)), (x, y0 + i * pk.debug.font_size))
    pg.display.flip()
    i += 1