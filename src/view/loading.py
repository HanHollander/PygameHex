
from config import cfg
import pygame as pg
import pynkie as pk

x: int = 0
y0: int = 0
i: int = 0


def display_message(surface: pg.Surface, message: str, in_place: bool = False)  -> None:
    global x, y0, i
    print("loading -", message)
    if (in_place):
        i -= 1
        black = pg.Surface((cfg.SCREEN_WIDTH, pk.debug.font_size))
        black.fill((0, 0, 0))
        surface.blit(black, (x, y0 + i * pk.debug.font_size))
    surface.blit(pk.debug.font.render(message, pk.debug.antialias, (255, 255, 255)), (x, y0 + i * pk.debug.font_size))
    pg.display.flip()
    i += 1