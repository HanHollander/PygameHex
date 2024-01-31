import pygame as pg

import config
import graphics
import actions
from elements import *


def setup_screen():
    size = (config.SCREEN_WIDTH * config.INIT_SCALE, config.SCREEN_HEIGHT * config.INIT_SCALE)
    flags = pg.HWSURFACE | pg.DOUBLEBUF
    display = pg.display.set_mode(size, flags, vsync=1)
    pg.display.set_caption("gmtk2023test")
    display.fill(graphics.c_BLACK)

    return display

