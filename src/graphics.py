from enum import Enum
import os

import pygame as pg
import config

# dimensions

d_UNIT_WIDTH = config.SCREEN_WIDTH / 16
d_UNIT_HEIGHT = config.SCREEN_HEIGHT / 9

d_GUI_WIDTH = config.SCREEN_WIDTH / 16 * 5  # 320 -> 100
d_GUI_HEIGHT = config.SCREEN_HEIGHT

d_MAP_CENTER = ((config.SCREEN_WIDTH - d_GUI_WIDTH) / 2 + d_GUI_WIDTH, config.SCREEN_HEIGHT / 2)

d_MAP_POS_X = d_GUI_WIDTH
d_MAP_POS_Y = 0
d_MAP_WIDTH = config.SCREEN_WIDTH - d_GUI_WIDTH
d_MAP_HEIGHT = config.SCREEN_HEIGHT
d_MAP_RECT = pg.Rect(d_MAP_POS_X, d_MAP_POS_Y, d_MAP_WIDTH, d_MAP_HEIGHT)


d_CIRCLE_RADIUS = 10

# colours

c_COLOUR_KEY = (255, 0, 255)  # ff00ff

c_RED = (204, 27, 27)  # cc1b1b
c_DARK_RED = (133, 17, 17)  # 851111

c_WHITE = (255, 255, 255)  # ffffff
c_BLACK = (0, 0, 0)  # 000000
c_DARK_GRAY = (50, 50, 50)  # 333333
c_GRAY = (82, 82, 82)  # 525252
c_LIGHT_GRAY = (148, 148, 148)  # 949494

# fonts

pg.font.init()

f_SMALL = pg.font.SysFont(name="Corbel", size=16, bold=True, italic=False)
c_f_DEBUG_SIZE = 16
f_DEBUG = pg.font.Font(os.path.join("font", "rainyhearts.ttf"), c_f_DEBUG_SIZE)

# cursor

img_cursor = pg.image.load("./img/cursor.png")

# viking

img_viking = pg.image.load("./img/viking.png")