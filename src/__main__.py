import pygame as pg

pg.init()

import pynkie as pk
from config import MAX_FRAMERATE, SCREEN_WIDTH, SCREEN_HEIGHT, DEBUG_INFO, USE_DEFAULT_CURSOR

pynkie: pk.run.Pynkie = pk.init(SCREEN_WIDTH, SCREEN_HEIGHT)

from model.game import Game
from view.hex import HexView
from model.hex import HexController

init_scale: int = 1
camera_width: int = SCREEN_WIDTH
camera_height: int = SCREEN_HEIGHT
hex_view: HexView = HexView(pk.view.Viewport(
    pg.Rect(0, 0, camera_width, camera_height),
    pg.Vector2(camera_width * init_scale, camera_height * init_scale)
))

hex_controller: HexController = HexController(hex_view)
game = Game(hex_view, hex_controller)

pynkie.add_view(hex_view)
pynkie.add_model(game)
pynkie.add_model(hex_controller)
pynkie.add_event_listeners(pg.KEYDOWN, [game])
pynkie.add_event_listeners(pg.MOUSEBUTTONDOWN, [hex_view])
pynkie.add_event_listeners(pg.MOUSEBUTTONUP, [hex_view])
pynkie.add_event_listeners(pg.MOUSEMOTION, [game, hex_view])
pynkie.add_event_listeners(pg.MOUSEWHEEL, [hex_view, hex_controller])

pynkie.set_debug_info(DEBUG_INFO)
pynkie.set_max_framerate(MAX_FRAMERATE)
pynkie.set_use_default_cursor(USE_DEFAULT_CURSOR)
                           
    

pynkie.run()