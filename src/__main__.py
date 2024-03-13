# config
from config import cfg

cfg.init()

# yappi (profiler)
import yappi

if cfg.PROFILE: yappi.start()

import atexit

def exit_handler() -> None:
    if cfg.PROFILE:
        stats = yappi.get_func_stats()  # type: ignore
        stats.save("/tmp/tmp.prof", type="pstat")  # type: ignore

atexit.register(exit_handler)

# pygame
import pygame as pg

pg.init()

# pynkie
import pynkie as pk

pynkie: pk.run.Pynkie = pk.init(cfg.SCREEN_WIDTH, cfg.SCREEN_HEIGHT)

# game
from model.game import Game
from view.hex import HexView
from model.hex import HexController
from view.loading import display_message

display_message(pynkie.display, "> initialising view...")
init_scale: int = 1
camera_width: int = cfg.SCREEN_WIDTH
camera_height: int = cfg.SCREEN_HEIGHT
hex_view: HexView = HexView(pk.view.Viewport(
    pg.Rect(0, 0, camera_width, camera_height),
    pg.Vector2(camera_width * init_scale, camera_height * init_scale)
))

display_message(pynkie.display, "> initialising hex controller...")
hex_controller: HexController = HexController(hex_view, pynkie.display)

display_message(pynkie.display, "> initialising main game model...")
game = Game(hex_view, hex_controller, pynkie)

display_message(pynkie.display, "> adding views...")
pynkie.add_view(hex_view)

display_message(pynkie.display, "> adding models...")
pynkie.add_model(game)
pynkie.add_model(hex_controller)

display_message(pynkie.display, "> adding event listeners...")
pynkie.add_event_listeners(pg.KEYDOWN, [game])
pynkie.add_event_listeners(pg.MOUSEBUTTONDOWN, [hex_view])
pynkie.add_event_listeners(pg.MOUSEBUTTONUP, [hex_view])
pynkie.add_event_listeners(pg.MOUSEMOTION, [game, hex_view])
pynkie.add_event_listeners(pg.MOUSEWHEEL, [hex_view, hex_controller])

display_message(pynkie.display, "> configuring...")
pynkie.set_debug_info(cfg.DEBUG_INFO)
pynkie.set_max_framerate(cfg.MAX_FRAMERATE)
pynkie.set_use_default_cursor(cfg.USE_DEFAULT_CURSOR)


display_message(pynkie.display, "loading complete, press any key to continue")
pg.event.clear()
wait: bool = True
while wait:
    event: pg.event.Event = pg.event.wait()
    if event.type == pg.KEYDOWN or event.type == pg.MOUSEBUTTONDOWN:
        wait = False
                           
pynkie.run()

