import pygame as pg

pg.init()

import pynkie as pk
# import pynkie.view as view

from config import SCREEN_WIDTH, SCREEN_HEIGHT, DEBUG_INFO
from model.game import Game

init_scale = 1
camera_width = SCREEN_WIDTH
camera_height = SCREEN_HEIGHT
game_view: pk.view.ScaledView = pk.view.ScaledView(pk.view.Viewport(
    pg.Rect(0, 0, camera_width, camera_height),
    pg.Vector2(camera_width * init_scale, camera_height * init_scale)
))
game = Game(game_view)

event_listeners: dict[int, list[pk.events.EventListener]] = {pg.KEYDOWN: [game],
                   pg.MOUSEMOTION: [game]}

p: pk.run.Pynkie = pk.run.Pynkie(
           views=[game_view],
           models=[game],
           event_listenters=event_listeners,
           screen_width=game_view.viewport.screen_size.x,
           screen_height=game_view.viewport.screen_size.y,
           max_framerate=512,
           debug_info=DEBUG_INFO,
           use_default_cursor=True)
p.run()

