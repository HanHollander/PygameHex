import pygame as pg

pg.init()

from pynkie.run import Pynkie
from pynkie.view import ScaledView, Viewport

from config import SCREEN_WIDTH, SCREEN_HEIGHT, DEBUG_INFO
from model.game import Game

init_scale = 1
camera_width = SCREEN_WIDTH
camera_height = SCREEN_HEIGHT
game_view = ScaledView(Viewport(
    pg.Rect(0, 0, camera_width, camera_height),
    pg.Vector2(camera_width * init_scale, camera_height * init_scale)
))
game = Game(game_view)

event_listeners = {pg.KEYDOWN: [game]}

p = Pynkie(
           views=[game_view],
           models=[game],
           event_listenters=event_listeners,
           screen_width=game_view.viewport.screen_size.x,
           screen_height=game_view.viewport.screen_size.y,
           debug_info=DEBUG_INFO,
           use_default_cursor=True)
p.run()

