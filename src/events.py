import pygame as pg

import actions
from elements import *
from model.game import Game, GUI

def handle_events(game: Game, ui_layer: GUI):
    for event in pg.event.get():

        # quit
        if event.type == pg.QUIT:
            actions.quit()

        # mouse click
        # if event.type == pg.MOUSEBUTTONDOWN:
        #     on_mouse_down(event, elements, game)
        # if event.type == pg.MOUSEBUTTONUP:
        #     on_mouse_up(event, elements, game)

        # mouse movement
        if event.type == pg.MOUSEMOTION:
            ui_layer.on_mouse_motion(event)

        # on key down
        if event.type == pg.KEYDOWN:
            game.on_key_down(event)

        # on key up
        if event.type == pg.KEYUP:
            game.on_key_up(event)

