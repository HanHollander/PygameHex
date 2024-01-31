import time

import pygame as pg

import config
import events
import setup
import debug
from model.game import *
from elements import *
from view import View, GUIView

def run():
    # setup mouse
    pg.mouse.set_visible(False)

    # setup screen and clock
    display = setup.setup_screen()
    clock = pg.time.Clock()

    # Background layers

    # setup game
    view = View()
    game = Game(view)

    # foreground layers
    gui_view = GUIView()
    gui = GUI(gui_view)

    # run main loop
    main_loop(display, clock, view, game, gui_view, gui)


def main_loop(display: pg.Surface, clock: pg.time.Clock, view: View, game: Game, gui_view: GUIView, gui_layer: GUI):
    prev_time = time.time()
    dt = 0  # delta time [s]
    while True:
        # handle events
        # update the state of elements based on events/player triggers
        events.handle_events(game, gui_layer)
        
        # update game
        game.update(dt)

        # draw elements
        view.draw(display)
        gui_view.draw(display)

        # display debug info
        if config.DEBUG_INFO: debug.display_debug(display)
        
        # update screen
        pg.display.flip()

        # tick
        clock.tick(config.MAX_FRAMERATE)

        # delta time
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if config.DEBUG_INFO:
            debug.debug["DT"] = dt
            debug.debug["FPS"] = int(round(1 / dt))