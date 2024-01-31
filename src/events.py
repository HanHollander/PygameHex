import pygame as pg

import actions
from elements import *
from model import Game, GUILayer

def handle_events(game: Game, ui_layer: GUILayer):
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


# # mouse down

# def on_mouse_down(event: pg.event.Event, elements: ElementList, game: Game):
#     for element in elements:
#         if event.button == 1:
#             if isinstance(element, Button):
#                 element.on_mouse_down(event, game)
#             if isinstance(element, GroupElement):
#                 on_mouse_down_group(event, element, elements, game)
                

# def on_mouse_down_group(event: pg.event.Event, group_element: GroupElement, elements: ElementList, game: Game):
#     for sprite in group_element.sprites():
#         if isinstance(sprite, Button):
#             sprite.on_mouse_down(event, game)


# # mouse up
   
# def on_mouse_up(event: pg.event.Event, elements: ElementList, game: Game):
#     for element in elements:
#         if event.button == 1:
#             if isinstance(element, Button):
#                 element.on_mouse_up(event, elements, game)
                    
# def on_mouse_up_group(event: pg.event.Event, group_element: GroupElement, elements: ElementList, game: Game):
#     for sprite in group_element.sprites():
#         if isinstance(sprite, Button):
#             sprite.on_mouse_up(event, elements, game)


# # mouse motion

# def on_mouse_motion(event: pg.event.Event, elements: ElementList, game: Game):
#     for element in elements:
#         if isinstance(element, GroupElement):
#             on_mouse_motion_group(event, element, elements, game)
                

# def on_mouse_motion_group(event: pg.event.Event, group_element: GroupElement, elements: ElementList, game: Game):
#     for sprite in group_element.sprites():
#         if isinstance(sprite, Cursor):
#             sprite.on_mouse_motion(event)


# # key down

# def on_key_down(event: pg.event.Event, elements: ElementList, game: Game):
#     game.on_key_down(event, elements)
#     for element in elements:
#         if isinstance(element, GroupElement):
#             for sprite in element.sprites():
#                 pass
