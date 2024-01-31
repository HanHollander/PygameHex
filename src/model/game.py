import math
import pygame as pg
from pygame import Vector2

import graphics
import util
from elements import GroupElement, SpriteElement, CursorElement, PhysicsElement
import config
import actions
import debug
from view import GUIView, View
from model.hex import Hex


class Game:

    def __init__(self, view: View):
        self.keys_down: set[int] = set()
        self.hexes = []
        for i in range(0, 10):
            for j in range(0, 10):
                self.hexes.append(Hex(i, j, 50))
        for hex in self.hexes:
            view.add(hex.element)

    def update(self, dt: float):
        pass

    def on_key_down(self, event: pg.event.Event):
        self.keys_down.add(event.key)
        if event.key == pg.K_q:
            actions.quit()
    
    def on_key_up(self, event: pg.event.Event):
        self.keys_down.remove(event.key)


class GUI:

    def __init__(self, view: GUIView) -> None:
        self.cursor = Cursor()
        view.add(self.cursor.element)

    def on_mouse_motion(self, event: pg.event.Event):
        self.cursor.on_mouse_motion(event)


class Cursor:
    
    def __init__(self):
        self.element = CursorElement(
            pos=(20, 240 - graphics.img_cursor.get_size()[1] / 2),
            img=graphics.img_cursor
        )

    def on_mouse_motion(self, event: pg.event.Event):
        self.element.rect = pg.Rect(
            event.pos, (self.element.rect.w, self.element.rect.h))