from collections.abc import Iterable
import copy
from enum import Enum
from math import floor
from random import random

import pygame as pg

import graphics
import util
# from model import *

from typing import TYPE_CHECKING, Any, Callable
if TYPE_CHECKING:
    from model import Game

# element

class Element:

    idgen = 0

    def __init__(self, pos: tuple[int, int]=(0, 0), size: tuple[int, int]=(0, 0)):
        self.id = Element.idgen
        Element.idgen += 1
        self.rect = pg.Rect(pos, size)

    # abstract
    def update(self, elements: list["Element"], game: "Game"):
        msg = "update() called on Element with id=" + str(self.id) + ", this is not allowed."
        raise Exception(msg)
        
    # abstract
    def draw(self, screen: pg.Surface):
        msg = "draw() called on Element with id=" + str(self.id) + ", this is not allowed."
        raise Exception(msg)
        
    def __str__(self):
        return type(self).__name__ + "[id=" + str(self.id) + "]"


###################
# pygame wrappers #
###################

# sprite

class SpriteElement(Element, pg.sprite.Sprite):

    def __init__(self, 
                 pos: tuple[int, int], 
                 img: pg.Surface):
        Element.__init__(self, pos, img.get_size())
        pg.sprite.Sprite.__init__(self)
        self.image = img.copy()

    # override
    def update(self, elements: "ElementList", game: "Game"):
        pg.sprite.Sprite.update(self)

    # override
    def draw(self, screen: pg.Surface):
        raise Exception("draw() called on SpriteElement, this is not allowed. \
            Use GroupElement to contain the sprites!")


# group

class GroupType(Enum):
    MISC = 1
    BUTTONS = 2
    CURSOR = 3


class GroupElement(Element, pg.sprite.Group):
    
    def __init__(self, 
                 background: pg.Surface, 
                 type: GroupType):
        Element.__init__(self)
        pg.sprite.Group.__init__(self)
        self.background = background
        self.type = type

    # override
    def update(self, elements: "ElementList", game: "Game"):
        for sprite in self.sprites():
            sprite.update(elements, game)

    # override
    def draw(self, screen: pg.Surface):
        pg.sprite.Group.clear(self, screen, self.background)
        pg.sprite.Group.draw(self, screen)

    def __str__(self):
        return str(self.type) + ": " + pg.sprite.Group.__str__(self)


class ElementList(list[Element]):

    def __init__(self) -> None:
        super(ElementList, self).__init__()
        self.groups = {}

    def append(self, elem: Any) -> None:
        super().append(elem)
        if isinstance(elem, GroupElement):
            self.groups[elem.type] = elem
    
    def get_group(self, group_type: GroupType) -> GroupElement:
        return self.groups[group_type]


# surface

class SurfaceElement(Element, pg.Surface):

    def __init__(self,
                  pos: tuple[int, int], 
                  size: tuple[int, int]):
        Element.__init__(self, pos, size)
        pg.Surface.__init__(self, size)

    # override
    def update(self, elements: ElementList, game: "Game"):
        pass

    # override
    def draw(self, screen: pg.Surface):
        screen.blit(self, (self.rect.x, self.rect.y))


###################
# custom elements #
###################

# buttons

class Button():
    
    def __init__(self, 
                 action: Callable[[ElementList, "Game"], None]):
        self.action = action
        self.is_down = False

    def on_mouse_down(self, event: pg.event.Event, game: "Game"):
        if self.rect.collidepoint(event.pos):
            self.is_down = True

    def on_mouse_up(self, event: pg.event.Event, elements: ElementList, game: "Game"):
        # do action
        if self.is_down == True:
            self.action(event, elements, game)
            self.is_down = False


class SpriteButton(Button, SpriteElement):

    def __init__(self, 
                 pos: tuple[int, int], 
                 action: Callable[[ElementList, "Game"], None], 
                 img: pg.Surface, 
                 img_pressed: pg.Surface, 
                 img_hover: pg.Surface):
        Button.__init__(self, action)
        SpriteElement.__init__(self, pos, img)
        self.img = img
        self.img_pressed = img_pressed
        self.img_hover = img_hover

    # override
    def update(self, elements: ElementList, game: "Game"):
        SpriteElement.update(self, elements, game)

    # override
    def draw(self, screen: pg.Surface):
        SpriteElement.draw(self, screen)

    # override
    def on_mouse_down(self, event: pg.event.Event, game: "Game"):
        Button.on_mouse_down(self, event, game)
        if self.is_down:
            self.image = self.img_pressed.copy()
    
    # override
    def on_mouse_up(self, event: pg.event.Event, elements: ElementList, game: "Game"):
        Button.on_mouse_up(self, event, elements, game)
        self.image = self.img.copy()

    # override
    def __str__(self):
        return Element.__str__(self)


# cursor

class CursorElement(SpriteElement):

    def __init__(self, pos: tuple[int, int], img: pg.Surface):
        SpriteElement.__init__(self, pos, img)

    # override
    def update(self, elements: ElementList, game: "Game"):
        SpriteElement.update(self, elements, game)

    # override
    def draw(self, screen: pg.Surface):
        SpriteElement.draw(self, screen)

    def on_mouse_motion(self, event: pg.event.Event):
        self.rect = pg.Rect(event.pos, (self.rect.w, self.rect.h))


# physics sprite

class PhysicsElement(SpriteElement):

    def __init__(self, 
                 pos: tuple[int, int], 
                 img: pg.Surface):
        SpriteElement.__init__(self, pos, img)
        self.mask = pg.mask.from_surface(self.image)
        self.old_rect = self.rect.copy()

    # override
    def update(self, elements: ElementList, game: "Game"):
        SpriteElement.update(self, elements, game)

    # override
    def draw(self, screen: pg.Surface):
        SpriteElement.draw(self, screen)
