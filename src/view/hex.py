from typing import Any
from model.hex import AxialCoordinates, Hex
import pygame as pg
from pygame.event import Event
import pynkie as pk

from util import RMB, add_tuple, f2, sub_tuple
from config import DRAG_MOVE_FACTOR, MAX_SCALE, MIN_SCALE, ZOOM_STEP_SIZE, ZOOM_MOVE_FACTOR


class HexView(pk.view.ScaledView):

    def __init__(self, viewport: pk.view.Viewport) -> None:
        pk.view.ScaledView.__init__(self, viewport)
        self.mouse_pos: tuple[int, int] = (0, 0)
        self.mouse_down: tuple[bool, bool, bool] = (False, False, False)
        self.diff_total: tuple[int, int] = (0, 0)

    def move_viewport(self, diff: tuple[int | float, int | float]) -> None:
        self.diff_total = f2(add_tuple(self.diff_total, (round(diff[0]), round(diff[1]))))
        self.viewport.camera.x = self.viewport.camera.x + round(diff[0])
        self.viewport.camera.y = self.viewport.camera.y + round(diff[1])

    def handle_event(self, event: Event) -> None:
        pk.view.ScaledView.handle_event(self, event)
        match event.type:
            case pg.MOUSEBUTTONDOWN:
                self.on_mouse_down(event)
            case pg.MOUSEBUTTONUP:
                self.on_mouse_up(event)
            case pg.MOUSEMOTION:
                self.on_mouse_motion(event)
            case pg.MOUSEWHEEL:
                self.on_mouse_wheel(event)
            case _: pass
    

    def on_mouse_down(self, event: pg.event.Event) -> None:
        self.mouse_down = pg.mouse.get_pressed()
        pk.debug.debug["mouse down"] = self.mouse_down
    
    def on_mouse_up(self, event: pg.event.Event) -> None:
        self.mouse_down = pg.mouse.get_pressed()
        pk.debug.debug["mouse down"] = self.mouse_down

    def on_mouse_motion(self, event: pg.event.Event) -> None:
        new_mouse_pos: tuple[int, int] = pg.mouse.get_pos()
        pk.debug.debug["mouse pos"] = new_mouse_pos
        if (self.mouse_down[RMB]):
            mouse_diff: tuple[int, int] = f2(sub_tuple(self.mouse_pos, new_mouse_pos))
            self.move_viewport((DRAG_MOVE_FACTOR * mouse_diff[0], DRAG_MOVE_FACTOR * mouse_diff[1]))
        self.mouse_pos = new_mouse_pos

    def on_mouse_wheel(self, event: pg.event.Event) -> None:
        if event.y == -1:
            new_size: int = max(8, Hex.size - 8)
            if new_size != Hex.size: Hex.set_size(new_size)
        elif event.y == 1:
            new_size: int = min(256, Hex.size + 8)
            if new_size != Hex.size: Hex.set_size(new_size)

        # if event.y < 0:
        #     self.scale = min(MAX_SCALE, self.scale + ZOOM_STEP_SIZE)
        #     self.recalculate_window_dimensions(self.viewport.screen_size)
        # elif event.y > 0:
        #     self.scale = max(MIN_SCALE, self.scale - ZOOM_STEP_SIZE)
        #     self.recalculate_window_dimensions(self.viewport.screen_size)

        # if (event.y == 1 and Hex.size < HEX_MAX_SIZE):
        #     center: tuple[int, int] = (round(self.viewport.screen_size.x / 2), round(self.viewport.screen_size.y / 2))
        #     mouse_diff: tuple[int, ...] = sub_tuple(self.mouse_pos, center)
        #     diff: tuple[float, float] = (ZOOM_MOVE_FACTOR * mouse_diff[0] if mouse_diff[0] != 0 else 0., 
        #                                  ZOOM_MOVE_FACTOR * mouse_diff[1] if mouse_diff[1] != 0 else 0.)
        #     if diff[0] != 0 or diff[1] != 0:
        #         self.move_viewport((diff[0], diff[1]))

        

