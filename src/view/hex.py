from typing import Any
from model.hex import AxialCoordinates, Hex, HexChunk
import pygame as pg
from pygame.event import Event
import pynkie as pk

from util import RMB, add_tuple, f2, sub_tuple
from config import DRAG_MOVE_FACTOR, HEX_CHUNK_SIZE, HEX_MAX_SIZE, HEX_MIN_SIZE, ZOOM_STEP_FACTOR


class HexView(pk.view.ScaledView):

    def __init__(self, viewport: pk.view.Viewport) -> None:
        pk.view.ScaledView.__init__(self, viewport)
        self.mouse_pos: tuple[int, int] = (0, 0)
        self.mouse_down: tuple[bool, bool, bool] = (False, False, False)
        self.request_in_camera = True
        self.min_max_of: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0))
        self.in_camera: set[HexChunk] = set()

    def move_viewport(self, diff: tuple[int | float, int | float]) -> None:
        self.viewport.camera.x = self.viewport.camera.x + round(diff[0])
        self.viewport.camera.y = self.viewport.camera.y + round(diff[1])

    def get_mouse_pos_offset(self)-> tuple[int, int]:
        return f2(add_tuple(pg.mouse.get_pos(), self.viewport.camera.topleft))
    
    def get_min_max_of(self)-> tuple[tuple[int, int], tuple[int, int]]:
        min_of: tuple[int, int] = AxialCoordinates.ax_to_of(AxialCoordinates.px_to_ax(self.viewport.camera.topleft))
        max_of: tuple[int, int] = AxialCoordinates.ax_to_of(AxialCoordinates.px_to_ax(self.viewport.camera.bottomright))
        pk.debug.debug["min/max of"] = [min_of, max_of]
        return (min_of, max_of)

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
        pk.debug.debug["mouse pos"] = [new_mouse_pos, self.get_mouse_pos_offset()]
        if (self.mouse_down[RMB]):
            mouse_diff: tuple[int, int] = f2(sub_tuple(self.mouse_pos, new_mouse_pos))
            self.move_viewport((DRAG_MOVE_FACTOR * mouse_diff[0], DRAG_MOVE_FACTOR * mouse_diff[1]))
            new_min_max_of: tuple[tuple[int, int], tuple[int, int]] = self.get_min_max_of()
            if (self.min_max_of != new_min_max_of):
                self.min_max_of = new_min_max_of
                self.request_in_camera = True
        self.mouse_pos = new_mouse_pos

    def on_mouse_wheel(self, event: pg.event.Event) -> None:
        scale: float = 1.
        if event.y == -1:  # size down, zoom out
            new_size: int = max(HEX_MIN_SIZE, round(Hex.size * (1 / ZOOM_STEP_FACTOR)))
            scale = new_size / Hex.size
            if new_size != Hex.size: Hex.set_size(new_size)
        elif event.y == 1:  # size up, zoom in
            new_size: int = min(HEX_MAX_SIZE, round(Hex.size * ZOOM_STEP_FACTOR))
            scale = new_size / Hex.size
            if new_size != Hex.size: Hex.set_size(new_size)

        if scale != 1.:
            mouse_px: tuple[int, int] = f2(add_tuple(pg.mouse.get_pos(), self.viewport.camera.topleft))
            diff_px: tuple[float, float] = (mouse_px[0] * scale - mouse_px[0], mouse_px[1] * scale - mouse_px[1])
            pk.debug.debug["move viewport: mouse, diff, scale"] = [mouse_px, diff_px, scale]
            self.move_viewport(diff_px)
            new_min_max_of: tuple[tuple[int, int], tuple[int, int]] = self.get_min_max_of()
            if (self.min_max_of != self.get_min_max_of()):
                self.min_max_of = new_min_max_of
                self.request_in_camera = True

    # override ScaledView.draw
    def draw(self, surface: pg.Surface, *args: Any) -> list[pg.Rect]:
        self.center_camera()

        self.view_surface.blit(self.background, pg.Rect(0, 0, self.viewport.camera.width, self.viewport.camera.height), None, 0)

        n = 0
        for chunk in self.in_camera:
            for x in range (HEX_CHUNK_SIZE):
                for y in range(HEX_CHUNK_SIZE):
                    hex: Hex | None = chunk.hexes[x][y]
                    if isinstance(hex, Hex):
                        spr: pk.elements.SpriteElement = hex.element
                        target_rect = pg.Rect(spr.rect.x - self.viewport.camera.x, spr.rect.y - self.viewport.camera.y,
                                            spr.rect.width, spr.rect.height)
                        self.spritedict[spr] = self.view_surface.blit(spr.image, target_rect, None, 0)
                        n+=1

        print(n)

        # Scale the view surface to the dimensions of the screen and blit directly
        # pg.transform.scale(self.view_surface, self.viewport.screen_size, surface)
        surface.blit(self.view_surface, (0, 0))

        return []
            


