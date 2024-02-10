from typing import Any
from model.hex import Ax, Hex, HexChunk
import pygame as pg
from pygame.event import Event
import pynkie as pk

from util import RMB, V2
from config import DRAG_MOVE_FACTOR, HEX_CHUNK_SIZE, HEX_MAX_SIZE, HEX_MIN_SIZE, ZOOM_STEP_FACTOR


class HexView(pk.view.ScaledView):

    def __init__(self, viewport: pk.view.Viewport) -> None:
        pk.view.ScaledView.__init__(self, viewport)
        self.mouse_pos: V2[int] = V2(0, 0)
        self.mouse_down: tuple[bool, bool, bool] = (False, False, False)

        self.request_in_camera = True
        self.min_max_of: V2[V2[int]] = V2(V2(0, 0), V2(0, 0))
        self.chunk_surface: pg.Surface = pg.Surface((0, 0))
        self.chunk_surface_topleft: V2[int] = V2(0, 0)
        self.chunk_surface_offset: V2[int] = V2(0, 0)

    def move_viewport(self, diff: V2[int | float]) -> None:
        self.viewport.camera.x = self.viewport.camera.x + round(diff[0])
        self.viewport.camera.y = self.viewport.camera.y + round(diff[1])
        
        pk.debug.debug["chunk_surface"] = (self.chunk_surface_topleft, self.chunk_surface_offset)
        self.chunk_surface_offset = self.chunk_surface_topleft - V2(self.viewport.camera.x, self.viewport.camera.y)

    def get_mouse_pos_offset(self)-> V2[int]:
        return V2(*pg.mouse.get_pos()) + V2(*self.viewport.camera.topleft)
    
    def get_min_max_of(self)-> V2[V2[int]]:
        min_of: V2[int] = Ax.ax_to_of(Ax.px_to_ax(V2(*self.viewport.camera.topleft)))
        max_of: V2[int] = Ax.ax_to_of(Ax.px_to_ax(V2(*self.viewport.camera.bottomright)))
        return V2(min_of, max_of)
    
    def update_chunk_surface(self, topleft: V2[int], nof_chunks: V2[int], in_camera: set[HexChunk]) -> None:
        surface_size: V2[int] = V2(nof_chunks.x() * HexChunk.size.x(), nof_chunks.y() * HexChunk.size.y())
        self.chunk_surface = pg.Surface(surface_size.get())
        self.chunk_surface_topleft = topleft
        self.chunk_surface_offset = topleft - V2(self.viewport.camera.x, self.viewport.camera.y)
        pk.debug.debug["update surface"] = (surface_size, nof_chunks)
        pk.debug.debug["chunk_surface"] = (self.chunk_surface_topleft, self.chunk_surface_offset)
        for chunk in in_camera:
            for x in range (HEX_CHUNK_SIZE):
                for y in range(HEX_CHUNK_SIZE):
                    hex: Hex | None = chunk.hexes()[x][y]
                    if isinstance(hex, Hex):
                        spr: pk.elements.SpriteElement = hex.element()
                        target_rect = pg.Rect(spr.rect.x - self.chunk_surface_topleft.x(), spr.rect.y - self.chunk_surface_topleft.y(),
                                            spr.rect.width, spr.rect.height)
                        self.chunk_surface.blit(spr.image, target_rect, None, 0)
        # calc surface size
        # make surface
        # for every chunk:     
        #    for every hex:
        #        get px coordinates
        #        blit image on surface
        # also calculate chunk topleft and store in chunk surface offset
        pass


    # override ScaledView.draw
    def draw(self, surface: pg.Surface, *_: Any) -> list[pg.Rect]:
        w: int = self.view_surface.get_width()
        h: int = self.view_surface.get_height()
        cs_x: int = self.chunk_surface_offset.x()
        cs_y: int = self.chunk_surface_offset.y()
        self.view_surface.blit(self.background, pg.Rect(0, 0, w, h), None, 0)
        self.view_surface.blit(self.chunk_surface, pg.Rect(0, 0, w, h), pg.Rect(-cs_x, -cs_y , w, h), 0)

        # for chunk in self.in_camera:
        #     for x in range (HEX_CHUNK_SIZE):
        #         for y in range(HEX_CHUNK_SIZE):
        #             hex: Hex | None = chunk.hexes()[x][y]
        #             if isinstance(hex, Hex):
        #                 spr: pk.elements.SpriteElement = hex.element()
        #                 target_rect = pg.Rect(spr.rect.x - self.viewport.camera.x, spr.rect.y - self.viewport.camera.y,
        #                                     spr.rect.width, spr.rect.height)
        #                 self.spritedict[spr] = self.view_surface.blit(spr.image, target_rect, None, 0)
        surface.blit(self.view_surface, (0, 0))
                        

        return []







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
        pk.debug.debug["Mouse down"] = self.mouse_down
    
    def on_mouse_up(self, event: pg.event.Event) -> None:
        self.mouse_down = pg.mouse.get_pressed()
        pk.debug.debug["Mouse down"] = self.mouse_down

    def on_mouse_motion(self, event: pg.event.Event) -> None:
        new_mouse_pos: V2[int] = V2(*pg.mouse.get_pos())
        pk.debug.debug["Mouse pos (screen, real)"] = [new_mouse_pos, self.get_mouse_pos_offset()]
        if (self.mouse_down[RMB]):
            mouse_diff: V2[int] = self.mouse_pos - new_mouse_pos
            self.move_viewport(V2(DRAG_MOVE_FACTOR * mouse_diff[0], DRAG_MOVE_FACTOR * mouse_diff[1]))
            new_min_max_of: V2[V2[int]] = self.get_min_max_of()
            if (self.min_max_of != new_min_max_of):
                self.min_max_of = new_min_max_of
                self.request_in_camera = True
        self.mouse_pos = new_mouse_pos

    def on_mouse_wheel(self, event: pg.event.Event) -> None:
        scale: float = 1.
        new_size: int = Hex.size
        if event.y == -1:  # size down, zoom out
            new_size: int = max(HEX_MIN_SIZE, round(Hex.size * (1 / ZOOM_STEP_FACTOR)))
            scale = new_size / Hex.size
        elif event.y == 1:  # size up, zoom in
            new_size: int = min(HEX_MAX_SIZE, round(Hex.size * ZOOM_STEP_FACTOR))
            scale = new_size / Hex.size

        if scale != 1.:
            Hex.set_size(new_size)
            HexChunk.set_size()
            mouse_px: V2[int] = V2(*pg.mouse.get_pos()) + V2(*self.viewport.camera.topleft)
            diff_px: V2[float] = V2(mouse_px[0] * scale - mouse_px[0], mouse_px[1] * scale - mouse_px[1])
            self.move_viewport(diff_px)
            new_min_max_of: V2[V2[int]] = self.get_min_max_of()
            if (self.min_max_of != self.get_min_max_of()):
                self.min_max_of = new_min_max_of
                self.request_in_camera = True

    
            


