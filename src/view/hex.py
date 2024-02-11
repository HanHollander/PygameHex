from typing import Any
from model.hex import Ax, Hex, HexChunk, HexChunkSet, HexController, HexSpriteElement, HexStore
import pygame as pg
from pygame.event import Event
import pynkie as pk

from util import RMB, V2
from config import CHUNKS_PER_FRAME, DRAG_MOVE_FACTOR, HEX_CHUNK_SIZE, HEX_MAX_SIZE, HEX_MIN_SIZE, HEX_NOF_CHUNKS, ZOOM_STEP_FACTOR


class HexView(pk.view.ScaledView):

    def __init__(self, viewport: pk.view.Viewport) -> None:
        pk.view.ScaledView.__init__(self, viewport)
        # mouse related
        self.mouse_pos: V2[int] = V2(0, 0)
        self.mouse_down: tuple[bool, bool, bool] = (False, False, False)
        # draw related
        self.request_position_update = True
        self.request_chunk_surface_update = True
        self.request_sprite_store_update = False
        self.request_draw_chunk = False
        self.min_max_chunk_idx: V2[V2[int]] = V2(V2(0, 0), V2(0, 0))
        self.chunk_surface: pg.Surface = pg.Surface((0, 0))
        self.chunk_surface_topleft: V2[int] = V2(0, 0)
        self.chunk_surface_offset: V2[int] = V2(0, 0)
        self.chunks_to_be_drawn: set[HexChunk] = set()

    # utility

    def move_viewport(self, diff: V2[int | float]) -> None:
        self.viewport.camera.x = self.viewport.camera.x + round(diff[0])
        self.viewport.camera.y = self.viewport.camera.y + round(diff[1])
        self.chunk_surface_offset = self.chunk_surface_topleft - V2(self.viewport.camera.x, self.viewport.camera.y)

    def get_mouse_pos_offset(self)-> V2[int]:
        return V2(*pg.mouse.get_pos()) + V2(*self.viewport.camera.topleft)
    
    def get_min_max_of(self)-> V2[V2[int]]:
        min_of: V2[int] = Ax.ax_to_of(Ax.px_to_ax(V2(*self.viewport.camera.topleft)))
        max_of: V2[int] = Ax.ax_to_of(Ax.px_to_ax(V2(*self.viewport.camera.bottomright)))
        return V2(min_of, max_of)
    
    def get_min_max_chunk_idx(self, min_max_of: V2[V2[int]]) -> V2[V2[int]]:
        min_chunk_idx: V2[int] = HexStore.of_to_chunk_idx(min_max_of[0])
        min_chunk_idx_bounded: V2[int] = V2(min(HEX_NOF_CHUNKS[0], max(0, min_chunk_idx[0])), min(HEX_NOF_CHUNKS[1], max(0, min_chunk_idx[1])))
        max_chunk_idx: V2[int] = HexStore.of_to_chunk_idx(min_max_of[1])
        max_chunk_idx_bounded: V2[int] = V2(min(HEX_NOF_CHUNKS[0], max(0, max_chunk_idx[0])), min(HEX_NOF_CHUNKS[1], max(0, max_chunk_idx[1])))
        return V2(min_chunk_idx_bounded, max_chunk_idx_bounded)
    
    # def get_chunk_surface_size(self, nof_chunks: V2[int]) -> V2[int]:
    #     match Hex.orientation:
    #         case HexOrientation.FLAT:
    #             surface_width: int = round(nof_chunks.x() * HexChunk.size.x() 
    #                                        - (nof_chunks.x() - 1) * Hex.dim.x() / 4 
    #                                        + 2 * CHUNK_SURFACE_PADDING * Hex.dim.x())
    #             surface_height: int = round(nof_chunks.y() * HexChunk.size.y() 
    #                                        - (nof_chunks.y() - 1) * Hex.dim.y() / 2 
    #                                        + 2 * CHUNK_SURFACE_PADDING * Hex.dim.y())
    #         case HexOrientation.POINTY:
    #             surface_width: int = round(nof_chunks.x() * HexChunk.size.x()   # TODO something still wrong with this calculation... Rounding?
    #                                        - (nof_chunks.x() - 1) * Hex.dim.x() / 2 
    #                                        + 2 * CHUNK_SURFACE_PADDING * Hex.dim.x())
    #             surface_height: int = round(nof_chunks.y() * HexChunk.size.y() 
    #                                         - (nof_chunks.y() - 1) * Hex.dim.y() / 4 
    #                                         + 2 * CHUNK_SURFACE_PADDING * Hex.dim.y())
    #     return V2(surface_width, surface_height)
    
    def determine_request_chunk_surface_update(self) -> None:
        new_min_max_of: V2[V2[int]] = self.get_min_max_of()
        new_min_max_chunk_idx: V2[V2[int]] = self.get_min_max_chunk_idx(new_min_max_of)
        if (self.min_max_chunk_idx != new_min_max_chunk_idx):
            self.request_chunk_surface_update = True

    # drawing

    def update_chunk_surface(self, in_camera: HexChunkSet, topleft: V2[int], bottomright: V2[int]) -> None:
        surface_size: V2[int] = bottomright - topleft
        if self.request_sprite_store_update:  # zoom, add all chunks
            self.chunks_to_be_drawn.clear()
            for chunk in in_camera.chunks():
                self.chunks_to_be_drawn.add(chunk)
            self.chunk_surface = pg.Surface(surface_size.get())
        else:  # pan, only add new chunks
            self.chunks_to_be_drawn = self.chunks_to_be_drawn.union({c for c in in_camera.chunks() if c.x() < self.min_max_chunk_idx[0].x()})
            self.chunks_to_be_drawn = self.chunks_to_be_drawn.union({c for c in in_camera.chunks() if c.y() < self.min_max_chunk_idx[0].y()})
            self.chunks_to_be_drawn = self.chunks_to_be_drawn.union({c for c in in_camera.chunks() if c.x() > self.min_max_chunk_idx[1].x()})
            self.chunks_to_be_drawn = self.chunks_to_be_drawn.union({c for c in in_camera.chunks() if c.y() > self.min_max_chunk_idx[1].y()})
            old_chunk_surface: pg.Surface = self.chunk_surface.copy()
            self.chunk_surface = pg.Surface(surface_size.get())
            topleft_diff: V2[int] = self.chunk_surface_topleft - topleft
            self.chunk_surface.blit(old_chunk_surface, topleft_diff.get())

        self.min_max_chunk_idx = V2(in_camera.min_chunk_idx(), in_camera.max_chunk_idx())
        self.chunk_surface_topleft = topleft
        self.chunk_surface_offset = topleft - V2(self.viewport.camera.x, self.viewport.camera.y)
        self.request_draw_chunk = True

    def add_single_chunk_to_surface(self) -> None:
        chunk_nr: int = 0
        while len(self.chunks_to_be_drawn) > 0 and chunk_nr < CHUNKS_PER_FRAME:
            chunk: HexChunk = self.chunks_to_be_drawn.pop()
            for x in range (HEX_CHUNK_SIZE):
                for y in range(HEX_CHUNK_SIZE):
                    hex: Hex | None = chunk.hexes()[x][y]
                    if isinstance(hex, Hex):
                        element: HexSpriteElement = hex.element()
                        target_rect = pg.Rect(element.rect.x - self.chunk_surface_topleft.x(), 
                                              element.rect.y - self.chunk_surface_topleft.y(),  
                                              element.rect.width, element.rect.height)
                        self.chunk_surface.blit(element.image, target_rect, None, 0)
            chunk_nr += 1
        self.request_draw_chunk: bool = len(self.chunks_to_be_drawn) > 0
            


    # override ScaledView.draw
    def draw(self, surface: pg.Surface, *_: Any) -> list[pg.Rect]:
        w: int = self.view_surface.get_width()
        h: int = self.view_surface.get_height()
        cs_x: int = self.chunk_surface_offset.x()
        cs_y: int = self.chunk_surface_offset.y()
        self.view_surface.blit(self.background, pg.Rect(0, 0, w, h), None, 0)
        self.view_surface.blit(self.chunk_surface, pg.Rect(0, 0, w, h), pg.Rect(-cs_x, -cs_y , w, h), 0)
        surface.blit(self.view_surface, (0, 0))
                        
        return []

    # event handling

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
            self.determine_request_chunk_surface_update()
            self.request_position_update = True
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

        # if zoom happened
        if new_size != Hex.size:
            Hex.set_size(new_size)
            HexChunk.set_size()
            mouse_px: V2[int] = V2(*pg.mouse.get_pos()) + V2(*self.viewport.camera.topleft)
            diff_px: V2[float] = V2(mouse_px[0] * scale - mouse_px[0], mouse_px[1] * scale - mouse_px[1])
            self.move_viewport(diff_px)
            self.request_chunk_surface_update = True
            self.request_sprite_store_update = True

    
            


