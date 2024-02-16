from typing import Any
from model.hex import Ax, Hex, HexChunk, HexChunkSet, HexController, HexSpriteElement, HexStore
import pygame as pg
from pygame.event import Event
import pynkie as pk

from util import RMB, V2
from config import CHUNKS_PER_FRAME, DRAG_MOVE_FACTOR, HEX_MAX_SIZE, HEX_MIN_SIZE, ZOOM_STEP_FACTOR


class HexViewFlags():

    def __init__(self) -> None:
        self.request_reset_chunks = True
        self.request_reset_chunk_update_status = True
        self.request_reset_scaled_store = True
        self.request_update_in_camera = True
        self.request_update_chunk_surface = True
        self.request_update_and_add_single_chunk_to_surface = True

        self.init = True
        self.pan_happened = False
        self.zoom_happened = False


class HexView(pk.view.ScaledView):

    def __init__(self, viewport: pk.view.Viewport) -> None:
        pk.view.ScaledView.__init__(self, viewport)
        self.flags: HexViewFlags = HexViewFlags()
        # mouse related
        self.mouse_pos: V2[int] = V2(0, 0)
        self.mouse_down: tuple[bool, bool, bool] = (False, False, False)
        # draw related
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
        # Add a Hex.dim extra to account for irregularity of chunk boundaries
        min_of: V2[int] = Ax.ax_to_of(Ax.px_to_ax(V2(*self.viewport.camera.topleft) - Hex.dim))
        max_of: V2[int] = Ax.ax_to_of(Ax.px_to_ax(V2(*self.viewport.camera.bottomright) + Hex.dim))
        return V2(min_of, max_of)
    
    def get_min_max_chunk_idx(self, min_max_of: V2[V2[int]]) -> V2[V2[int]]:
        min_chunk_idx: V2[int] = HexStore.of_to_chunk_idx(min_max_of[0])
        min_chunk_idx_bounded: V2[int] = V2(min(HexStore.nof_chunks[0] - 1, max(0, min_chunk_idx[0])), min(HexStore.nof_chunks[1] - 1, max(0, min_chunk_idx[1])))
        max_chunk_idx: V2[int] = HexStore.of_to_chunk_idx(min_max_of[1])
        max_chunk_idx_bounded: V2[int] = V2(min(HexStore.nof_chunks[0] - 1, max(0, max_chunk_idx[0])), min(HexStore.nof_chunks[1] - 1, max(0, max_chunk_idx[1])))
        return V2(min_chunk_idx_bounded, max_chunk_idx_bounded)
    
    def min_max_chunk_idx_will_change(self) -> bool:
        new_min_max_of: V2[V2[int]] = self.get_min_max_of()
        new_min_max_chunk_idx: V2[V2[int]] = self.get_min_max_chunk_idx(new_min_max_of)
        if (self.min_max_chunk_idx != new_min_max_chunk_idx):
            return True
        return False

    # drawing

    def update_chunk_surface(self, in_camera: HexChunkSet, topleft: V2[int], bottomright: V2[int]) -> None:
        surface_size: V2[int] = bottomright - topleft
        pk.debug.debug["surface_size"] = surface_size

        if self.flags.init:
            for chunk in in_camera.chunks():
                self.chunks_to_be_drawn.add(chunk)
            self.chunk_surface = pg.Surface(surface_size.get())
            self.flags.init = False
        elif self.flags.zoom_happened:
            self.chunks_to_be_drawn.clear()
            for chunk in in_camera.chunks():
                self.chunks_to_be_drawn.add(chunk)
            self.chunk_surface = pg.Surface(surface_size.get())
            self.flags.zoom_happened = False  # reset flag
        elif self.flags.pan_happened:  # pan, only add new chunks
            self.chunks_to_be_drawn = self.chunks_to_be_drawn.union({c for c in in_camera.chunks() if c.x() < self.min_max_chunk_idx[0].x()})
            self.chunks_to_be_drawn = self.chunks_to_be_drawn.union({c for c in in_camera.chunks() if c.y() < self.min_max_chunk_idx[0].y()})
            self.chunks_to_be_drawn = self.chunks_to_be_drawn.union({c for c in in_camera.chunks() if c.x() > self.min_max_chunk_idx[1].x()})
            self.chunks_to_be_drawn = self.chunks_to_be_drawn.union({c for c in in_camera.chunks() if c.y() > self.min_max_chunk_idx[1].y()})
            old_chunk_surface: pg.Surface = self.chunk_surface  # TODO when zoomed in below code very costly! (surface BIG)
            self.chunk_surface = pg.Surface(surface_size.get())
            topleft_diff: V2[int] = self.chunk_surface_topleft - topleft
            self.chunk_surface.blit(old_chunk_surface, topleft_diff.get())  # TODO until here
            self.flags.pan_happened = False

        self.min_max_chunk_idx = V2(in_camera.min_chunk_idx(), in_camera.max_chunk_idx())
        self.chunk_surface_topleft = topleft
        self.chunk_surface_offset = topleft - V2(self.viewport.camera.x, self.viewport.camera.y)
        self.flags.request_update_and_add_single_chunk_to_surface = True

    def update_and_add_single_chunk_to_surface(self) -> None:
        chunk_nr: int = 0
        while len(self.chunks_to_be_drawn) > 0 and chunk_nr < CHUNKS_PER_FRAME:
            chunk: HexChunk = self.chunks_to_be_drawn.pop()
            filled: bool = False
            updated: bool = False
            if not chunk.filled():
                chunk.fill()
                filled = True
                chunk.set_filled(True)
            if not chunk.updated(): 
                chunk.reset_hexes()
                chunk.reset_topleft()
                chunk.reset_bottomright()
                updated = True
                chunk.set_updated(True)
            print(HexController.i, "update_and_add_single_chunk_to_surface", chunk.idx(), "filled:" + str(filled), "updated:" + str(updated))
            for x in range (HexChunk.nof_hexes):
                for y in range(HexChunk.nof_hexes):
                    hex: Hex | None = chunk.hexes()[x][y]
                    if hex:
                        if not hex.updated():
                            hex.update_element_rect()
                            hex.update_element_image()
                            hex.set_updated(True)
                        element: HexSpriteElement = hex.element()
                        target_rect = pg.Rect(element.rect.x - self.chunk_surface_topleft.x(), 
                                              element.rect.y - self.chunk_surface_topleft.y(),  
                                              element.rect.width, element.rect.height)
                        self.chunk_surface.blit(element.image, target_rect, None, 0)
            chunk_nr += 1
        self.flags.request_update_and_add_single_chunk_to_surface = len(self.chunks_to_be_drawn) > 0
            

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
            if self.min_max_chunk_idx_will_change():  # only if min and/or max chunk indices will change next frame (as a result of a pan)
                self.flags.request_update_in_camera = True
                self.flags.request_update_chunk_surface = True
                self.flags.pan_happened = True
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
            if scale > 1.:  # chunks smaller (zoom in)
                if HexChunk.set_nof_hexes(HexChunk.nof_hexes // int(ZOOM_STEP_FACTOR)):
                    HexStore.set_nof_chunks(HexStore.nof_chunks * V2(int(ZOOM_STEP_FACTOR), int(ZOOM_STEP_FACTOR)))
                    self.flags.request_reset_chunks = True
            elif scale < 1.:  # chunks bigger (zoom out)
                if HexChunk.set_nof_hexes(HexChunk.nof_hexes * int(ZOOM_STEP_FACTOR)):
                    HexStore.set_nof_chunks(HexStore.nof_chunks // V2(int(ZOOM_STEP_FACTOR), int(ZOOM_STEP_FACTOR)))
                    self.flags.request_reset_chunks = True

            Hex.set_size(new_size)
            HexChunk.set_size_px()
            mouse_px: V2[int] = V2(*pg.mouse.get_pos()) + V2(*self.viewport.camera.topleft)
            diff_px: V2[float] = V2(mouse_px[0] * scale - mouse_px[0], mouse_px[1] * scale - mouse_px[1])
            self.move_viewport(diff_px)
            if not self.flags.request_reset_chunks: self.flags.request_reset_chunk_update_status = True
            self.flags.request_reset_scaled_store = True  # request for the sprites to be scaled
            # if self.min_max_chunk_idx_will_change():  # only if min and/or max chunk indices changed
            self.flags.request_update_in_camera = True
            self.flags.request_update_chunk_surface = True  # chunk surface always changes
            self.flags.zoom_happened = True

    
            


