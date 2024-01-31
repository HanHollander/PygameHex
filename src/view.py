import pygame as pg

import config as c

class Viewport:

    def __init__(self) -> None:
        self.camera = pg.Rect(0, 0, c.SCREEN_WIDTH, c.SCREEN_HEIGHT)
        self.screen_size = pg.Vector2(c.SCREEN_WIDTH * c.INIT_SCALE, c.SCREEN_HEIGHT * c.INIT_SCALE)

class View(pg.sprite.Group):
    
    def __init__(self):
        pg.sprite.Group.__init__(self)
        self.update_viewport(Viewport())

    def update_viewport(self, new_viewport: Viewport) -> None:
        self.viewport = new_viewport
        self.background = pg.Surface(size=(self.viewport.camera.width, self.viewport.camera.height))
        self.background.fill((20, 20, 20))
        self.view_surface = pg.Surface(size=(self.viewport.camera.width, self.viewport.camera.height))


    # override
    def draw(self, screen: pg.Surface):
        self.view_surface.blit(self.background, pg.Rect(0, 0, self.viewport.camera.width, self.viewport.camera.height), None, 0)
        # - reposition camera
        #   - make it center
        #   - free cam mode
        #   - follow viking?
        # - Z order
        # - scale

        for spr in self.sprites():
            if spr.rect.colliderect(self.viewport.camera):
                target_rect = pg.Rect(spr.rect.x - self.viewport.camera.x, spr.rect.y - self.viewport.camera.y,
                                    spr.rect.width, spr.rect.height)
                self.spritedict[spr] = self.view_surface.blit(spr.image, target_rect, None, 0)

        # Scale the view surface to the dimensions of the screen and blit directly
        pg.transform.scale(self.view_surface, self.viewport.screen_size, screen)

    def __str__(self):
        return str(self.type) + ": " + pg.sprite.Group.__str__(self)

class GUIView(pg.sprite.Group):
    pass