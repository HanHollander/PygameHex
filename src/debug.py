import pygame as pg

import graphics

debug = {}
font = graphics.f_DEBUG
font_size = graphics.c_f_DEBUG_SIZE
x = 0
y0 = 0

def fmt(v) -> str:
    if isinstance(v, float):
        return "{:.3f}".format(v)
    elif isinstance(v, tuple) or isinstance(v, list) or isinstance(v, pg.Vector2):
        return fmt_listlike(v)
    else:
        return str(v)
    
def fmt_listlike(l) -> str:
    f = "["
    if len(l) > 0:
        for v in l:
            f += fmt(v)
            f += ", "
        f = f[:-2]
    f += "]"
    return f

def display_debug(surface: pg.Surface) :
    i = 0
    for (key, value) in debug.items():
        surface.blit(font.render(key + ": " + fmt(value), False, graphics.c_WHITE), (x, y0 + i * font_size))
        i += 1