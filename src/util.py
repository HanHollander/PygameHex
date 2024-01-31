import pygame as pg
import math

def get_middle(w: int, h: int, x: int, y: int) -> tuple[int, int]:
    return (x + w / 2, y + h / 2)

def get_middle_rect(rect: pg.Rect) -> tuple[int, int]:
    return get_middle(rect.w, rect.h, rect.x, rect.y)

def get_deltas_abs(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    return (abs(x1 - x2), abs(y1 - y2))

def get_deltas(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    return (x1 - x2, y1 - y2)

def get_top_left(w: int, h: int , x: int, y: int) -> tuple[int, int]:
    return (x - w / 2, y - h / 2)

def pow_v2(v: pg.Vector2, p: int) -> pg.Vector2:
    if p == 0:
        return pg.Vector2(1, 1)
    elif p == 1:
        return v
    elif p == 2:
        return v.elementwise() * v 
    else:
        return v.elementwise() * pow_v2(v, p-1)
    
def spow_v2(v: pg.Vector2, p: int) -> pg.Vector2:
    if p == 0:
        return pg.Vector2(1, 1)
    elif p == 1:
        return v
    elif p == 2:
        return v.elementwise() * abs_v2(v)
    else:
        return v.elementwise() * spow_v2(v, p-1)
    
def abs_v2(v: pg.Vector2) -> pg.Vector2:
    return pg.Vector2(abs(v.x), abs(v.y))

def mag_v2(v: pg.Vector2) -> int:
    return math.sqrt(math.pow(v.x, 2) + math.pow(v.y, 2))

def str_v2(v: pg.Vector2) -> str:
    return "[" + str(v.x) + ", " + str(v.y) + "]"

def str_v2_list(l: list[pg.Vector2]):
    str = "["
    for v in l:
        str += str_v2(v)
        str += ", "
    str = str[:-2]
    str += "]"
    return str