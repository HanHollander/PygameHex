from typing import Protocol, TypeVar


LMB: int = 0
MMB: int = 1
RMB: int = 2
MMB_UP: int = 4
MMB_DOWN: int = 5


_T = TypeVar('_T')
class Numeric(Protocol):
    def __add__(self: _T, __other: _T) -> _T: ...
    def __sub__(self: _T, __other: _T) -> _T: ...


T = TypeVar("T", bound=Numeric)

def f2(t: tuple[T, ...]) -> tuple[T, T]:
    assert len(t) > 1, "Size of tuple not > 1"
    return (t[0], t[1])

def add_tuple(t1: tuple[T, ...], t2: tuple[T, ...]) -> tuple[T, ...]:
    return tuple(map(lambda a, b: a + b, t1, t2))

def sub_tuple(t1: tuple[T, ...], t2: tuple[T, ...]) -> tuple[T, ...]:
    return tuple(map(lambda a, b: a - b, t1, t2))