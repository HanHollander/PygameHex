from __future__ import annotations
from typing import Protocol, TypeVar, Generic, Any


LMB: int = 0
MMB: int = 1
RMB: int = 2
MMB_UP: int = 4
MMB_DOWN: int = 5


_T = TypeVar('_T')
class Numeric(Protocol):
    def __add__(self: _T, __other: _T) -> _T: ...
    def __sub__(self: _T, __other: _T) -> _T: ...
    def __mul__(self: _T, __other: _T) -> _T: ...
    def __truediv__(self: _T, __other: _T) -> Any: ...
    def __floordiv__(self: _T, __other: _T) -> _T: ...
    def __mod__(self: _T, __other: _T) -> _T: ...

T = TypeVar("T", bound=Numeric)

class V2(Generic[T]):

    def __init__(self, a: T, b: T) -> None:
        self._t: tuple[T, T] = (a, b)

    def get(self) -> tuple[T, T]:
        return self._t
    
    def a(self) -> T:
        return self._t[0]
    def b(self) -> T:
        return self._t[1]
    
    def i(self) -> T:
        return self._t[0]
    def j(self) -> T:
        return self._t[1]
    
    def q(self) -> T:
        return self._t[0]
    def r(self) -> T:
        return self._t[1]
    
    def x(self) -> T:
        return self._t[0]
    def y(self) -> T:
        return self._t[1]

    def __add__(self, other: "V2[T]") -> "V2[T]":
        return V2(self._t[0] + other._t[0], self._t[1] + other._t[1])
    
    def __sub__(self, other: "V2[T]") -> "V2[T]":
        return V2(self._t[0] - other._t[0], self._t[1] - other._t[1])

    def __mul__(self, other: "V2[T]") -> "V2[T]":
        return V2(self._t[0] * other._t[0], self._t[1] * other._t[1])
    
    def __truediv__(self, other: "V2[T]") -> "V2[float]":
        return V2(self._t[0] / other._t[0], self._t[1] / other._t[1])
    
    def __floordiv__(self, other: "V2[T]") -> "V2[T]":
        return V2(self._t[0] // other._t[0], self._t[1] // other._t[1])
    
    def __mod__(self, other: "V2[T]") -> "V2[T]":
        return V2(self._t[0] % other._t[0], self._t[1] % other._t[1])
    
    def __getitem__(self, idx: int) -> T:
        assert idx == 0 or idx == 1, "idx out of range [0, 1]"
        return self._t[idx]

    def __str__(self) -> str:
        return "(" + str(self._t[0]) + ", " + str(self._t[1]) + ")"
    

class V3(Generic[T]):

    def __init__(self, a: T, b: T, c: T) -> None:
        self._t: tuple[T, T, T] = (a, b, c)

    def t(self) -> tuple[T, T, T]:
        return self._t
    
    def a(self) -> T:
        return self._t[0]
    def b(self) -> T:
        return self._t[1]
    def c(self) -> T:
        return self._t[2]
    
    def i(self) -> T:
        return self._t[0]
    def j(self) -> T:
        return self._t[1]
    def k(self) -> T:
        return self._t[2]
    
    def q(self) -> T:
        return self._t[0]
    def r(self) -> T:
        return self._t[1]
    def s(self) -> T:
        return self._t[2]
    
    def x(self) -> T:
        return self._t[0]
    def y(self) -> T:
        return self._t[1]
    def z(self) -> T:
        return self._t[2]

    def __add__(self, other: "V3[T]") -> "V3[T]":
        return V3(self._t[0] + other._t[0], self._t[1] + other._t[1], self._t[2] + other._t[2])
    
    def __sub__(self, other: "V3[T]") -> "V3[T]":
        return V3(self._t[0] - other._t[0], self._t[1] - other._t[1], self._t[2] - other._t[2])

    def __mul__(self, other: "V3[T]") -> "V3[T]":
        return V3(self._t[0] * other._t[0], self._t[1] * other._t[1], self._t[2] * other._t[2])
    
    def __truediv__(self, other: "V3[T]") -> "V3[float]":
        return V3(self._t[0] / other._t[0], self._t[1] / other._t[1], self._t[2] / other._t[2])
    
    def __floordiv__(self, other: "V3[T]") -> "V3[T]":
        return V3(self._t[0] // other._t[0], self._t[1] // other._t[1], self._t[2] // other._t[2])
    
    def __mod__(self, other: "V3[T]") -> "V3[T]":
        return V3(self._t[0] % other._t[0], self._t[1] % other._t[1], self._t[2] % other._t[2])
    
    def __getitem__(self, idx: int) -> T:
        assert idx == 0 or idx == 1 or idx == 2, "idx out of range [0, 2]"
        return self._t[idx]

    def __str__(self) -> str:
        return "(" + str(self._t[0]) + ", " + str(self._t[1]) + ", " + str(self._t[2]) + ")"