import pynkie as pk

from math import floor

from model.hex import Hex, AxialCoordinates


class HexStore(list[list[Hex]]):

    @staticmethod
    def set_size(size: tuple[int, int]) -> None:
        HexStore.size = (size[0] if size[0] % 2 == 1 else size[0] + 1, size[1] if size[1] % 2 == 1 else size[1] + 1)

    # static varables
    size: tuple[int, int]  # size of store, [#cols, #rows]. Always odd, hex with (q=0, r=0) is the middle hex

    def __init__(self, size: tuple[int, int]) -> None:
        HexStore.set_size(size)
        self.store: list[list[Hex | None]] = [[None for _ in range(HexStore.size[0])] for _ in range(HexStore.size[1])]

    # store negative coordinates on odd indices
    def fill_store(self) -> None:
        assert HexStore.size[0] % 2 == 1 and HexStore.size[1] % 2 == 1, "Store size not odd"
        half_size: tuple[int, int] = (floor(HexStore.size[0] / 2), floor(HexStore.size[1] / 2))
        for col in range(-half_size[0], half_size[0] + 1):
            for row in range(-half_size[1], half_size[1] + 1):
                ax: AxialCoordinates = AxialCoordinates.of_to_ax((col, row))
                col_act: int = col * 2 if col >= 0 else -col * 2 - 1
                row_act: int = row * 2 if row >= 0 else -row * 2 - 1
                self.store[row_act][col_act] = Hex(ax.q(), ax.r())


class HexController:

    def __init__(self, view: pk.view.ScaledView, hex_size: int, store_size: tuple[int, int]) -> None:
        self.view: pk.view.ScaledView = view
        self.hex_store: HexStore = HexStore(store_size)
        Hex.set_size(hex_size)
        self.hex_store.fill_store()

        pk.debug.debug["Hex.size"] = Hex.size
        pk.debug.debug["Hex.dim"] = Hex.dim
        pk.debug.debug["Hex.spacing"] = Hex.spacing
        pk.debug.debug["HexStore.size"] = HexStore.size

    def fill_screen(self) -> None:
        for hex_row in self.hex_store.store:
            for hex in hex_row:
                assert isinstance(hex, Hex), "Hex store contains empty element"
                self.view.add(hex.element)
    
    def get_hex_at_mouse_px(self, x: int, y: int) -> Hex | None:
        ax: AxialCoordinates = AxialCoordinates.px_to_ax((x, y))
        of: tuple[int, int] = AxialCoordinates.ax_to_of(ax)
        if of[0] < len(self.hex_store.store) and of[1] < len(self.hex_store.store[int(of[0])]):
            return self.hex_store.store[int(of[0])][int(of[1])]
        else:
            return None
