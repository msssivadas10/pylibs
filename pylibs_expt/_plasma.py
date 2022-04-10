from collections import namedtuple
from itertools import product, repeat
from typing import Any, Iterable, Sequence, Type, Union
from scipy.interpolate import CubicSpline
import numpy as np
import numpy.random as rnd
try:
    import tree
    from _tables import LevelsTable, LinesTable
except Exception:
    from . import tree
    from . import _tables
    LevelsTable, LinesTable = _tables.LevelsTable, _tables.LinesTable


class Plasma:
    """
    A plasma object.
    """
    __slots__ = 'lines', 'comp', 'Te', 'Ne', 
    __name__  = ''

    def __init__(self, *args, **kwargs) -> None:
        self.lines  : LinesTable
        self.comp   : tree.Node
        self.Te     : float
        self.Ne     : float 
    
    def _comp(self) -> tuple:
        ...

    @property
    def components(self) -> tuple:
        """ Plasma components. """
        return self._comp()

    def setup(self, Te: float = ..., Ne: float = ...) -> None:
        """ Set the plasma conditions. """
        if Te is not ... :
            if not Te > 0.0:
                raise ValueError("temperature 'Te' must be positive")
            self.Te = Te
        if Ne is not ... : 
            if not Ne > 0.0:
                raise ValueError("electron number density 'Ne' must be positive")
            self.Ne = Ne

    def _getComposition(self) -> None:
        return NotImplemented
    
    def _getIntensity(self) -> None:
        return NotImplemented
            
# def plasma(name: str, comp: Iterable[ElementNode], lines: LinesTable = None, interpolate: bool = True, T: Any = None) -> Type[Plasma]:
#     """
#     Create a specific plasma class.
#     """
#     _lines = lines
#     if lines is None:
#         _lines = LinesTable(elem = [], s = [])
#     elif isinstance(lines, LinesTable):
#         raise TypeError("lines must be a 'LinesTable' object")

#     def _U(self, Te: float) -> float:
#         if not np.isscalar(Te):
#             raise TypeError("Te must be a scalar")
#         if self.levels is None:
#             return self.Utable(Te)
#         return self.levels.U(Te)

#     if interpolate:
#         if T is None:
#             T = np.linspace(1e-3, 5.0, 101)
#         elif np.isscalar(T):
#             raise TypeError("T must be an array of floats")

#     # create a tree from the components
#     ct, _slots = tree.Node(), []
#     for elem in comp:
#         if not isinstance(elem, ElementNode):
#             raise TypeError("each element in comp should be an 'ElementNode' object")
#         ct.addchild(elem, key = elem.key)

#         if lines is None:
#             if elem.lines is None:
#                 raise ValueError("missing lines data")
#             _lines.join(elem.lines)

#         _slots.append(elem.key)

#     def _init(self: Plasma, *args, **kwargs) -> None:
#         self.comp  = ct
#         self.lines = _lines
#         self.Te    = None
#         self.Ne    = None

#         # parse arguments
#         args    = dict(zip(self.__slots__, args))
#         missing = None
#         for __name in self.__slots__:
#             if __name in args.keys() and __name in kwargs.keys():
#                 raise TypeError(f"got multiple values for argument '{__name}'")
#             if not (__name in args.keys() or __name in kwargs.keys()):
#                 if missing is None:
#                     missing = __name
#         args = {**args, **kwargs}
#         if len(args) == len(self.__slots__) - 1:
#             x = 100.0 - sum(args.values())
#             if x < 0.0:
#                 raise ValueError("total concentration should be 1")
#             args[missing], missing = x, None
#         if missing is not None:
#             raise TypeError(f"missing value for argument '{missing}'")

#         if sum(args.values()) != 100.0:
#             raise ValueError("total concentration must be 1")
        
#         for key, value in args.items():
#             setattr(self, key, value)
#             self.comp.child(key).Nx = value

#     def _comp(self: Plasma) -> tuple:
#         return self.__slots__

#     return type(
#                     name,
#                     (Plasma, ),
#                     {
#                         '__slots__' : _slots,
#                         '__init__'  : _init,
#                         '_comp'     : _comp,
#                     }
#                 )

    
    
        