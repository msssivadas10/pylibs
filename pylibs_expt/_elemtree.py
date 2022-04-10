from typing import Any, Iterable, Sequence, Union
from scipy.interpolate import CubicSpline
import numpy as np
try:
    import tree
    from _tables import LevelsTable, LinesTable
except Exception:
    from . import tree
    from . import _tables
    LevelsTable, LinesTable = _tables.LevelsTable, _tables.LinesTable


class _SpeciesNode(tree.Node):
    """ Base class for species or element nodes. """
    __slots__ = 'key', '_lines', '_levels', 

    def __init__(self, key: Any, levels: LevelsTable = None, lines: LinesTable = None) -> None:
        super().__init__()  

        self.key, self._levels, self._lines = key, levels, lines
        self.setLevels(levels)
        self.setLines(lines)   

    def keys(self) -> tuple:
        return self.__slots__  
        
    def setLevels(self, levels: LevelsTable) -> None:
        """ Set a levels table. """
        if levels is not None:
            if isinstance(levels, (list, tuple, np.ndarray)):
                levels = np.asfarray(levels)
                if np.ndim(levels) != 2:
                    raise TypeError("levels should be a 2D array")
                elif levels.shape[1] != 2:
                    raise TypeError("levels array should have 2 columns: 'g' and 'value'")
                levels = LevelsTable(levels[:,0], levels[:,1])
            elif not isinstance(levels, LevelsTable):
                raise TypeError("levels must be a 'LevelsTable'")
        self._levels = levels

    def setLines(self, lines: LinesTable) -> None:  
        """ Set a lines table. """      
        if lines is not None:
            if not isinstance(lines, LinesTable):
                raise TypeError("lines must be a 'LinesTable'")
        self._lines = lines    

    def freeLines(self) -> None:
        """ Disconnect the lines table from the node. """
        return self.setLines(None)

    def freeLevels(self) -> None:
        """ Disconnect the levels table from the node. """
        return self.setLevels(None)

class SpeciesNode(_SpeciesNode):
    """ 
    A node representing a species of some element.
    """
    __slots__ = 'Vs', 'Ns', 'Us', 'pfunc', 
    __name__  = 'SpeciesNode'

    def __init__(self, key: int, Vs: float, levels: LevelsTable, lines: LinesTable = None, interpolate: bool = True, T: Any = None) -> None:
        super().__init__(key, None, None)
        self.setLevels(levels)
        self.setLines(lines)

        if not np.isscalar(Vs):
            raise TypeError("Vs must be a scalar")

        self.Vs, self.Ns, self.Us = Vs, None, None
        self.pfunc                = None

        # if interpolation is on, make a spline of the partition function values 
        if interpolate:
            T = np.linspace(0.0, 5.0, 101) if T is None else np.asfarray(T)
            self.pfunc  = CubicSpline( T, self._levels.U(T) )
            # self.levels = None

    def keys(self) -> tuple:
        return _SpeciesNode.__slots__ + self.__slots__

    def U(self, T: Any) -> Any:
        """ 
        Calculate the partition function and store the last value to attribute `Us`. 
        """
        if self.pfunc is None:
            self.Us = self.pfunc(T)
        else:
            self.Us = self.levels.U(T)
        return self.Us

    def setLines(self, lines: LinesTable) -> None:
        if lines is not None:
            if not isinstance(lines, LinesTable):
                raise TypeError("lines must be a 'LinesTable'")
            if lines.s is None:
                lines.s = np.repeat( self.key, lines.nr )
        return super().setLines(lines)

class ElementNode(_SpeciesNode):
    """ 
    A node representing a specific element.
    """
    __slots__ = 'm', 
    __name__  = 'ElementNode'

    def __init__(self, key: str, m: float, lines: LinesTable = None) -> None:
        super().__init__(key, None, lines)
        self.m = m

    def keys(self) -> tuple:
        return _SpeciesNode.__slots__ + self.__slots__
        
    @property
    def nspec(self) -> int:
        """ Number of species of this element. """
        return self.nchildren

    @property
    def Us(self) -> Any:
        """ Last calculated partition function values. """
        return np.array([ s.Us for s in self._child ])
    
    @property
    def lines(self) -> LinesTable:
        """ Get the all lines of this element, including that of species. """
        if self._lines is not None:
            return self._lines
        lines = LinesTable(elem = [], s = [])
        for s in self._child:
            if s.lines is None:
                # raise ValueError("species {} has no lines".format(i))
                continue
            lines.join( s.lines )
        return lines 

    @lines.setter
    def lines(self, lines: LinesTable) -> None:
        """ Link a lines table. This will free lines of child species. """
        if not isinstance(lines, LinesTable):
            raise TypeError("lines must be a 'LinesTable'")
        self.setLines(lines)         
        for s in self._child:
            s.freeLines()

    def setLines(self, lines: LinesTable) -> None:
        if lines is not None:
            if not isinstance(lines, LinesTable):
                raise TypeError("lines must be a 'LinesTable'")
            if lines.elem is None:
                lines.elem = np.repeat( self.key, lines.nr )
            if lines.s is None:
                raise ValueError("table should have species column ('s')")
        return super().setLines(lines)

    def U(self, T: Any) -> Any:
        """ Calculate the partition function for each attached species. """
        return np.array([ s.U(T) for s in self._child ])
    
    def species(self, key: int) -> SpeciesNode:
        """ Get a species node. """
        return super().child(key)

    def addspecies(self, __spec: SpeciesNode) -> None:
        """ Add a child species. """
        if not isinstance(__spec, SpeciesNode):
            raise TypeError("node must be a 'SpeciesNode'")
        return self.addchild(__spec, None)

def element(key: str, m: float, nspec: int, Vs: Sequence[float], levels: Sequence[LevelsTable], lines: Union[LinesTable, Sequence[LinesTable]], interpolate: bool = True, T: Any = None) -> ElementNode:
    """ 
    Create a new element node with species. 
    """
    key = key.lower()

    if m < 0.0:
        raise ValueError("atomic mass cannot be negative")

    if nspec < 1:
        raise ValueError("there should be at least one species")

    if len(Vs) != nspec:
        raise ValueError("incorrect number of ionization energies, must be same as the number of species")
    Vs = np.asfarray(Vs)

    if len(levels) != nspec:
        raise ValueError("incorrect number of energy levels, must be same as the number of species")

    if isinstance(lines, LinesTable):
        if lines.s is None:
            raise ValueError("table should have species column ('s')")

        # check if the table has `elem` column
        if lines.elem is None:
            lines.elem = np.repeat(elem.key, lines.nr)

        _lines = []
        for s in range(nspec):
            _lines.append( lines.slice( ( lines.elem == key ) & ( lines.s == s ) ) )
        lines = _lines

    else:
        if len(lines) != nspec:
            raise ValueError("incorrect number of lines tables, must be same as the number of species")
        elif False in map(lambda o: isinstance(o, LinesTable), lines):
            raise TypeError("lines lust be an array of 'LinesTable'")

        for s, _lines in enumerate(lines):

            # check if the table has `elem` column else add
            if _lines.elem is None:
                _lines.elem = np.repeat(elem.key, _lines.nr)

            # check if the table `s` column else add
            if _lines.s is None:
                _lines.s = np.repeat(s, _lines.nr)

    elem = ElementNode(key, m, None)
    for s in range(nspec):
        elem.addspecies( SpeciesNode(s, Vs[s], levels[s], lines[s], interpolate, T) )

    return elem

def elementTree(__nodes: Iterable[ElementNode]):
    """
    Create a tree of elements. 
    """
    if len(__nodes) == 0:
        raise TypeError("input cannot be empty")
    
    root = tree.Node()
    for elem in __nodes:
        root.addchild( elem, key = elem.key )
    
    return root
    
 