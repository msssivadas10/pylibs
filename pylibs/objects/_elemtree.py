import numpy as np
from typing import Any
from scipy.interpolate import CubicSpline
from functools import reduce
from pylibs.objects._tables import LinesTable, LevelsTable
from pylibs.objects.tree import Node

class SpeciesNode(Node):
    """ 
    A node representing a species of some element. Each species will have its 
    own ionisation energy, spectral lines etc. These data can be stored in a 
    :class:`SpeciesNode` and connected to a node representing the element.

    Parameters
    ----------
    key: int
        Key to identify this species.
    Vs: float
        Ionisation energy of this species (in units of eV).
    levels: LevelsTable
        A table of energy of levels.
    lines: LinesTable
        A table of spectral lines data.
    interpolate: bool, optional
        If set true, then use an interpolation table to calculate the partition 
        function. Otherwise, use the energy levels data.
    T: array_like, optional
        An array of temperature values to pre-compute the partition function values. 
        Default is 101 points in [0,5]. 

    Attributes
    ----------
    Ns: float
        Fraction of atoms of this species in some multi-element mixture.
    Us: float
        Last computed value of the partition function will be saved here.
    T: float
        Temperature (in eV) used to compute the partition function last time.
    pfunc: CubicSpline, None
        If interpolation is set, store the partition function table.
    lines: LinesTable
        Table of spectral lines (getter).
    levels: LevelsTable
        Table of energy levels (getter).

    """
    __slots__ = 'key', '_lines', '_levels', 'Vs', 'Ns', 'Us', 'pfunc', 'T'
    __name__  = 'SpeciesNode'

    def __init__(self, key: int, Vs: float, levels: LevelsTable, lines: LinesTable = None, interpolate: bool = True, T: Any = None) -> None:
        super().__init__()
        self.setLevels(levels)
        self.setLines(lines)

        self.key = key

        if not np.isscalar(Vs):
            raise TypeError("Vs must be a scalar")

        self.Vs, self.Ns, self.Us = Vs, None, None
        self.T                    = None
        self.pfunc                = None

        # if interpolation is on, make a spline of the partition function values 
        if interpolate:
            if T is not None:
                T = np.asfarray(T)
                if np.ndim(T) != 1:
                    raise TypeError("T must be a 1D array")
                elif T.shape[0] < 10:
                    raise TypeError("T should have atleast 10 items")
            else:
                T = np.linspace(0.0, 5.0, 101) 
            self.pfunc  = CubicSpline( T, self._levels.U(T) )
        
    def keys(self) -> tuple:
        return self.__slots__

    def U(self, T: Any, save: bool = True) -> Any:
        """ 
        Calculate the partition function and store the last value to attribute `Us`. 
        """
        if self.pfunc is None:
            Us = self.pfunc(T)
        else:
            Us = self.levels.U(T)
        if save:
            self.T  = T
            self.Us = Us
        return Us

    def setLevels(self, levels: LevelsTable) -> None:
        """ Set a levels table. """
        if levels is not None:
            if not isinstance(levels, LevelsTable):
                raise TypeError("levels must be a 'LevelsTable'")
        self._levels = levels

    def setLines(self, lines: LinesTable) -> None:
        if lines is not None:
            if not isinstance(lines, LinesTable):
                raise TypeError("lines must be a 'LinesTable'")
            if lines.s is None:
                lines.s = np.repeat( self.key, lines.nr )
        self._lines = lines 

    def getLTEIntensities(self) -> Any:
        """
        Calculate line intensitiesat LTE.
        """
        if self.Ns is None:
            raise ValueError("composition 'Ns' is not set")
        elif self.Us is None or self.T is None:
            raise ValueError("partition function 'Us' is not calculated")
        I = (
                (self.Ns / self.Us) 
                    * self.lines.gk * self.lines.aki * np.exp(-self.lines.ek / self.T)
                    * (1.24E+03 / self.lines.wavelen)
            )
        return I
    
    @property
    def lines(self) -> LinesTable:
        """ Lines of this species. """
        return self._lines
    
    @property
    def levels(self) -> LevelsTable:
        """ Energy levels of this species. """
        return self._levels

class ElementNode(Node):
    """ 
    A node representing a specific element. An element node will have some species 
    nodes connected to it.

    Parameters
    ----------
    key: str
        Key to specify this element. Can be the chemical symbol or its name.
    m: float
        Atomic mass in amu.
    lines: LinesTable, optional
        Table of spectral lines (all species).
    
    Attributes
    ----------
    Nx: float
        Number of atoms of this element in some mixture.

    """
    __slots__ = 'key', 'm', 'Nx'
    __name__  = 'ElementNode'

    def __init__(self, key: str, m: float) -> None:
        super().__init__()  

        self.key = key
        self.m   = m
        self.Nx  = None

    def keys(self) -> tuple:
        return self.__slots__
        
    @property
    def nspec(self) -> int:
        """ Number of species of this element. """
        return self.nchildren

    @property
    def Us(self) -> Any:
        """ Last calculated partition function values. """
        return np.array([ s.Us for s in self.children() ])

    @property
    def Ns(self) -> Any:
        """ Present composition values. """
        return np.array([ s.Ns for s in self.children() ])
    
    @property
    def lines(self) -> LinesTable:
        """ Get the all lines of this element, including that of species. """
        if self._lines is not None:
            return self._lines

        hasI, hasXY = True, True
        for s in self._child:
            if s.lines is None:
                continue
            if s.lines.boltzX is None or s.lines.boltzY is None:
                hasXY = False
            if s.lines.I is None:
                hasI = False 

        lines = LinesTable(elem = [], s = [])
        if hasI:
            lines.setLineIntensity([])
        if hasXY:
            lines.setBoltzmannXY([], [])

        for s in self.children():
            if s.lines is None:
                # raise ValueError("species {} has no lines".format(i))
                continue
            lines.join( s.lines )
        return lines 

    def U(self, T: Any) -> Any:
        """ Calculate the partition function for each attached species. """
        return np.array([ s.U(T) for s in self._child ])
    
    def species(self, key: int) -> SpeciesNode:
        """ Get a species node. """
        return self.child(key)

    def addspecies(self, __spec: SpeciesNode) -> None:
        """ Add a child species. """
        if not isinstance(__spec, SpeciesNode):
            raise TypeError("node must be a 'SpeciesNode'")
        return self.addchild(__spec, None)

    def getLTEComposition(self, Te: float, Ne: float) -> None:
        """ 
        Calculate the composition of the species at LTE.
        """
        if not np.isscalar(Te):
            raise TypeError("Te must be a scalar")
        if not np.isscalar(Ne):
            raise TypeError("Ne must be a scalar")
        if self.Nx is None:
            raise ValueError("composition 'Nx' is not set")

        const = 6.009E+21 * Te**1.5 / Ne
        
        # calculate the partition functions of the species
        Us = [ s.U(Te) for s in self.children() ]
        Vs = [ s.Vs    for s in self.children() ]

        # calculate composition 
        fs = [ const * Us[s+1] / Us[s] * np.exp(-Vs[s] / Te) for s in range(self.nspec-1) ]  
        N0 = self.Nx / (1 + reduce( lambda x,y: (1 + x)*y, reversed(fs) ) )
        Ns = np.cumprod([1.0, *fs]) * N0
        
        for s in range(self.nspec):
            self.species(s).Ns = Ns[s]



    

 