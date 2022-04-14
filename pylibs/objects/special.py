import numpy as np
from collections import namedtuple
from functools import reduce
from typing import Any, Sequence
from pylibs.objects.table import Table, TableSubsetGetter, TableError
from pylibs.objects.tree import Node
from scipy.interpolate import CubicSpline

# =========================================================================================
# Special Tables 
# ========================================================================================= 

class LinesTable(Table):
    r"""
    A table storing information about spectral lines. This include the wavelength (in nm), 
    transition probability :math:`A_{ki}` and its uncertainity, value and weight of the 
    upper level of transition :math:`E_k` and :math:`g_k`, and the keys for the element or 
    species producing the line. 

    Parameters
    ----------
    wavelen: array of float
        Wavelength of the transition in nanometers.
    Aki: array of float
        Transition probability in :math:`{\rm sec}^{-1}`.
    Ek: array of float
        Energy of the upper levels in :math:`{\rm eV}`.
    gk: array of float or int
        Weights of the upper levels.
    s: array of int
        Species key. Specify the ionization state of the species producing the transition.
    elem: array of str
        Element key. Specify the symbol of the species producing the transition.
    errAki: array of float, optional
        Uncertainity in transition probability as percentages.

    Attributes
    ----------
    I: array of float
        Store intensity values.
    botzX: array of float
        Store the X coordinates of the Boltzmann plot.
    botzY: array of float
        Store the Y coordinates of the Boltzmann plot.

    Examples
    --------
    todo.

    """
    __slots__ = (
                    'wavelen', 'aki', 'ek', 'gk', 'elem', 's', 'I', 'boltzX',
                    'boltzY', 'errAki', '_cols', 
                )
    __name__  = 'LinesTable'

    def __init__(self, wavelen: Sequence[float] = [], Aki: Sequence[float] = [], Ek: Sequence[float] = [], gk: Sequence[float] = [], elem: Sequence[str] = None, s: Sequence[int] = None, errAki: Sequence[float] = None, ) -> None:
        self._cols = ['wavelen', 'aki', 'ek', 'gk', ]

        if not min(map(lambda o: np.ndim(o) == 1, [wavelen, Aki, Ek, gk])):
            raise TableError("columns must be 1D arrays")

        nr, nc = len(wavelen), 4
        if not min(map(lambda o: len(o) == nr, [Aki, Ek, gk])):
            raise TableError("all coulmns should have the same size")
        
        self.wavelen = np.asfarray(wavelen)
        self.aki     = np.asfarray(Aki)
        self.ek      = np.asfarray(Ek)
        self.gk      = np.asfarray(gk)

        self.elem    = None
        self.s       = None
        self.I       = None
        self.boltzX  = None
        self.boltzY  = None
        self.errAki  = None

        self._nc, self._nr  = nc, nr
        self._subset_getter = TableSubsetGetter(self)
        self.table_row      = namedtuple(
                                            'Line',
                                            ['wavelen', 'aki', 'ek', 'gk', 'elem', 's', 'I', 'boltzX', 'boltzY', 'errAki']
                                        )

        # if elements keys are given, add column
        if elem is not None:
            self.setElementKey(elem)
        
        # if species keys are given, add column
        if s is not None:
            self.setSpeciesKey(s)

        # if error in Aki given, add column
        if errAki is not None:
            self.setAkiErrors(errAki)
        else:
            self.setAkiErrors( np.zeros(self.nr) )
    
    def r(self, __i: int) -> tuple:
        o = {
                'wavelen' : None, 
                'aki'     : None, 
                'ek'      : None, 
                'gk'      : None, 
                'elem'    : None, 
                's'       : None, 
                'I'       : None, 
                'boltzX'  : None, 
                'boltzY'  : None, 
                'errAki'  : None,
                **{
                        __col: self[__col][__i] for __col in self.colnames
                  }
            }
        return self.table_row(**o)

    def _arrangeColunmNames(self) -> None:
        r"""
        Arrange the column names so that the order would be `wavelen`, `aki`, `ek`, `gk`, 
        `elem`, `s`, `errAki`, `I`, `boltzX`, `boltzY`.
        """
        cols = ['wavelen', 'aki', 'ek', 'gk', ]
        for key in ['elem', 's', 'errAki', 'I', 'boltzX', 'boltzY']:
            if key in self._cols:
                cols.append(key)
        self._cols = cols

    def setElementKey(self, __x: Sequence[str]) -> None:
        r"""
        Set element keys for lines. This adds a new column `elem` to the table. Once 
        set, this column cannot be changed.

        Parameters
        ----------
        __x: array of str
            Element keys. Must be of the same size as other columns.
        """
        if self.elem is not None:
            raise TableError("column 'elem' cannot be redefined")

        if np.ndim(__x) != 1:
            raise TableError("columns must be 1D arrays")
        elif len(__x) != self.nr:
            raise TableError("all coulmns should have the same size")
        elif False in map(lambda o: isinstance(o, str), __x):
            raise TypeError("element key must be 'str'")
        self.elem = np.array(__x).astype('str')
        self._cols.append('elem')
        self._nc += 1
        self._arrangeColunmNames()

    def setSpeciesKey(self, __x: Sequence[int]) -> None:
        r"""
        Set species keys for lines. This adds a new column `s` to the table. Once 
        set, this column cannot be changed.

        Parameters
        ----------
        __x: array of int
            Species keys. Must be of the same size as other columns.
        """
        if self.s is not None:
            raise TableError("column 's' cannot be redefined")

        if np.ndim(__x) != 1:
            raise TableError("columns must be 1D arrays")
        elif len(__x) != self.nr:
            raise TableError("all coulmns should have the same size")
        self.s = np.array(__x).astype('int')
        self._cols.append('s')
        self._nc += 1
        self._arrangeColunmNames()

    def setAkiErrors(self, __x: Sequence[float]) -> None:
        r"""
        Set accuracy of :math:`A_{ki}` for lines. This adds a new column `elem` to the 
        table. Once set, this column cannot be changed.

        Parameters
        ----------
        __x: array of float
            Accuracy in percent. Must be of the same size as other columns.
        """
        # if self.errAki is not None:
        #     raise TableError("column 'errAki' cannot be redefined")

        if np.ndim(__x) != 1:
            raise TableError("columns must be 1D arrays")
        elif len(__x) != self.nr:
            raise TableError("all coulmns should have the same size")
        self.errAki = np.asfarray(__x)
        if sum(self.errAki < 0.0):
            raise ValueError("uncertainity must be positive")
        self._cols.append('errAki')
        self._nc += 1
        self._arrangeColunmNames()

    def setLineIntensity(self, __x: Sequence[float]) -> None:
        r"""
        Set intensity of the lines. This adds a new column `I` to the table.

        Parameters
        ----------
        __x: array of float
            Intensity values. Must be of the same size as other columns.
        """
        if np.ndim(__x) != 1:
            raise TableError("columns must be 1D arrays")
        elif len(__x) != self.nr:
            raise TableError("all coulmns should have the same size")
        if self.I is None:
            self._nc += 1
            self._cols.append('I')
            self._arrangeColunmNames()
        self.I = np.asfarray(__x)

    def setBoltzmannXY(self, __x: Sequence[float], __y: Sequence[float]) -> None:
        r"""
        Set Boltzmann plane coordinates of the lines. This adds two new columns `boltzX` 
        and `boltzY` to the table.

        Parameters
        ----------
        __x, __y: array of float
            X and Y coordinates.
        """
        if np.ndim(__x) != 1 or np.ndim(__y) != 1:
            raise TableError("columns must be 1D arrays")
        elif len(__x) != self.nr or len(__y) != self.nr:
            raise TableError("all coulmns should have the same size")
        if self.boltzX is None and self.boltzY is None:
            self._nc += 2
            self._cols.append('boltzX')
            self._cols.append('boltzY')
            self._arrangeColunmNames()
        self.boltzX = np.asfarray(__x)
        self.boltzY = np.asfarray(__y)

    def _colnames(self) -> tuple:
        return self._cols

    def slice(self, __i: Any) -> object:
        r""" Get a slice of the table. """
        if np.isscalar(__i):
            raise TypeError("argument must be an array")
        _slice = LinesTable(
                                self.wavelen[__i],
                                self.aki[__i],
                                self.ek[__i],
                                self.gk[__i],
                                None, 
                                None,
                                self.errAki[__i],
                           )
        if self.elem is not None:
            _slice.setElementKey(self.elem[__i])
        if self.s is not None:
            _slice.setSpeciesKey(self.s[__i])
        if self.boltzX is not None and self.boltzY is not None:
            _slice.setBoltzmannXY(self.boltzX[__i], self.boltzY[__i])
        if self.I is not None:
            _slice.setLineIntensity(self.I[__i])
        return _slice

class LevelsTable(Table):
    r"""
    A table storing the energy level data. This table will have two columns: 
    weight of the level `g` and its value in eV `value`. 

    Parameters
    ----------
    g: array_like
        Weights for the levels.
    value: array_like
        Value of the level in eV.

    """
    __slots__ = 'g', 'value', '_cols', 'attrs', 
    __name__  = 'LevelsTable'

    def __init__(self, g: Sequence[int], value: Sequence[float]) -> None:
        self._cols = ['g', 'value']

        if np.ndim(g) != 1 or np.ndim(value) != 1:
            raise Table('columns must be 1D arrays')
        nr = len(g)
        if len(value) != nr:
            raise Table('all columns must have the same size')
        self.g     = np.asfarray(g)
        self.value = np.asfarray(value)
        self.attrs = {}

        self._nc, self._nr  = 2, nr
        self._subset_getter = TableSubsetGetter(self)
        self.table_row      = namedtuple('Level', self._cols)
    
    def _colnames(self) -> tuple:
        return self._cols

    def U(self, T: Any) -> Any:
        r"""
        Calculate the partition function using the energy levels in the table. It is
        given by 

        .. math::
            U(T) = \sum_k g_k \exp \left( - \frac{E_k}{T} \right)

        Parameters
        ----------
        T: array_like
            Temperature in eV.
        
        Returns
        -------
        U: array_like
            Value of partition function.
        """
        flag = np.isscalar(T)
        T    = np.asfarray(T).flatten()
        u    = np.zeros(T.shape)
        m    = (T != 0.0)
        u[m] = np.exp(-self.value / T[m,None]) @ self.g
        return u[0] if flag else u


# =========================================================================================
# Special Node Objects 
# =========================================================================================

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
        self.key = key

        if not np.isscalar(Vs):
            raise TypeError("Vs must be a scalar")

        self.setLevels(levels)
        self.setLines(lines)

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



    
