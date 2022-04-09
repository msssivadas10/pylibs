from collections import namedtuple
from itertools import product, repeat
from typing import Any, Iterable, Sequence, Type, Union
from scipy.interpolate import CubicSpline
import numpy as np
import numpy.random as rnd
try:
    import table, tree
except Exception:
    from . import table, tree

class Line:
    """
    Store data for a spectral line.
    """
    __slots__ = (
                    'wavelen', 'aki', 'ek', 'gk', 'elem', 's', 'I', 'boltzX',
                    'boltzY', 'errAki'
                )

    def __init__(self, wavelen: float, aki: float, ek: float, gk: float, elem: str = None, s: int = None, I: float = None, boltzX: float = None, boltzY: float = None, errAki: float = None) -> None:
        self.wavelen = wavelen
        self.aki     = aki 
        self.ek      = ek 
        self.gk      = gk 
        self.elem    = elem
        self.s       = s 
        self.I       = I
        self.boltzX  = boltzX
        self.boltzY  = boltzY
        self.errAki  = errAki

    def __repr__(self) -> str:
        s = [
                f'wavelen={self.wavelen:.3f} nm',
                f'aki={self.aki:.3e} Hz',
                f'ek={self.ek:.3f} eV',
                f'gk={self.gk}',
            ]
        if self.elem is not None:
            s.append( f"elem='{self.elem}'" )
        if self.s is not None:
            s.append( f"s={self.s}" )
        if self.I is not None:
            s.append( f"I={self.I:.3e}" )
        if self.boltzX is not None:
            s.append( f"boltzX={self.boltzX:.3e}" )
        if self.boltzY is not None:
            s.append( f"boltzY={self.boltzY:.3e}" )
        if self.errAki is not None:
            s.append( f"errAki={self.errAki:.3e} %" )
        return 'Line({})'.format(', '.join(s))

class LinesTable(table.Table):
    """
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
    u_wavelen: float
        Unit of wavelength in nanometer. For example, if want the wavelength to be in 
        angstrom (= 0.1 nm), this must be set to `0.1`.
    u_Aki: float
        Unit of transition probability in Hz.
    u_energy: float
        Unit of energy in eV.
    I: array of float
        Store intensity values.
    botzX: array of float
        Store the X coordinates of the Boltzmann plot.
    botzY: array of float
        Store the Y coordinates of the Boltzmann plot.
    
    Notes
    -----
    To get the values of wavelength, transition probability or energy in default units 
    (nm, Hz and eV respectively) use the lowercased names and for their values in the 
    specified units use titlecased names (e.g., `wavelen` is in nanometers, but `Wavelen` 
    is in the units specified by user).

    Examples
    --------
    todo.

    """
    __slots__ = (
                    'wavelen', 'aki', 'ek', 'gk', 'elem', 's', 'I', 'boltzX',
                    'boltzY', 'errAki', '_cols', '_u', 
                )
    __name__  = 'LinesTable'

    def __init__(self, wavelen: Sequence[float] = [], Aki: Sequence[float] = [], Ek: Sequence[float] = [], gk: Sequence[float] = [], elem: Sequence[str] = None, s: Sequence[int] = None, errAki: Sequence[float] = None, ) -> None:
        self._cols = ['wavelen', 'aki', 'ek', 'gk', ]

        if not min(map(lambda o: np.ndim(o) == 1, [wavelen, Aki, Ek, gk])):
            raise table.TableError("columns must be 1D arrays")

        nr, nc = len(wavelen), 4
        if not min(map(lambda o: len(o) == nr, [Aki, Ek, gk])):
            raise table.TableError("all coulmns should have the same size")
        
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

        self._u = [1.0, 1.0, 1.0] # units of wavelength, Aki and energy

        self._nc, self._nr  = nc, nr
        self._subset_getter = table.TableSubsetGetter(self)
        self.table_row      = Line

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

    def _arrangeColunmNames(self) -> None:
        """
        Arrange the column names so that the order would be `wavelen`, `aki`, `ek`, `gk`, 
        `elem`, `s`, `errAki`, `I`, `boltzX`, `boltzY`.
        """
        cols = ['wavelen', 'aki', 'ek', 'gk', ]
        for key in ['elem', 's', 'errAki', 'I', 'boltzX', 'boltzY']:
            if key in self._cols:
                cols.append(key)
        self._cols = cols

    def setElementKey(self, __x: Sequence[str]) -> None:
        """
        Set element keys for lines. This adds a new column `elem` to the table. Once 
        set, this column cannot be changed.

        Parameters
        ----------
        __x: array of str
            Element keys. Must be of the same size as other columns.
        """
        if self.elem is not None:
            raise table.TableError("column 'elem' cannot be redefined")

        if np.ndim(__x) != 1:
            raise table.TableError("columns must be 1D arrays")
        elif len(__x) != self.nr:
            raise table.TableError("all coulmns should have the same size")
        elif not min(map(lambda o: isinstance(o, str), __x)):
            raise TypeError("element key must be 'str'")
        self.elem = np.array(__x).astype('str')
        self._cols.append('elem')
        self._nc += 1
        self._arrangeColunmNames()

    def setSpeciesKey(self, __x: Sequence[int]) -> None:
        """
        Set species keys for lines. This adds a new column `s` to the table. Once 
        set, this column cannot be changed.

        Parameters
        ----------
        __x: array of int
            Species keys. Must be of the same size as other columns.
        """
        if self.s is not None:
            raise table.TableError("column 's' cannot be redefined")

        if np.ndim(__x) != 1:
            raise table.TableError("columns must be 1D arrays")
        elif len(__x) != self.nr:
            raise table.TableError("all coulmns should have the same size")
        self.s = np.array(__x).astype('int')
        self._cols.append('s')
        self._nc += 1
        self._arrangeColunmNames()

    def setAkiErrors(self, __x: Sequence[float]) -> None:
        """
        Set accuracy of :math:`A_{ki}` for lines. This adds a new column `elem` to the 
        table. Once set, this column cannot be changed.

        Parameters
        ----------
        __x: array of float
            Accuracy in percent. Must be of the same size as other columns.
        """
        if self.errAki is not None:
            raise table.TableError("column 'errAki' cannot be redefined")

        if np.ndim(__x) != 1:
            raise table.TableError("columns must be 1D arrays")
        elif len(__x) != self.nr:
            raise table.TableError("all coulmns should have the same size")
        self.errAki = np.asfarray(__x)
        if sum(self.errAki < 0.0):
            raise ValueError("uncertainity must be positive")
        self._cols.append('errAki')
        self._nc += 1
        self._arrangeColunmNames()

    def setLineIntensity(self, __x: Sequence[float]) -> None:
        """
        Set intensity of the lines. This adds a new column `I` to the table.

        Parameters
        ----------
        __x: array of float
            Intensity values. Must be of the same size as other columns.
        """
        if np.ndim(__x) != 1:
            raise table.TableError("columns must be 1D arrays")
        elif len(__x) != self.nr:
            raise table.TableError("all coulmns should have the same size")
        if self.I is None:
            self._nc += 1
            self._cols.append('I')
            self._arrangeColunmNames()
        self.I = np.asfarray(__x)

    def setBoltzmannXY(self, __x: Sequence[float], __y: Sequence[float]) -> None:
        """
        Set Boltzmann plane coordinates of the lines. This adds two new columns `boltzX` 
        and `boltzY` to the table.

        Parameters
        ----------
        __x, __y: array of float
            X and Y coordinates.
        """
        if np.ndim(__x) != 1 or np.ndim(__y) != 1:
            raise table.TableError("columns must be 1D arrays")
        elif len(__x) != self.nr or len(__y) != self.nr:
            raise table.TableError("all coulmns should have the same size")
        if self.boltzX is None and self.boltzY is None:
            self._nc += 2
            self._cols.append('boltzX')
            self._cols.append('boltzY')
            self._arrangeColunmNames()
        self.boltzX = np.asfarray(__x)
        self.boltzY = np.asfarray(__y)

    def _colnames(self) -> tuple:
        return self._cols

    @property
    def randomErrorAki(self) -> Sequence[float]:
        """
        Generate a random error in :math:`A_{ki}` values (in percent).
        """
        if self.errAki is None:
            return np.zeros(self.nr)
        return rnd.normal(scale = self.errAki)

    @property
    def u_wavelen(self) -> float:
        return self._u[0]
    
    @u_wavelen.setter
    def u_wavelen(self, __value: float) -> None:
        if __value < 0.0:
            raise ValueError("value must be positive")
        self._u[0] = __value

    @property
    def u_Aki(self) -> float:
        return self._u[1]
    
    @u_Aki.setter
    def u_Aki(self, __value: float) -> None:
        if __value < 0.0:
            raise ValueError("value must be positive")
        self._u[1] = __value

    @property
    def u_energy(self) -> float:
        return self._u[2]
    
    @u_energy.setter
    def u_energy(self, __value: float) -> None:
        if __value < 0.0:
            raise ValueError("value must be positive")
        self._u[2] = __value
    
    @property
    def Wavelen(self) -> Sequence[float]:
        return self.wavelen * self.u_wavelen
    
    @property
    def Aki(self) -> Sequence[float]:
        return self.aki * self.u_Aki

    @property
    def Ek(self) -> Sequence[float]:
        return self.ek * self.u_energy

class LevelsTable(table.Table):
    """
    A table storing the energy level data.
    """
    __slots__ = 'g', 'value', '_cols', 'attrs', 
    __name__  = 'LevelsTable'

    def __init__(self, g: Sequence[int], value: Sequence[float]) -> None:
        self._cols = ['g', 'value']

        if np.ndim(g) != 1 or np.ndim(value) != 1:
            raise table.Table('columns must be 1D arrays')
        nr = len(g)
        if len(value) != nr:
            raise table.Table('all columns must have the same size')
        self.g     = np.asfarray(g)
        self.value = np.asfarray(value)
        self.attrs = {}

        self._nc, self._nr  = 2, nr
        self._subset_getter = table.TableSubsetGetter(self)
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

        T = np.asfarray(T).flatten()
        u = np.exp(-self.value / T[:,None]) @ self.g
        return u[0] if flag else u

class ElementData:
    """
    An object storing the data for an element.
    """
    __slots__ = 'key', 'm', 'smax', 'Vs', 'levels', 'lines', 

    def __init__(self, key: str, m: float, smax: int, Vs: Sequence[float], levels: Sequence[LevelsTable], lines: Union[LinesTable, Sequence[LinesTable]] = None) -> None:
        self.key = key.lower()

        if m < 0.0:
            raise ValueError("atomic mass cannot be negative")
        self.m = m

        if smax < 1:
            raise ValueError("there should be at least one species")
        self.smax = smax

        if len(Vs) != smax:
            raise ValueError("incorrect number of ionization energies, must be same as the number of species")
        self.Vs = np.asfarray(Vs)

        if len(levels) != smax:
            raise ValueError("incorrect number of energy levels, must be same as the number of species")
        elif not min(map(lambda o: isinstance(o, LevelsTable), levels)):
            raise TypeError("levels must be a list of 'LevelsTable' objects")
        self.levels = tuple(levels)

        self.lines = None
        if isinstance(lines, LinesTable):
            if lines.s is None:
                raise ValueError("table should have species column ('s')")

            # check if the table has `elem` column else add
            if lines.elem is None:
                lines.setElementKey(np.repeat(self.key, lines.nr))
            else:
                keys = np.unique(lines.elem)
                if len(keys) != 1:
                    raise ValueError("table has lines of multiple elements")
                elif self.key != keys[0]:
                    raise ValueError("table has no lines correspond to this element key")
            self.lines = lines
        elif min(map(lambda o: isinstance(o, LinesTable), lines)):
            self.lines = LinesTable(elem = [], s = [])

            for s, _lines in enumerate(lines):
                # check if the table has `elem` column else add
                if _lines.elem is None:
                    _lines.setElementKey(np.repeat(self.key, _lines.nr))
                else:
                    keys = np.unique(_lines.elem)
                    if len(keys) != 1:
                        raise ValueError("table has lines of multiple elements")
                    elif self.key != keys[0]:
                        raise ValueError("table has no lines correspond to this element key")

                # check if the table `s` column else add
                if _lines.s is None:
                    _lines.setSpeciesKey(np.repeat(s, _lines.nr))
                else:
                    ss = np.unique(_lines.s)
                    if len(ss) != 1:
                        raise ValueError("table has lines of multiple species")
                    elif s != ss[0]:
                        raise ValueError("table has no lines correspond to this species")
                
                self.lines.join(_lines)
        elif lines is not None:
            raise TypeError("lines must be a 'LinesTable' or 'list' of 'LinesTable'")

    def __repr__(self) -> str:
        return f"ElementData('{self.key}')"

    def U(self, T: Any) -> Any:
        """
        Calculate the partition function vector. 
        """
        return np.array(
                        list(
                             map(
                                    lambda o: o.U(T), self.levels
                                )
                            )
                        )

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

             
def plasma(name: str, comp: Iterable[ElementData], lines: LinesTable = None, interpolate: bool = True, T: Any = None) -> Type[Plasma]:
    """
    Create a specific plasma class.
    """
    _lines = lines
    if lines is None:
        _lines = LinesTable(elem = [], s = [])
    elif isinstance(lines, LinesTable):
        raise TypeError("lines must be a 'LinesTable' object")

    def _U(self, Te: float) -> float:
        if not np.isscalar(Te):
            raise TypeError("Te must be a scalar")
        if self.levels is None:
            return self.Utable(Te)
        return self.levels.U(Te)

    if interpolate:
        if T is None:
            T = np.linspace(1e-3, 5.0, 101)
        elif np.isscalar(T):
            raise TypeError("T must be an array of floats")

    ElementNode = tree.node('ElementNode', ['key', 'm', 'smax', 'Nx'])
    SpeciesNode = tree.node('SpeciesNode', ['Vs', 'Ns', 'Us', 'levels', 'Utable'], {'U': _U}) 
    
    # create a tree from the components
    ct, _slots = tree.Node(), []
    for elem in comp:
        if not isinstance(elem, ElementData):
            raise TypeError("each element in comp should be an 'ElementData' obect")
        et = ElementNode(elem.key, elem.m, elem.smax, None)
        for s in range(elem.smax):
            st = SpeciesNode(elem.Vs[s], None, None, elem.levels[s], None)
            if interpolate:
                st.levels = None
                st.Utable = CubicSpline(T, elem.levels[s].U(T))
            et.addchild(st, key = s)
        ct.addchild(et, key = elem.key)

        if lines is None:
            if elem.lines is None:
                raise ValueError("missing lines data")
            _lines.join(elem.lines)

        _slots.append(elem.key)

    def _init(self: Plasma, *args, **kwargs) -> None:
        self.comp  = ct
        self.lines = _lines
        self.Te    = None
        self.Ne    = None

        # parse arguments
        args    = dict(zip(self.__slots__, args))
        missing = None
        for __name in self.__slots__:
            if __name in args.keys() and __name in kwargs.keys():
                raise TypeError(f"got multiple values for argument '{__name}'")
            if not (__name in args.keys() or __name in kwargs.keys()):
                if missing is None:
                    missing = __name
        args = {**args, **kwargs}
        if len(args) == len(self.__slots__) - 1:
            x = 100.0 - sum(args.values())
            if x < 0.0:
                raise ValueError("total concentration should be 1")
            args[missing], missing = x, None
        if missing is not None:
            raise TypeError(f"missing value for argument '{missing}'")

        if sum(args.values()) != 100.0:
            raise ValueError("total concentration must be 1")
        
        for key, value in args.items():
            setattr(self, key, value)
            self.comp.child(key).Nx = value

    def _comp(self: Plasma) -> tuple:
        return self.__slots__

    return type(
                    name,
                    (Plasma, ),
                    {
                        '__slots__' : _slots,
                        '__init__'  : _init,
                        '_comp'     : _comp,
                    }
                )

    
    
        