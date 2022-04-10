from collections import namedtuple
from typing import Any, Sequence
import numpy as np
import numpy.random as rnd
try:
    import table
except Exception:
    from . import table

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
        elif False in map(lambda o: isinstance(o, str), __x):
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

    def slice(self, __i: Any) -> object:
        """ Get a slice of the table. """
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
        _slice._u = self._u
        return _slice

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
        T    = np.asfarray(T).flatten()
        u    = np.zeros(T.shape)
        m    = (T != 0.0)
        u[m] = np.exp(-self.value / T[m,None]) @ self.g
        return u[0] if flag else u
