from collections import namedtuple
from itertools import product, repeat
from typing import Any, Dict, Sequence, Type, Union
from scipy.interpolate import CubicSpline
import table, warnings, re
import numpy as np

space = namedtuple('space', ['start', 'stop', 'num'])

class LinesTable(table.Table):
    """
    Store the data of spectral lines.
    """
    __slots__ = 'wavelen', 'aki', 'ek', 'gk', 'intens', 'sp', 'elem', '_cols', 'acc', 
    __name__  = 'LinesTable'

    def __init__(self, wavelen: Sequence[float], aki: Sequence[float], ek: Sequence[float], gk: Sequence[float], elem: Sequence[str] = None, sp: Sequence[int] = None, acc: Sequence[float] = None) -> None:
        self._cols = ['wavelen', 'aki', 'ek', 'gk', 'intens']

        if not min(map(lambda o: np.ndim(o) == 1, [wavelen, aki, ek, gk])):
            raise TypeError("all aruments must be arrays")
        __len = len(wavelen)
        if not min(map(lambda o: len(o) == __len, [aki, ek, gk])):
            raise table.TableError("all columns should have the same size")
        self.wavelen = np.array(wavelen, dtype = 'float')
        self.aki     = np.array(aki,     dtype = 'float')
        self.ek      = np.array(ek,      dtype = 'float')
        self.gk      = np.array(gk,      dtype = 'float')
        self.intens  = np.zeros_like(self.wavelen)

        self.acc = None
        if acc is not None:
            if np.ndim != 1:
                raise TypeError("all aruments must be arrays")
            elif len(acc) != __len:
                raise table.TableError("all columns should have the same size")
            self.acc = acc
            self._cols.append('acc')

        self.sp, self.elem = None, None
        if sp is not None:
            if np.ndim(sp) == 0:
                self.sp = int(sp)
            elif np.ndim(sp) == 1:
                if len(sp) != __len:
                    raise table.TableError("all columns should have the same size")
                self.sp = np.array(sp, dtype = 'int')
                self._cols.append('sp')
            else:
                raise TypeError("'sp' must be a 1d array")

        if elem is not None:
            if np.ndim(elem) == 0:
                self.elem = str(elem).lower()
            elif np.ndim(elem) == 1:
                if len(elem) != __len:
                    raise table.TableError("all columns should have the same size")
                elem      = list(map(str.lower, elem))
                self.elem = np.array(elem, dtype = 'str')
                self._cols.append('elem')
            else:
                raise TypeError("'elem' must be a 1d array")

        self._nc, self._nr  = len(self._cols), __len
        self._subset_getter = table.TableSubsetGetter(self)
        self.table_row      = namedtuple('Line', self._colnames())

    def __repr__(self) -> str:
        return f"<LinesTable: {self.nr} lines'>"

    def _colnames(self) -> tuple:
        return self._cols
    
    def append(self, wavelen: float, aki: float, ek: float, gk: float, acc: float = ..., sp: int = ..., elem: int = ..., ) -> None:
        """
        Append a line data to the table.
        """
        if max(map(lambda o: np.ndim(o), [wavelen, aki, gk, ek, acc, sp, elem])):
            raise TypeError("all arguments must be scalars")
        values = {
                    'wavelen': wavelen, 
                    'aki'    : aki, 
                    'ek'     : ek, 
                    'gk'     : gk, 
                    'intens' : 0.0,
                 }
        if self.hascolumn('sp'):
            values['sp'] = sp
        else:
            raise table.TableError("table has no column called 'sp'")
        if self.hascolumn('elem'):
            values['elem'] = elem
        else:
            raise table.TableError("table has no column called 'elem'")
        if self.hascolumn('acc'):
            values['acc']  = acc
        else:
            raise table.TableError("table has no column called 'acc'")
        return super().append(**values)

    def getIntensities(self, Te: float, ns: Union[Sequence[float], float] = 1.0):
        """
        Fill the `intens` column with calculated intensity values.
        """
        if Te < 0.0:
            raise ValueError("Te must be positive")
        if np.ndim(ns):
            if len(ns) != self.nr:
                raise TypeError("ns must have the same size as other columns")
        
        population  = np.exp(-self.ek / Te) * ns
        self.intens = population * self.gk * self.aki * (1239.84198 / self.wavelen) / (4*np.pi) 

class LevelsTable(table.Table):
    """
    Store the data of energy levels of a species.
    """
    __slots__ = 'g', 'value', 'sp', 'elem', '_cols', 'acc', 
    __name__  = 'LevelsTable'

    def __init__(self, g: Sequence[int], value: Sequence[float], acc: Sequence[float] = None, sp: int = ..., elem: str = ...) -> None:
        self._cols = ['g', 'value']

        if np.ndim(g) != 1 or np.ndim(value) != 1:
            raise TypeError("positional aruments must be arrays")
        __len = len(g)
        if len(value) != __len:
            raise table.TableError("all columns should have the same size")
        self.g     = np.array(g,     dtype = 'float')
        self.value = np.array(value, dtype = 'float')

        self.acc = None
        if acc is not None:
            if np.ndim != 1:
                raise TypeError("positional aruments must be arrays")
            elif len(acc) != __len:
                raise table.TableError("all columns should have the same size")
            self.acc = acc
            self._cols.append('acc')

        self.sp, self.elem  = sp, elem
        self._nc, self._nr  = len(self._cols), __len
        self._subset_getter = table.TableSubsetGetter(self)
        self.table_row      = namedtuple('Level', self._colnames())

    def __repr__(self) -> str:
        return f"<LevelsTable: {self.nr} levels'>"

    def _colnames(self) -> tuple:
        return self._cols
    
    def append(self, g: float, value: float, acc: float = ...) -> None:
        """
        Append a line data to the table.
        """
        if max(map(lambda o: np.ndim(o), [g, value, acc])):
            raise TypeError("all arguments must be scalars")
        values = { 'g': g, 'value': value, }
        if self.hascolumn('acc'):
            values['acc']  = acc
        else:
            raise table.TableError("table has no column called 'acc'")
        return super().append(**values)

    def z(self, T: Any) -> Any:
        """
        Compute the partition function. 
        """
        T = np.asarray(T).flatten() 
        z = np.exp(-self.value / T[:, None]) @ self.g
        return z[0] if len(T) == 1 else z

class Plasma:
    """
    A plasma object.
    """
    __slots__ = 'spectab', 'linetab', 'ztab', 'Te', 'Ne'
    __name__  = ''

    Element = namedtuple('Element', ['key', 'nspec', 'm', 'eion', 'levels']) # to store an element

    # species data: stores the element, charge, ionization energy and population 
    # (fraction) and mass of the species' of the components.  
    SpeciesTable = table.table('SpeciesTable', ['elem', 'sp', 'mass', 'eion', 'ns', 'z'])

    def __init__(self, *args, **kwargs) -> None:
        self.spectab : Plasma.SpeciesTable 
        self.linetab : LinesTable          
        self.ztab    : CubicSpline         
        
        self.Te, self.Ne = None, None

    def _components(self) -> tuple:
        ...

    @property
    def components(self) -> tuple:
        """ Get keys of the components of the plasma. """
        return self._components()

    def set(self, Te: float = ..., Ne: float = ...) -> None:
        """ Set the plasma conditions. """
        if Te is not ... :
            if not Te > 0.0:
                raise ValueError("Te (temperature) must be positive")
            self.Te = Te
        if Ne is not ... : 
            if not Ne > 0.0:
                raise ValueError("Ne (electron number density) must be positive")
            self.Ne = Ne

    def _configure(self) -> None:
        """ Configure the plasma with current settings. """
        if self.Te is ... or self.Ne is ... :
            raise ValueError("plasma is not setup")
            
        self.spectab.z = self.ztab(self.Te)



def element(key: str, nspec: int, m: float, eion: Sequence[float], levels: Sequence[LevelsTable]) -> Plasma.Element:
    """
    Store data for an element.
    """
    if not isinstance(key, str):
        raise TypeError("key must be an 'str' object")
    key = key.lower()

    if not isinstance(nspec, int):
        raise TypeError("nspec must be an 'int' object")
    elif nspec < 1:
        raise ValueError("nspec must be positive")

    if not isinstance(m, float):
        raise TypeError("m must be an 'float' object")
    elif m <= 0:
        raise ValueError("m must be positive")

    if np.ndim(eion) != 1:
        raise TypeError("eion must be an 1d array")
    elif len(eion) != nspec:
        raise TypeError("eion must have size `nspec`")
    eion = list(eion)

    if np.ndim(levels) != 1:
        raise TypeError("levels must be an 1d array")
    elif len(levels) != nspec:
        raise TypeError("levels must have size `nspec`")
    elif not min(map(lambda o: isinstance(o, LevelsTable), levels)):
        raise TypeError("levels must be a list of 'LevelsTable' objects")
    levels = list(levels)

    return Plasma.Element(key, nspec, m, eion, levels)

def plasma(name: str, elems: Sequence[Plasma.Element], linetab: LinesTable, Tspace: space = space(0.1, 3.0, 11)) -> Type[Plasma]:
    """
    Create a plasma type. 
    """
    if not isinstance(elems, list):
        raise TypeError("elems must be a list")
    elif not min(map(lambda o: isinstance(o, Plasma.Element), elems)):
        raise TypeError("elems must be a list of 'Plasma.Element' objects")
    _nelems = len(elems)

    if not isinstance(linetab, LinesTable):
        raise TypeError("linetab must be a 'LinesTable'")
    elif not min(map(linetab.hascolumn, ['elem', 'sp'])):
        raise TypeError("linetab must have 'elem' and 'sp' columns")

    slots = []
    T     = np.linspace(Tspace.start, Tspace.stop, Tspace.num)

    # create the element/species table
    key, sp, m, eion, ztab = [], [], [], [], []
    for __elem in elems:
        if __elem.key in key:
            raise ValueError("elements must have different keys")
        if __elem.key not in linetab.elem:
            warnings.warn(f"element key {__elem.key} not present in linetab")
        
        slots.append(__elem.key)

        key  = key  + list( repeat( __elem.key, __elem.nspec ) )
        m    = m    + list( repeat( __elem.m,   __elem.nspec ) )
        eion = eion + list( __elem.eion )
        sp   = sp   + list( range( 1, __elem.nspec+1 ) )

        for __lvtab in __elem.levels:
            ztab.append( __lvtab.z(T) )

    spectab = Plasma.SpeciesTable( 
                                    elem = key, 
                                    sp   = sp, 
                                    mass = m, 
                                    eion = eion, 
                                    ns   = np.zeros_like(m),
                                    z    = np.zeros_like(m),
                                )
    # print(np.stack(ztab, 1))
    ztab    = CubicSpline(T, np.stack(ztab, 1))

    def _init(self: Plasma, *args, **kwargs):
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
        if len(args) == len(self.__slots__)-1:
            x = 1.0 - sum(args.values())
            if x < 0.0:
                raise ValueError("total concentration should be 1")
            args[missing], missing = x, None
        if missing is not None:
            raise TypeError(f"missing value for argument '{missing}'")

        if sum(args.values()) != 1.0:
            raise ValueError("total concentration must be 1")
        
        for key, value in args.items():
            setattr(self, key, value)

        self.spectab = spectab
        self.linetab = linetab
        self.ztab    = ztab
        # self.Te, self.Ne = None, None

    def _components(self: Plasma) -> tuple:
        return self.__slots__
            
    
    return type(
                    name,
                    (Plasma, ),
                    {
                        '__name__'    : name,
                        '__slots__'   : tuple(slots),
                        '__init__'    : _init,
                        '_components' : _components,
                    }
                )





from numpy.random import uniform, choice, randint

x = element('x', 2, 10.0, [1.0, 3.0], [LevelsTable(g=randint(1, 3, 10), value=uniform(0,5,10)), LevelsTable(g=randint(1, 3, 10), value=uniform(0,5,10))])
y = element('y', 1, 14.0, [13.0, ], [LevelsTable(g=randint(1, 3, 10), value=uniform(0,5,10)), ])


l = LinesTable(
    uniform(400, 600, 10), uniform(1e+7, 1e+8, 10), uniform(0.0, 10.0, 10), randint(0, 5, 10), choice(['x', 'y'], 10), choice([1, 2], 10)
)

p = plasma('a', [x, y], l)
q = p(x=0.1,)
q.set(1.0, 1e+15)
