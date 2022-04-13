r"""

PyLIBS Plasma Module
======================

The plasma module `pylibs.plasma` cna be used to model a plasma. A plasma model is 
represented by an instance of :class:`Plasma`, which stores the plasma conditions 
as its attributes. These conditions are the plasma temperature :math:`T_e`, electron 
number density :math:`N_e` and the plasma composition. A plasma of some mixture of 
elements is represented by a plasma type, a subclass of :class:`Plasma`. This then 
initialised at different combinations of compositions to create a plasma model. One 
can use the method `plasma.plasma` to create a plasma type.

Eg., To create a copper-tin alloy plasma, one need the spectroscopic data of copper and 
tin. This must be arranged as a *tree* for the plasma object to use. Simples way is to 
arrange them as a `dict` of a special format (see documentation of `plasma.plasma`). If 
`t` is the element data tree, then

>>> copper_tin = plasma( 'copper_tin', t )

This can be initialised to make, for example a 70% copper and 30% tin plasma as 

>>> p1 = copper_tin(70.0, 30.0) # or simply copper_tin(70.0)

This object can be then used for generating an ideal plasma spectrum or as a model for 
analysing a real life spectrum using the `pylibs.analysis` module! 

"""

from typing import Any, Type, Iterable, Union
from pylibs.objects import LinesTable, ElementNode, elementTree
from pylibs.objects.tree import Node, node
import pylibs.objects.table as table
import numpy as np
import warnings

class PlasmaError(Exception):
    """ Base class of exceptions used by plasma objects. """
    ...

class PlasmaWarning(Warning):
    """ Base class for warning used by plasma objects. """
    ...

class Plasma:
    r"""
    Base class for all plasma classes. A `Plasma` object represents a plasma at LTE. 
    This object can be then used to get the composition, plasma spectrum etc. at 
    local thermal equilibrium (LTE). A plasma type can created using the function 
    `pylibs.plasma`, given the component elements data.

    Parameters
    ----------
    *args, **kwargs: Any
        Input parameters to initialise the plasma. Usually, they will be the composition 
        of plasma components as weight percentages. So, they must add to 100.

    Examples
    --------
    todo

    See Also
    --------
    plasma:
        Function to create a plasma class.

    """
    __slots__ = 'comp', 'Te', 'Ne', '_laststate', '_lnt'
    __name__  = ''

    def __init__(self, *args, **kwargs) -> None:
        self.comp : Node
        self._lnt : Node
        self.Te   : float
        self.Ne   : float

        self._laststate: bool

    def __repr__(self) -> str:
        return '{}({})'.format(
                                    self.__name__,
                                    ', '.join(
                                                map( 
                                                        lambda o: f'{o[0]}={o[1]}', self.composition.items() 
                                                    )
                                            )
                               )
    
    def _comp(self) -> tuple:
        ...

    def lock(self, value: bool = True) -> None:
        """ Lock the plasma. i.e., No changes are made to the tables. """
        if value: 
            if self.locked:
                return

            # lock the plasma: get the current settings and store it
            etree = {}
            for __elem in self.comp.children():
                e = { 
                        'Nx'     : __elem.Nx, 
                        'species': [],
                    }
                for __s in __elem.children():
                    e['species'].append(
                                            {
                                                'Ns': __s.Ns,
                                                'Us': __s.Us,
                                            }
                                        )
                etree[ __elem.key ] = e
            self._laststate = { 
                                    'Te'   : self.Te, 
                                    'Ne'   : self.Ne, 
                                    'etree': etree, 
                              }
            return
        
        if not self.locked:
            return 

        # unlock the plasma: reset current settings
        self.Te = self._laststate[ 'Te' ]
        self.Ne = self._laststate[ 'Ne' ]
        for key, value in self._laststate[ 'etree' ].items():
            self.comp.child( key ).Nx = value[ 'Nx' ]
            for s, spec in enumerate( value[ 'species' ] ):
                self.comp.child( key ).species( s ).Ns = spec[ 'Ns' ]
                self.comp.child( key ).species( s ).Us = spec[ 'Us' ]

                self.comp.child( key ).species( s ).T  = self._laststate[ 'Te' ]
        self._laststate = None
        return

    def unlock(self) -> None:
        """ Unlock the plasma. """
        return self.lock(False)

    def setComposition(self, *args, **kwargs) -> None:
        """ Set plasma composition. """
        ...

    def setup(self, Te: float = ..., Ne: float = ..., warn: bool = False) -> None:
        r""" 
        Set the plasma conditions. This will initialise a plasma and calculate the 
        composition and line intensities at LTE.

        Parameters
        ----------
        Te: float
            Plasma temperature in eV. Must be a non-zero positive value.
        Ne: float
            Electron density in the plasma in :math:`{\rm cm}^{-3}`
        warn: bool, optional
            If set, print warning messages, if any.
        """
        if Te is not ... :
            if not Te > 0.0:
                raise ValueError("temperature 'Te' must be positive")
            self.Te = Te
        if Ne is not ... : 
            if not Ne > 0.0:
                raise ValueError("electron number density 'Ne' must be positive")
            self.Ne = Ne
        self._getComposition( warn )
        self._getIntensity( warn )

    def _getComposition(self, warn: bool = False) -> None:
        """ 
        Calculate the composition at LTE. 
        """
        if self.Te is None or self.Ne is None:
            raise PlasmaError("plasma is not setup")
        for __elem in self.comp.children():
            __elem.getLTEComposition(self.Te, self.Ne)
    
    def _getIntensity(self, warn: bool = False) -> None:
        """
        Calculate intensities at LTE.
        """
        if self.locked:
            if warn:
                warnings.warn("cannot compute line intensities: plasma is locked", PlasmaWarning)
            return 
        for __elem in self.comp.children():
            for __s in __elem.children():
                I = __s.getLTEIntensities()
                self._lnt.child( __elem.key ).child( __s.key ).lines.setLineIntensity( I )

    def _makeLinesTree(self) -> None:
        """
        Create a copy of lines data as tree.
        """
        def copylines(lnt: LinesTable):
            cp = LinesTable( lnt.wavelen, lnt.aki, lnt.ek, lnt.gk, lnt.elem, lnt.s, lnt.errAki )
            if lnt.boltzX is not None and lnt.boltzY is not None:
                cp.setBoltzmannXY( lnt.boltzX, lnt.boltzY )
            if lnt.I is not None:
                cp.setLineIntensity( lnt.I )
            return cp

        linenode = node( 'linesnode', [ 'key', 'lines' ] )

        root = Node()
        for __elem in self.comp.children():
            en   = linenode( __elem.key, None )
            for __s in __elem.children():
                en.addchild( linenode( __s.key, copylines( __s.lines ) ) )
            root.addchild( en, en.key )
        self._lnt = root
        
    def getSpectrum(self, x: Any, res: float = 500) -> table.Table:
        r"""
        Generate the LTE plasma spectrum at the specified conditions.

        Parameters
        ----------
        x: array_like
            Wavelengths to calculate the intensity. Must be in nanometer units.
        res: float, optional
            Resolution parameter. Must be a non-zero positive number. Width of a 
            line at wavelength `wavelen` is then calculated as `wavelen / res`. Its 
            default value is 500.

        Returns
        -------
        spectrum_t: Table
            Output spectrum as a table. It contains the spectrum of all the component 
            species (column `{element_key}_{species_key}`) and their total (`y`) and 
            wavelength (`x`, same as the input).
            
        """
        # if self.locked:
            # raise PlasmaError("cannot compute spectrum: plasma is locked")

        if self.Te is None or self.Ne  is None:
            raise PlasmaError("plasma is not setup.")

        def gaussian(x: Any) -> Any:
            return np.exp(-4*np.log(2) * x**2)

        # sampling the spectrum using a gaussian shape
        x           = np.asfarray(x).flatten()
        spec, total = {'x': None, 'y': None}, np.zeros_like(x)
        for __elem in self._lnt.children():
            for __s in __elem.children():
                w      = __s.lines.wavelen / res
                I      = gaussian( (x[:,None] -__s.lines.wavelen) / w ) @ __s.lines.I
                total += I

                spec[ '{}_{}'.format(__elem.key, __s.key) ] = I
        
        spec[ 'y' ] = total
        spec[ 'x' ] = x

        tb_spectrum = table.table(
                                        '{}_spectrum'.format(self.__name__),
                                        list( spec.keys() )
                                 )

        return tb_spectrum( **spec )
    
    @property
    def lines(self) -> LinesTable:
        r""" 
        Get a table of all lines in the components. 
        """
        # go through the tree to check if the tables has optional columns
        hasI, hasXY = True, True

        lines = []
        for __enode in self._lnt.children():
            for __snode in __enode.children():
                o = __snode.lines
                if o.I is None:
                    hasI = False
                if o.boltzX is None or o.boltzY is None:
                    hasXY = False
                lines.append(o)
        
        o = LinesTable(elem = [], s = [])
        if hasI:
            o.setLineIntensity([])
        if hasXY:
            o.setBoltzmannXY([], [])
        for _lines in lines:
            o.join( _lines )
        return o

    @property
    def components(self) -> tuple:
        """ Plasma components. """
        return self._comp()

    @property
    def composition(self) -> dict:
        """ Plasma composition. """
        return dict(
                        map(
                                lambda o: (o, getattr(self, o)), self.components
                           )
                   )
    
    @property
    def N(self) -> Any:
        """ 
        Composition matrix. Each row of the matrix correspond to a component element 
        (in the order given by `components` property) and columns correspond to the 
        species. 
        """
        if self.Te is None or self.Ne  is None:
            raise PlasmaError("plasma is not setup.")
        N = []
        for __elem in self.comp.children():
            Ns = []
            for __s in __elem.children():
                Ns.append( __s.Ns )
            N.append( Ns )
        
        nelem, ns = len(N), max( map( len, N ) )
        
        Nmatrix   = np.zeros( (nelem, ns) )
        for i in range(nelem):
            for j in range( len(N[i]) ):
                Nmatrix[i,j] = N[i][j]
        return Nmatrix

    @property
    def locked(self) -> bool:
        return (self._laststate is not None)

def plasma(name: str, comp: Union[Iterable[ElementNode], dict, Node]) -> Type[Plasma]:
    r"""
    Create a specific plasma class. It can be used to create specific plasma types, 
    given the components. Then, this type can be initialised with various combinations 
    of concentrations to create a plasma.

    Parameters
    ----------
    name: str 
        Name for the new plasma type.
    comp: sequence of ElementNode, dict, Node
        A sequence of component elements. Each element is represented by a `Node` object 
        with child species nodes containing required data. It can also be a dict of special 
        format or a tree of element nodes.

    Returns
    -------
    plasma_t: type
        New plasma object.

    Notes
    -----
    If the input argument `comp` is a dict, then the top level key-value pairs should correspond 
    to element key-data pairs (i.e., an `str` key and `dict` data). Each element data should be 
    a dict with fields `m` (`float` atomic mass) and `species` for species data. Each species data 
    should be a dict with fields `Vs` (`float` ionization energy), `levels` (energy level table) 
    and `lines` (spectral lines table as a `LinesTable`). These lines could be a single table of 
    all lines of these elements. In that case, lines of the corresponding species/element are 
    filtered out.

    Examples
    --------
    To create a copper-tin plasma (element data stored in the dict `t`) at a ratio of 70:30, 

    >>> ctplasma = plasma( 'ctplasma', t )
    >>> cu7sn3   = ctplasma( cu = 70.0, sn = 30.0 )

    It can also initialized as `ctplasma(70, 30)` or `ctplasma(70)` or `ctplasma(sn = 30)`. But, 
    total composition should be 100 always.


    """
    # create a tree from the components
    ct     = elementTree(comp)
    _slots = tuple([ o.key for o in ct.children() ]) 

    def _init(self: Plasma, *args, **kwargs) -> None:
        self.comp       = ct
        self.Te         = None
        self.Ne         = None
        self._laststate = None

        self.setComposition(*args, **kwargs)
        self._makeLinesTree()

    def _setComposition(self, *args, **kwargs) -> None:
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
            print(args)
            raise ValueError("total concentration must be 100")

        # convert values from weight to number and store it at 
        # element nodes. use weights at plasma
        total = 0.0         
        for key, value in args.items():
            setattr(self, key, value)
            args[key] = value / self.comp.child(key).m
            total    += args[key]

        for key, value in args.items():
            self.comp.child(key).Nx = value / total

    def _comp(self: Plasma) -> tuple:
        return self.__slots__

    return type(
                    name,
                    (Plasma, ),
                    {
                        '__slots__'     : _slots,
                        '__init__'      : _init,
                        '__name__'      : name,
                        '_comp'         : _comp,
                        'setComposition': _setComposition,
                    }
                )

