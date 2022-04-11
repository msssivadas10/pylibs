r"""

`pylibs.plasma`: Plasma Computations at LTE
===========================================

The plasma module contains the `Plasma` class, which is the base class for all plasma 
types. Instances of these type represent specific plasmas in ideal conditions. These 
are then used to calculate the distribution of species and plasma spectrum. All these 
calculations assume the plasma to be in llocal thermal equilibrium (LTE). 

The `plasma` function creates a plasma type with given components. These types are then 
initialised with different combinations to create a plasma model.

Notes
-----

All the calculations assumes the plasma to be at LTE. Species contributions are estimated 
using *Saha ionization equation* and line intensities are calculted assuming a *Boltzmann 
distribution* for the states. Each line will be given a Gaussian shape.

It also assumes the wavelengths to be in nanometers, energy and temperature in electron 
volts and transition rates in hertz. Electron densities are assumed to be in per cubic 
centimetre. 

"""

from typing import Any, Type, Iterable, Union
from pylibs.objects import LinesTable, ElementNode, elementTree
from pylibs.objects.tree import Node
import pylibs.objects.table as table
import numpy as np

class PlasmaError(Exception):
    """ Base class of exceptions used by plasma objects. """


class Plasma:
    r"""
    Base class for all plasma classes. A `Plasma` object represents a plasma at LTE. 
    This object can be then used to get the composition, plasma spectrum etc. at 
    local thermal equilibrium (LTE).

    Parameters
    ----------
    *args, **kwargs: Any
        Input parameters to initialise the plasma. Usually, they will be the composition 
        of plasma components as weight percentages.

    See Also
    --------
    table:
        Function to create a plasma class.

    """
    __slots__ = 'comp', 'Te', 'Ne', 
    __name__  = ''

    def __init__(self, *args, **kwargs) -> None:
        self.comp : Node
        self.Te   : float
        self.Ne   : float

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

    def setup(self, Te: float = ..., Ne: float = ...) -> None:
        r""" 
        Set the plasma conditions. This will initialise a plasma and calculate the 
        composition and line intensities at LTE.

        Parameters
        ----------
        Te: float
            Plasma temperature in eV. Must be a non-zero positive value.
        Ne: float
            Electron density in the plasma in :math:`{\rm cm}^{-3}`
        """
        if Te is not ... :
            if not Te > 0.0:
                raise ValueError("temperature 'Te' must be positive")
            self.Te = Te
        if Ne is not ... : 
            if not Ne > 0.0:
                raise ValueError("electron number density 'Ne' must be positive")
            self.Ne = Ne
        self._getComposition()
        self._getIntensity()

    def _getComposition(self) -> None:
        """ 
        Calculate the composition at LTE. 
        """
        for __elem in self.comp.children():
            __elem.getLTEComposition(self.Te, self.Ne)
    
    def _getIntensity(self) -> None:
        """
        Calculate intensities at LTE.
        """
        for __elem in self.comp.children():
            __elem.getLTEInntensities()

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
        if self.Te is None or self.Ne  is None:
            raise PlasmaError("plasma is not setup.")

        def gaussian(x: Any) -> Any:
            return np.exp(-4*np.log(2) * x**2)

        # sampling the spectrum using a gaussian shape
        x           = np.asfarray(x).flatten()
        spec, total = {'x': None, 'y': None}, np.zeros_like(x)
        for __elem in self.comp.children():
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
        for __enode in self.comp.children():
            o = __enode.lines
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
        format (see documentation of `pylibs.objects.elementTree` for the format) or a tree 
        of element nodes.

    Returns
    -------
    plasma_t: type
        New plasma object.

    """
    # create a tree from the components
    ct     = elementTree(comp)
    _slots = tuple([ o.key for o in ct.children() ]) 

    def _init(self: Plasma, *args, **kwargs) -> None:
        self.comp  = ct
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
                        '__slots__' : _slots,
                        '__init__'  : _init,
                        '__name__'  : name,
                        '_comp'     : _comp,
                    }
                )

