from typing import Any, Type
import numpy as np
try:
    import _tables as tbl
    import _elemtree as et
except Exception:
    from . import _tables as tbl
    from . import _elemtree as et

class Plasma:
    """
    A plasma object.
    """
    __slots__ = 'comp', 'Te', 'Ne', 
    __name__  = ''

    def __init__(self, *args, **kwargs) -> None:
        self.comp : et.tree.Node
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

    @property
    def lines(self) -> tbl.LinesTable:
        """ Get a table of all lines in the components. """
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
        
        o = tbl.LinesTable(elem = [], s = [])
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

    def getSpectrum(self, x: Any, res: float = 500) -> tbl.table.Table:
        """
        Generate the LTE plasma spectrum at the specified conditions.
        """
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

        tb_spectrum = tbl.table.table(
                                        '{}_spectrum'.format(self.__name__),
                                        list( spec.keys() )
                                     )

        return tb_spectrum( **spec )
            
def plasma(name: str, comp: list) -> Type[Plasma]:
    """
    Create a specific plasma class.
    """
    # create a tree from the components
    ct     = et.elementTree(comp)
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

    
    
        