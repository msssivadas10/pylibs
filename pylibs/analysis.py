r"""

PyLIBS Analysis Module
======================

This module contains some functions for analysing plasma spectrum. Available features 
include calculation of Boltzmann coordinates and estimation of plasma conditions from 
a spectrum.

"""

import numpy as np
from pylibs.objects import LinesTable
from pylibs.plasma import Plasma
from typing import Any, Tuple

FIXED_TEMPERATURE = 0b01
FIXED_DENSITY     = 0b10

class AnalysisError(Exception):
    """ Base class of exceptions raised by members of analysis module. """
    ...

def boltzmannXY(lnt: LinesTable, mode: int = 0, I: Any = None, p: Plasma = None) -> Tuple[Any, Any]:
    r""" 
    Calculate the Boltzmann plot coordinates of a set of lines.

    Parameters
    ----------
    lnt: LinesTable
        A table of spectral lines.
    mode: int
        Specify the type of Boltzmann plot. 0 is for the usual one, 1 for the multi-
        species variant and 2 for the multi-element variant.
    I: array_like 
        Intensity values. If given use this values, otherwise take it from the table.
    p: Plasma
        Plasma object specifying the plasma model to use for mode 1 and 2 calculations.
    
    Returns
    -------
    X, Y: array_like
        X and Y coordinates of the Boltzmann plot.

    Examples
    --------
    todo

    """
    if not isinstance( lnt, LinesTable ):
        raise TypeError("lnt must be a 'LinesTable'")
    elif lnt.I is None and I is None:
        raise AnalysisError("no intensity data in the table") 

    if I is None:
        I = lnt.I
    else:
        I = np.asfarray( I )
        if np.ndim(I) != 1:
            raise AnalysisError("I should be a 1D array")
        elif I.shape[0] != lnt.nr:
            raise AnalysisError("I should have same size as the lines table")
    
    # Type-0 boltzmann plot
    Yb = np.log( I * lnt.wavelen / lnt.gk / lnt.aki )
    Xb = lnt.ek
    if mode == 0:
        return Xb, Yb 

    if p is None:
        raise AnalysisError("plasma object must be given for mode 1 or 2")
    elif not isinstance( p, Plasma ):
        raise AnalysisError("p must be 'Plasma' object")
    elif not p.locked:
        raise AnalysisError("plasma object is not locked: may cause data loss.") 

    const = 6.009E+21 * p.Te**1.5 / p.Ne
    
    et    = p.comp

    if lnt.elem is None or lnt.s is None:
        raise AnalysisError("lines table should have element and species columns")

    # Type-1 boltzmann plot: species corrections
    Vs = np.asfarray([ 
                        ( 0.0 if s == 0 else et.child(elem).species( int(s-1) ).Vs ) for elem, s in zip( lnt.elem, lnt.s ) 
                     ]) # X correction

    Bs = lnt.s * np.log(const) # Y correction

    Xb, Yb = Xb + Vs, Yb - Bs
    if mode == 1:
        return Xb, Yb

    # Type-2 boltzmann plot: element corrections
    D = {}
    for elem in np.unique( lnt.elem ):
        __elem = et.child( elem )

        Us   = __elem.species(0).U( p.Te )
        Usp1 = __elem.species(1).U( p.Te )
        Vs   = __elem.species(0).Vs
        yp1  = 1 + const * ( Usp1 / Us ) * np.exp( - Vs / p.Te )
        
        D[ elem ] = np.log( __elem.Nx / Us / yp1 )

    Yb = Yb - np.asfarray([ D[ elem ] for elem in lnt.elem ])

    if mode == 2:
        return Xb, Yb

    raise AnalysisError("mode should be a number between 0 and 2")
    
def boltzmann(lnt: LinesTable, mode: int = 0, p: Plasma = None) -> None:
    r""" 
    Calculate the Boltzmann plot coordinates of a set of lines. This will not return any 
    output, but update the given lines table.

    Parameters
    ----------
    lnt: LinesTable
        A table of spectral lines. Use the intensities from this table.
    mode: int
        Specify the type of Boltzmann plot. 0 is for the usual one, 1 for the multi-
        species variant and 2 for the multi-element variant.
    p: Plasma
        Plasma object specifying the plasma model to use for mode 1 and 2 calculations.
    
    Examples
    --------
    todo

    """
    boltzX, boltzY = boltzmannXY( lnt, mode, None, p )
    return lnt.setBoltzmannXY( boltzX, boltzY )

def optimizePlasma(lnt: LinesTable, p: Plasma, flags: int = FIXED_DENSITY) -> None:
    """
    Optimize the plasma conditions to match with the given lines intensities. This is done 
    minimising the spread of the Boltzmann coordinates from a straight line.

    Parameters
    ----------
    lnt: LinesTable:
        Data for the spectral lines are taken from this table. Must have the intensity column.
    p: Plasma
        Plasma object setup at an initial 'guess' condition. This must be *locked* to prevent 
        any data loss. After successful minimization, this object will be set at the optimum 
        condition.
    flags: int, optional
        Special flags tell whether to fix the temperature of density as constants. The flag 
        `FIXED_DENSITY` is to fix electron density (default) and `FIXED_TEMPERATURE` is to 
        fix temperature.

    Examples
    --------
    todo
    
    """
    if not isinstance(lnt, LinesTable):
        raise TypeError("lnt must be a 'LinesTable'")
    elif not isinstance(p, Plasma):
        raise TypeError("p must be a 'Plasma' object")
    elif not isinstance(flags, int):
        raise AnalysisError("flags must be 'int'")

    from scipy.optimize import minimize

    def prepareVector(__p: Plasma) -> Any:
        x = []
        if not flags & FIXED_TEMPERATURE:
            x.append( __p.Te )
        if not flags & FIXED_DENSITY:
            x.append( __p.Ne )
        for Cx in __p.composition.values():
            x.append( Cx )
        return np.asfarray( x[:-1] )

    def applyVector(__p: Plasma, __x: Any) -> None:
        i      = 0
        Te, Ne = ... , ...
        if not flags & FIXED_TEMPERATURE:
            Te = __x[i]
            i += 1
        if not flags & FIXED_DENSITY:
            Ne = __x[i]
            i += 1
        __p.setComposition( **dict( zip( __p.components, __x[i:] ) ) )
        return __p.setup( Te, Ne )

    def cost(x: Any, __p: Plasma, __lnt) -> float:
        applyVector( __p, x )
        bx, by           = boltzmannXY( __lnt, 2, None, __p )
        slope, intercept = np.polyfit( bx, __p.Te * by, deg = 1 )
        return np.sum( ( by - ( slope * bx + intercept) )**2 )
    
    xopt = minimize(
                        cost,
                        prepareVector( p ),
                        args = ( p, lnt )
                   )

    return applyVector( p, xopt.x )

def findPeaks(*args, **kwargs) -> Any:
    """
    Find the peaks in a series.
    """
    raise NotImplementedError("function not implemented")

def matchLines(*args, **kwargs) -> None:
    """
    Match a set of lines with another.
    """
    raise NotImplementedError("function not implemented")

