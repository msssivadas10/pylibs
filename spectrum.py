#!/usr/bin/python3
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
from objects import Element, ElementSpecifier
from utils import Table, shape_
from typing import Any

class Spectrum:
    """
    Object to store and analyse a spectrum. A spectrum stores the wavelength values and the corresponding intensity values as two lists/arrays.

    Parameters
    ----------
    wavelen: array_like
        Wavelength in some unit (like nm or angstrum).
    intens: array_like
        Intensity data.
    attrs: dict, optional
        Python dict of custom informations about the spectrum.

    Attributes
    ----------
    wavelen: array_like
        Wavelength data.
    intens: array_like
        Intensity data.

    """
    __slots__ = 'x', 'y', '_transform', 'attrs'

    def __init__(self, wavelen: Any, intens: Any, attrs: dict = {}) -> None:
        x = np.asarray(wavelen).flatten()
        y = np.asarray(intens).flatten()
        if x.shape[0]  != y.shape[0]:
            raise ValueError("wavelength and intensity data should have same size")

        # transformation to make y data in range [0, 1]
        shift = y.min()
        scale = 1. / (y.max() - shift)
        self._transform = (shift, scale)

        self.x, self.y = x, (y - shift) * scale
        self.attrs     = attrs

    def _wavelen(self, ) -> Any:
        """ Wavelength """
        return self.x
        
    def _intens(self, ) -> Any:
        """ Intensity """
        shift, scale = self._transform
        return self.y / scale + shift

    def __getattr__(self, name: str) -> Any:
        if name == 'wavelen':
            return self._wavelen()
        elif name == 'intens':
            return self._intens()
        else:
            return super().__getattr__(name)

    def __repr__(self) -> str:
        return f"<Spectrum: size = {self.x.shape[0]}>"

    def _find_peaks(self, prominence: float) -> Any:
        """ Find peaks in itensity """
        if not 0. <= prominence <= 1.:
            raise ValueError("prominence value should be between 0 and 1")

        _ppos, _ = find_peaks(self.y, prominence = prominence)
        return _ppos

    def _fit_line(self, pos: int) -> tuple:
        """ Fit a shape to the line at given position """
        xpos, ypos = self.wavelen[pos], self.intens[pos]
        _ycut, n   = ypos * 0.5, self.x.shape[0]
        
        # # find the left end of the data
        _left  = pos
        while _left >= 0 and self.intens[_left] > _ycut:
            _left -= 1
            if self.intens[_left+1] - self.intens[_left] < 0.:
                break

        # # find the right end of the data
        _right  = pos
        while _right < n and self.intens[_right] > _ycut:
            _right += 1
            if self.intens[_right-1] - self.intens[_right] < 0.:
                break
        
        if pos - _left < 2 or _right - pos < 2:
            print(_left, _right, pos)
            raise ValueError("not enough data to fit")

        wpos = self.wavelen[_right] - self.wavelen[_left] # width
        
        popt, _ = curve_fit(
                                shape_['gaussian'], 
                                self.wavelen[_left:_right+1], 
                                self.intens[_left:_right+1], 
                                p0 = [xpos, ypos, wpos]
                            )
        return list(popt)

    def searchPeaks(self, prominence: float = 0.005) -> Any:
        """
        Search for peaks in the intensity data. This will fit the line with a gaussian function to get an accurate estimate of the line shape parameters. 

        Parameters
        ----------
        prominence: float
            Value of prominence to find the peaks.

        Returns
        -------
        line_param: array_like
            Fit parameters to the lines found (and matched).

        """
        _ppos = self._find_peaks(prominence)
        
        # find fits for each line
        line_param = np.asarray(list(map(self._fit_line, _ppos)))
        return line_param


class BoltzError(Exception):
    """ Exception used by Boltzmann plane objects """
    ...


class BoltzPlane:
    """
    A object representing the Boltzmann plane. 
    """
    __slots__ = 'elems', 'data', 'fmt', '_elem_idx', 'Te', 'Ne'

    def __init__(self, elems: list, tab: Table, Te: float, Ne: float, fmt: str = 'boltz', tab_fmt: str = 'lines') -> None:
        if not min(map(lambda _x: isinstance(_x, Table), elems)):
            raise BoltzError("`elems` should be an iterable of 'Element' objects")
        self.elems = elems

        self._elem_idx = {_el.sym: i for i, _el in enumerate(elems)}

        if not isinstance(tab, Table):
            raise TypeError("`tab` is not a 'Table'")
        if tab_fmt == 'lines':
            if not tab.has_columns(['elem', 'sp', 'lambda', 'Aki', 'Ek', 'gk', 'int']):
                raise BoltzError("given table has not all lines data")
            # compute x and y
        elif tab_fmt == 'points':
            if not tab.has_columns(['elem', 'sp', 'x', 'y']):
                raise BoltzError("given table has not all points data")
            self.data = tab
        else:
            raise ValueError(f"invalid value for `{tab_fmt}` table format")
        
        if not fmt in ['boltz', 'saha']:
            raise ValueError(f"invalid format `{fmt}`")

        self.Te, self.Ne = Te, Ne

    def _boltz_coords(self, _int: Any, _aki: Any, _ek: Any, _gk: Any, ) -> tuple:
        """ Compute coordinates on the Boltzmann plane """
        x = _ek
        y = np.log(_int / _gk / _aki)

        if self.fmt == 'boltz':
            return x, y

        const = -np.log(6.009e+21 * self.Te**1.5 / self.Ne)

        def _xcorr(sym: str, _id: int):
            return 0 if _id == 1 else self.comp[self._elem_idx[sym]].species[_id-1].eion

        def _ycorr(_id: int):
            return 0 if _id == 1 else const
        
        x = x + np.array(list(map(_xcorr, self.data.c('elem'), self.data.c('sp'))))
        y = y + np.array(list(map(_ycorr, self.data.c('sp'))))

        return x, y

    def _boltz_coords_table(self, tab: Table) -> tuple:
        """ Compute coordinates on the Boltzmann plane """
        return self._boltz_coords(tab.c('int'), tab.c('Aki'), tab.c('Ek'), tab.c('gk'))

    def setTe(self, val: float) -> None:
        """
        Set the value of temeperature (in eV).
        """
        self.Te = val
        return

    def setNe(self, val: float) -> None:
        """
        Set the value of electron density (in /cc).
        """
        self.Ne = val
        return


class AnalyserError(Exception):
    """ Exception used by :class:`QAnalyser` objects """
    ...


class QAnalyser:
    """
    An object to analyse the (LIBS) spectrum and get information.
    """
    __slots__ = ('spec', 'comp', '_elem_idx', '_lines', '_has_reflines', 
                 '_has_matched', '_found_lines', 'Ne', 'Te', '_bfmt')

    def __init__(self, spec: Spectrum, comp: list = []) -> None:
        self.spec      = spec

        self._elem_idx = {}     # mapping the element index to symbol
        self._lines    = None   # lines in this spectrum (and their data)
        self.Ne        = None   # (estimated) value of electron temperature
        self.Te        = None   # (estimated) value of electron density
        self._bfmt     = None   # boltzmann plane format

        # Flags : 
        self._has_reflines = False  # has set reference lines
        self._has_matched  = False  # has matched with reference lines
        self._found_lines  = False  # lines are found from the spectrum
        

        self.setPlasmaComponents(comp)

    def setPlasmaComponents(self, comp: list):
        """
        Set the components of the plasma, from which this spectrum is got.
        """
        if len(comp):
            if not min(map(lambda e: isinstance(e, Element), comp)):
                raise TypeError("all entries in the component list should be an 'Element' object")
        self.comp = comp

        for i, elem in enumerate(self.comp):
            self._elem_idx[elem.sym] = i
        return

    def setReferenceLines(self, linetab: dict, tol: float = 0.001) -> None:
        """
        Set the refernce lines to use for analysis.
        """
        if not isinstance(linetab, dict):
            raise TypeError("line table should be a 'dict'")
        for _lambda, es in linetab.items():
            sym, _id = ElementSpecifier(es).value()
            
            i = self._elem_idx[sym]
            if not self.comp[i].add_refline(_id, _lambda, tol):
                print(f"Warning: no matching line for {_lambda} is found in {sym}-{_id}")
        
        self._has_reflines = True
        return

    def getReferenceLines(self, ) -> Table:
        """
        Get the reference lines as a table.
        """
        if not self._has_reflines:
            raise AnalyserError("no reference lines are set")
        out = Table()
        for elem in self.comp:
            t = elem.get_reflines()
            if not t.shape[1]:
                continue
            t.insert_column(np.repeat(elem.sym, t.shape[0]), 0, 'elem', str)
            out = out + t
        return out

    def searchPeaks(self, prominence: float = 0.005, match: bool = False, dist_ub: float = 0.1) -> None:
        """
        Search for peaks in the spectrum and optionally, match with a set of referenece lines given.
        """
        out = self.spec.searchPeaks(prominence)
        
        self._lines = Table(out, ['lambda', 'int', 'width'], [float, float, float])
        self._found_lines = True

        # match with reference lines
        if match:
            self.matchPeaks(dist_ub)
        return 

    def matchPeaks(self, dist_ub: float = 0.1) -> None:
        """
        Match the lines in the spectrum with reference lines.

        Parameters
        ----------
        dist_ub: float, optional
            Upper bound to the distanceto find a matching value (default is 0.1).

        """
        if not self._found_lines:
            raise AnalyserError("lines in the spectrum is not yet found")

        _mylines  = self._lines.c('lambda0') if self._has_matched else self._lines.c('lambda')
        _myints   = self._lines.c('int')
        _mywidths = self._lines.c('width') 

        reflines = self.getReferenceLines()

        match_to = reflines.c('lambda')

        tree = KDTree(_mylines[:, np.newaxis])
        _, i = tree.query(match_to[:, np.newaxis], distance_upper_bound = dist_ub)

        # filter lines not matched
        filt  = (i != tree.n)
        i, j  = i[filt], np.where(filt)[0]
        
        reflines = reflines.subtable(j) # matched reference lines
        reflines.add_column(_mylines[i],  'lambda0', float)
        reflines.add_column(_myints[i],   'int',     float)
        reflines.add_column(_mywidths[i], 'width',   float)

        self._lines = reflines

        self._has_matched = True
        return

    def _boltz_coords(self, fmt: str = 'boltz', Te: float = ..., Ne: float = ...) -> tuple:
        """ Boltzmann plane coordinates (internal) """
        
        x = self._lines.c('Ek')
        y = np.log(self._lines.c('int') / self._lines.c('gk') / self._lines.c('Aki'))

        if fmt == 'boltz':
            return x, y

        if Te is Ellipsis or Ne is Ellipsis:
            raise AnalyserError("both Te and Ne should be given if fmt is`saha`")

        const = -np.log(6.009e+21 * Te**1.5 / Ne)

        def _xcorr(sym: str, _id: int):
            if _id == 1:
                return 0.
            return self.comp[self._elem_idx[sym]].species[_id-1].eion

        def _ycorr(sym: str, _id: int):
            if _id == 1:
                return 0.
            return const
        
        x = x + np.array(list(map(_xcorr, self._lines.c('elem'), self._lines.c('sp'))))
        y = y + np.array(list(map(_ycorr, self._lines.c('elem'), self._lines.c('sp'))))

        return x, y

    def computeBoltzCoords(self, fmt: str = 'boltz', Te: float = ..., Ne: float = ...) -> None:
        """
        Compute the Boltzmann plane coordinates from the matched lines data.
        """
        if not self._has_matched:
            raise AnalyserError("to compute Boltzmann plane coordinates, lines have to found and matched")
        if fmt not in ('boltz', 'saha'):
            raise ValueError(f"unknown format `{fmt}` for Boltzmann plane")

        x, y = self._boltz_coords(fmt, Te, Ne)

        self._bfmt = fmt

        if self._lines.has_columns(['x', 'y']):
            # replace is already computed the coordinates
            self._lines.replace_column('x', x)
            self._lines.replace_column('y', y)
            return
        
        # add if not computed 
        self._lines.add_column(x, 'x', float)
        self._lines.add_column(y, 'y', float)
        return

    def getBoltzCoords(self, elem: str = ..., sp: int = ...) -> Table:
        """
        Get the last computed Boltzmann plane coordinates.

        Parameters
        ----------
        elem: str, optional
            If given, get the coordinates for that element.
        sp: int, optional
            If given, along with the element, get the coordinates for that species.

        Returns
        -------
        tab: :class:`Table`
            Boltzmann plane coordinates as a table.

        """
        if not self._lines.has_columns(['x', 'y']):
            raise AnalyserError("Boltzmann coordinates not computed yet.")
        tab = Table(
                        {
                            'elem': self._lines.c('elem'),
                            'sp'  : self._lines.c('sp'),
                            'x'   : self._lines.c('x'),
                            'y'   : self._lines.c('y'),
                        },
                        coltypes = [str, int, float, float]
                    )
        if elem is not Ellipsis:
            i = np.where(tab.c('elem') == elem)[0]
            if sp is not Ellipsis:
                i = i[np.where(tab.c('sp')[i] == sp)[0]]
            return tab.subtable(i)
        return tab

    def estimateNe(self, *args, **kwargs) -> None:
        """
        Estimate the electron density in the plasma assuming LTE.  
        """
        return NotImplemented
        
    def estimateTe(self, use: str = ...) -> None:
        """
        Estimate the temperature from Boltzmann plots
        """
        btab = self.getBoltzCoords()

        if use is Ellipsis:
            use = ElementSpecifier(use)
            if use.is_element():
                raise AnalyserError("`use` should indicate a species")
            elem, sp = use.value()

            i    = np.where((btab.c('elem') == elem) & (btab.c('sp') == sp))[0]
            btab = btab.subtable(i)

        pass 


# ===========================================================================
# Test functions
# ===========================================================================

def _test_cu_sn_spectrum(cu: float = 0.7, sn: float = 0.3, Te: float = 1., Ne: float = 1e+17, _from: float = 350., _to: float = 650., res: int = 500):
    """
    The spectrum of a test plasma (copper-tin alloy) for test uses.

    Parameters
    ----------
    cu: float
        Copper abundance (default is 0.7).
    sn: float
        Tin abundance (default is 0.3).
    Te: float
        Temperature in eV (default is 1).
    Ne: float
        Electron number density in cm^{-3} (default is 1e+17).
    _from: float
        Minimum wavelength value. Must be greater than 100.
    _to: float
        Maximum wavelength value. Must be less than 1000.
    res: int, optional
        Resolution (number of samples will be 20 times this, qpprox.). Default is 500.

    Returns
    -------
    spectrum: :class:`Spectrum`
        Computed plasma spectrum.

    """
    from objects import _test_cu_sn_plasma

    p = _test_cu_sn_plasma(cu, sn, Te, Ne) # test plasma
    s = p.computeSpectrum(_from, _to, res) # spectrum

    info = {
                "comp"  : {
                             'copper': cu, 
                             'tin': sn
                          },
                "Te"    : Te,
                "Ne"    : Ne,
                "res"   : res,  
                "plasma": p,  
           }
    return Spectrum(wavelen = s.c('lambda'), intens = s.c('sum'), attrs = info)
