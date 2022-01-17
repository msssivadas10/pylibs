#!/usr/bin/python3
import numpy as np
import warnings
from .objects import Element, ElementSpecifier
from .utils import Table, shape_
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
from typing import Any

class Spectrum:
    """
    Object to store and analyse a spectrum. A spectrum stores the wavelength values and the 
    corresponding intensity values as two lists/arrays.

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
        Search for peaks in the intensity data. This will fit the line with a gaussian 
        function to get an accurate estimate of the line shape parameters. 

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
    """ 
    Exception used by Boltzmann plane objects 
    """
    ...

class BoltzWarning(Warning):
    """ 
    Warning used by Boltzmann plane objects 
    """
    ...

class BoltzPlane:
    """
    A object representing the Boltzmann plane. 
    """
    __slots__ = (
                    'elems', 'data', 'fmt', '_elem_idx', 'Te', 
                    'Ne', "_qa", "_fits", "Te_err", 
                )

    def __init__(self, elems: list, tab: Table, Te: float = ..., Ne: float = ..., fmt: str = 'boltz', tab_fmt: str = 'lines') -> None:
        self.Te, self.Ne = Te, Ne
        self.Te_err      = 0.

        if not fmt in ['boltz', 'saha']:
            raise ValueError(f"invalid format `{fmt}`")
        self.fmt = fmt

        if not min(map(lambda _x: isinstance(_x, Element), elems)):
            raise BoltzError("`elems` should be an iterable of 'Element' objects")
        self.elems = elems

        self._elem_idx = {_el.sym: i for i, _el in enumerate(elems)}

        if not isinstance(tab, Table):
            raise TypeError("`tab` is not a 'Table'")
        if tab_fmt == 'lines':
            if not tab.has_columns(['elem', 'sp', 'lambda', 'Aki', 'Ek', 'gk', 'int']):
                raise BoltzError("given table has not all lines data")
            x, y = self._boltz_coords(tab)
            self.data = tab[['elem', 'sp'], ]
            self.data.add_column(x, 'x', float)
            self.data.add_column(y, 'y', float)
        elif tab_fmt == 'points':
            if not tab.has_columns(['elem', 'sp', 'x', 'y']):
                raise BoltzError("given table has not all points data")
            self.data = tab
        else:
            raise ValueError(f"invalid value for `{tab_fmt}` table format")

        self._qa         = None # setted by a QAnalyser object   
        self._do_fitting()     

    def _boltz_coords(self, tab: Table) -> tuple:
        """ 
        Compute coordinates on the Boltzmann plane 
        """
        x = tab.c('Ek')
        y = np.log(tab.c('int') / tab.c('gk') / tab.c('Aki'))

        if self.fmt == 'boltz':
            return x, y

        if self.Te is ... or self.Ne is ... :
            raise BoltzError("Te and Ne should be given for saha-boltzmann plane")

        const = -np.log(6.009e+21 * self.Te**1.5 / self.Ne)

        def _xcorr(sym: str, _id: int):
            return 0 if _id == 1 else self.elems[self._elem_idx[sym]].species[_id-1].eion

        def _ycorr(_id: int):
            return 0 if _id == 1 else const
        
        x = x + np.array(list(map(_xcorr, tab.c('elem'), tab.c('sp'))))
        y = y + np.array(list(map(_ycorr, tab.c('sp'))))

        return x, y

    def _f(self, x: Any, islope: float, const: float) -> Any:
        """
        Line fitting the points of a species/element. It has the form `y ~ const - x / islope`,
        where `islope` is the negative inverse of the slope (i.e., the temperature in eV) and
        `const` is the intercept. This linear form is derived from Saha-Boltzmann equation.

        Parameters
        ----------
        x: array_like
            Independent variable or X coordinates of the points (energy level).
        islope: float
            Slope parameter for the line.
        const: float
            Y intercept for the line.

        Returns
        -------
        y: array_like
            Y coordinate of the points.

        """
        return const - x / islope

    def _fit_line_sp(self, elem: str, sp: int = ...) -> tuple:
        """
        Fit the points to a straight line.
        """
        pts  = self.getPoints(elem, sp)
        if not pts.shape[0]: # empty table
            return [None, None], [None, None]

        popt, pcov = curve_fit(self._f, pts.c('x'), pts.c('y'), )
        return popt, np.sqrt(np.diag(pcov))

    def _do_fitting(self, ) -> None:
        """
        Fit lines to points on the plane.
        """
        t = Table(
                    colnames = ['elem', 'Te', 'const', 'err_Te', 'err_const'],
                    coltypes = [str, float, float, float, float]
                 )
        if self.fmt == 'boltz':
            t.insert_column([], 1, 'sp', int) # insert a column after 'elem'

            for elem in self.elems:
                for sp in range(elem.nspecies):
                    (Te, const), (err_Te, err_const) = self._fit_line_sp(elem.sym, sp)
                    if Te is None:
                        continue
                    # print([type(x) for x in [elem.sym, sp, Te, const, err_Te, err_const]])
                    t.add_row([elem.sym, sp, Te, const, err_Te, err_const])
        else:
            # fmt is 'saha' - consider all species of an element
            for elem in self.elems:
                (Te, const), (err_Te, err_const) = self._fit_line_sp(elem.sym)
                if Te is None:
                    continue
                # print([type(x) for x in [elem.sym, Te, const, err_Te, err_const]])
                t.add_row([elem.sym, Te, const, err_Te, err_const])
        self._fits = t
        return

    def _filter_by_keys(self, tab: Table, elem: str = ..., sp: int = ...) -> Table:
        """
        Filter the given table by the keys.
        """
        out = tab
        i = np.arange(out.shape[0])
        if elem is not ... :
            i = i[np.where(out.c('elem') == elem.capitalize())[0]]
        if sp is not ... :
            i = i[np.where(out.c('sp')[i] == sp)[0]]
        i = list(i)
        return out.subtable(i)

    def getLineParam(self, elem: str = ..., sp: int = ...) -> Any:
        """
        Get the fitting parameters of a line.
        """
        return self._filter_by_keys(self._fits, elem, sp)

    def setTe(self, val: float) -> None:
        """
        Set the value of temeperature (in eV).
        """
        self.Te, self.Te_err = val, 0.
        return

    def setNe(self, val: float) -> None:
        """
        Set the value of electron density (in /cc).
        """
        self.Ne = val
        return

    def estimateTe(self, use: str = 'av') -> None:
        """
        Estimate temperature from boltzmann plane points.
        """
        if use == "av":
            t = self._filter_by_keys(self._fits)
        else:
            elem, sp = ElementSpecifier(use).value()
            if self.fmt == 'boltz':
                if sp is None:
                    raise BoltzError("species should be specified to estimate Te")
            else:
                if sp is not None:
                    warnings.warn("species value not used for `saha` format", BoltzWarning)
                sp = ...

            t  = self._filter_by_keys(self._fits, elem, sp)

        self.Te     = np.mean(t.c('Te'))
        self.Te_err = np.std(t.c('err_Te'))
        return

    def getPoints(self, elem: str = ..., sp: int = ...) -> Table:
        """
        Get the points in the Boltzmann plane. If given, return those for the specied species.

        Parameters
        ----------
        elem: str, optional
            Symbol of the element. If given only give values correspond to this element.
        sp: int, optional
            Species index. If given only give values correspond to this species.

        Returns
        -------
        pts: :class:`Table`
            Points on the Boltzmann plane. This table will have four columns - `x`, `y`, 'elem` and `sp` for x and y coordinates, element symbol and species index respectively. 

        """
        return self._filter_by_keys(self.data, elem, sp)


class AnalyserError(Exception):
    """ 
    Exception used by :class:`QAnalyser` objects 
    """
    ...


class QAnalyser:
    """
    An object to analyse the (LIBS) spectrum and get information.
    """
    __slots__ = (
                    'spec', 'comp', '_elem_idx', '_lines', '_has_reflines', 
                    '_has_matched', '_found_lines', 'Ne', 'Te', '_boltz', 
                )

    def __init__(self, spec: Spectrum, comp: list = []) -> None:
        self.spec      = spec

        self._elem_idx = {}     # mapping the element index to symbol
        self._lines    = None   # lines in this spectrum (and their data)
        self._boltz    = None   # computed boltzmann plane
        self.Ne        = None   # (estimated) value of electron temperature
        self.Te        = None   # (estimated) value of electron density

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
        Search for peaks in the spectrum and optionally, match with a set of referenece lines 
        given.
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

    def makeBoltzPlane(self, Te: float = ..., Ne: float = ..., fmt: str = 'boltz') -> None:
        """
        Prepare the Boltzmann plane for this spectrum.

        Parameters
        ----------
        Te: float, optional
            Temperature in eV (could be a guess value). Not needed if specifed in object.
        Ne: float, optional
            Electron density in :math:`{\rm cm}^{-3}`. Not needed if specifed in object.
        fmt: str, optional
            Boltzmann plane format. `boltz` for normal Boltzmann plane and `saha` for Saha-Boltzmann plane.

        """
        if fmt == 'saha':
            if Ne is ...:
                if  self.Ne is None:
                    raise AnalyserError("Ne should be given if not set")
                Ne = self.Ne
            if Te is ...:
                if self.Te is None:
                    raise AnalyserError("Te should be given if not set")
                self.Te = Te
        self._boltz = BoltzPlane(self.comp, self._lines, Te, Ne, fmt, 'lines')
        self._boltz._qa = self # set a reference to the analyser object
        return

    def getBoltzPlane(self, ) -> BoltzPlane:
        """
        Get the computed Boltzmann plane.
        """
        if self._boltz is None:
            raise AnalyserError("no boltzmann plane computed")
        return self._boltz

    def setNe(self, value: float, action: str = 'set', ref: float = ..., elem: str = ...) -> None:
        """
        Compute or set the value of electron density.

        Parameters
        ----------
        value: float
            If the action is `set`, it should be the value of :math:`N_e` to set, given in :math:`{\rm cm}^{-3}`. If the action is to `compute`, then it should be the value of stark width (FWHM) appropriate unit.
        action: str, optional
            What to do - `set` to set the given value or `compute` to estimate from the given information.
        ref: float, optional
            Reference wavelength, whose stark width is used to compute the value.
        elem: str, optional
            Element to which the reference wavelength correspond to. It should be given in a specific format, i.e., the element symbol and the species number seperated by `-` (eg., for copper-I, one can use `Cu-1`, provided there is an :class:`Element` object with symbol `Cu`). 

        """
        if action == 'set':
            if not isinstance(value, (float, int)):
                raise TypeError("value should be a number")
            self.Ne = value
            return
        elif action == 'compute':
            # compute Ne from a refernce line
            return NotImplemented
        raise ValueError(f"invalid action `{action}`")
