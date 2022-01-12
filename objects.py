#!/usr/bin/python3
from typing import Any
from utils import Table, shape_
import numpy as np

class Species:
    """
    An object to represent a species/ion of an element. This stores the spectroscopic data for that species.

    Parameters
    ----------
    _id: int
        Species id. This represent the charge of the species.
    eion: float
        Ionization energy of the species in eV.
    linetab: :class:`Table`, optional
        Spectroscopic data for lines of this species. This should be a :class:`Table` object with columns `lambda` (wavelength in nm), `Aki` (transition rate), `Ei` and `Ek` (levels), `gi` and `gk` (multiplicity).
    leveltab: :class:`Table`, optional
        Energy levels of this species. This should be a :class:`Table` object with columns `E` (level energy) and `g` (multiplicity).
    
    """
    __slots__ = '_id', 'eion', 'linetab', 'leveltab', '_last_z', "_spec", '_reflines'

    def __init__(self, _id: int, eion: float, linetab: Table = ..., leveltab: Table = ...) -> None:
        self._id  = _id  # this indicate the charge of the species
        self.eion = eion # ionization energy

        self.linetab  = None
        if linetab is not Ellipsis:
            self.set_linesTable(linetab)

        self.leveltab = None
        if leveltab is not Ellipsis:
            self.set_levelsTable(leveltab)

        self._last_z   = None
        self._spec     = None
        self._reflines = []

    def __repr__(self) -> str:
        if self.linetab is None:
            ln_info = 'lines: not set'
        else:
            ln_info = f"lines: {self.linetab.shape[0]}"
        
        if self.leveltab is None:
            lv_info = 'levels: not set'
        else:
            lv_info = f"levels: {self.leveltab.shape[0]}"

        return f"<Species id: {self._id}, E_ion: {self.eion}, {ln_info}, {lv_info}>"

    def set_linesTable(self, table: Table) -> None:
        """
        Spectroscopic data for the lines of this species.

        Parameters
        ----------
        table: :class:`Table`
            Data as a table. This should be a :class:`Table` object with columns `lambda` (wavelength in nm), `Aki` (transition rate), `Ei` and `Ek` (levels), `gi` and `gk` (multiplicity). If any of these columns is absent, raise :class:`ValueError`.

        """
        if not isinstance(table, Table):
            raise TypeError("`linetab` should be a 'Table'")
        if not table.has_columns(['lambda', 'Aki', 'Ek', 'gk']):
            print(table.colnames)
            raise ValueError("table has not all the required data")
        self.linetab  = table[['lambda', 'Aki', 'Ek', 'gk']]
        return 

    def set_levelsTable(self, table: Table) -> None:
        """
        Data for the energy levels of this species.

        Parameters
        ----------
        table: :class:`Table`
            Data as a table. This should be a :class:`Table` object with columns `E` (level energy) and `g` (multiplicity). If any of these columns is absent, raise :class:`ValueError`.

        """
        if not isinstance(table, Table):
            raise TypeError("`leveltab` should be a 'Table'")
        if not table.has_columns(['E', 'g']):
            raise ValueError("table has not all the required data")
        self.leveltab = table
        return

    def z(self, T: float) -> float:
        r"""
        Partition function for this species at the given temperature :math:`T`, given in eV. Partition function is given by the sum

        .. math::
            Z(T) = \sum_j g_j \exp \left( -\frac{E_j}{T} \right)

        Parameters
        ----------
        T: float
            Temperature to evaluate the partition function.

        Returns
        -------
        z: float
            Value of partition function.

        Examples
        --------
        TODO

        """
        if not isinstance(T, float):
            raise ValueError("`T` should be a 'float'")
        if self.leveltab is None:
            raise ValueError("no energy level data available for this species")

        # use previous computed value if last input repeated
        if self._last_z is not None:
            if self._last_z[0] == T:
                return self._last_z[1]
        z = np.sum(self.leveltab.c('g') * np.exp(-self.leveltab.c('E') / T))
        self._last_z = (T, z)
        return z

    def _compute_lte_intensity(self, n: float, T: float) -> None:
        """ Internal function to compute and save LTE line intensities """
        lines  = self.linetab 
        intens = lines.c('Aki') * 1239.84198 / lines.c('lambda') / 4. / np.pi
        intens = (n * lines.c('gk') * np.exp(-lines.c('Ek') / T) / self.z(T)) * intens
        self._spec = intens
        return

    def _lte_spectrum(self, x: Any, res: int) -> Any:
        """ Compute the spectrum """
        if self._spec is None:
            raise ValueError("no saved intensity table")

        intens = 0.
        for x0, y0 in zip(self.linetab.c('lambda'), self._spec):
            intens += shape_["gaussian"](x, x0, y0, x0 / res)
        return intens   

    def _all_lines(self, ) -> Table:
        """ Get all the lines, with intensity """
        if self._spec is None:
            raise ValueError("no saved intensity table")
        out = self.linetab.copy()
        out.add_column(self._spec, 'intens', float)
        return out

    def _add_refline(self, _lambda: float, tol: float = 0.001) -> bool:
        """ Add a reference line """
        dists  = np.abs(self.linetab.c('lambda') - _lambda)
        matchs = np.where(dists < tol)[0]

        nmatchs = len(matchs)
        if nmatchs:
            if nmatchs == 1:
                self._reflines.append(matchs[0])
            else:
                self._reflines.append(matchs[np.argmin(dists[matchs])])
            return True
        return False

    def _get_reflines(self, ) -> Table:
        """ Get the reference lines as a table """
        if not len(self._reflines):
            return Table()
        return self.linetab.subtable(self._reflines)


class Element:
    """
    An object to represent an element. This stores data such as its symbol, atomic number and mass, as well as the spectroscopic data for different ionic species of that element.

    Parameters
    ----------
    z: int
        Atomic number of the element.
    mass: float
        Atomic mass in atomin mass unit (u).
    sym: str, optional
        Symbol for the element, such as H for hydrogen and He for helium.
    species: list, optional
        Species data for this element. Each entry in the list should be a :class:`Species` object.

    Examples
    --------
    To create a object representing helium (symbol He, Z = 2 and mass ~ 4 u),

    >>> He = Element(z = 2, mass = 4., sym = "He")
    >>> He
    <Element 'He', Z = 2, mass = 4.000 u>

    """
    __slots__ = "z", "mass", "sym", "nspecies", "species"

    def __init__(self, z: int, mass: float, sym: str = ..., species: list = []) -> None:
        self.z    = z
        self.mass = mass
        self.sym  = sym.capitalize() if sym is not Ellipsis else f"Element{z}"
        
        self.nspecies = len(species)
        self.species = []
        if self.nspecies:
            if not min(map(lambda __x: isinstance(__x, Species), species)):
                raise ValueError("all entries in the `specdata` list should be a 'Species' object")
            self.species = species

    def __repr__(self) -> str:
        return f"<Element '{self.sym}', Z = {self.z}, mass = {self.mass:.3f} u>"
       
    def _lte_fraction(self, Te: float, Ne: float, ntot: float = 1.) -> float:
        r"""
        Find the fractional population of each species with respect to the total population, at a specific temperature and electron density, assuming local thermal equilibrium (LTE). This is computed using the Saha LTE equation,

        .. math::
            f_i := \frac{n_{i+1}}{n_i} = 6.009 \times 10^{21} \left[ \frac{T_e^{3/2}}{N_e} \cdot \frac{Z_{i+1}(T_e)}{Z_i(T_e)} \right] e^{-V_i / T_e}

        where, :math:`Z` is the partition function and :math:`V_i` is the ionization energy of the :math:`i`-th species. Population of the neutral species is computed by

        .. math::
            n_0 = \left[ 1 + f_0 (1 + f_1 (\cdots)) \right]^{-1}

        Parameters
        ----------
        Te: float
            Electron temperature in units of kelvin.
        Ne: float
            Electron number density in units of :math:`{\rm cm}^{-3}`.
        ntot: float, optional
            Population of the element (default is 1).

        Returns
        -------
        frac: array_like
            Fractional population. This has the same size as the species list.

        Examples
        --------
        TODO

        """
        if not isinstance(Te, float):
            raise TypeError("`Te` should be 'float'")
        if not isinstance(Ne, float):
            raise TypeError("`Ne` should be 'float'")

        const = 6.009e+21 * Te**1.5 / Ne

        def __lte_frac(i):
            return const * self.species[i].z(Te) / self.species[i-1].z(Te) * np.exp(-self.species[i-1].eion / Te)

        frac  = list(map(__lte_frac, range(1, self.nspecies)))
        frac0 = 1.
        for _frac in reversed(frac):
            frac0 = frac0 * _frac + 1
        frac0 = ntot / frac0

        frac = [frac0, ] + frac
        for i in range(1, self.nspecies):
            frac[i] *= frac[i-1]
        return np.asarray(frac)

    def _compute_lte_intensity(self, Te: float, Ne: float, ntot: float = 1.) -> None:
        r"""
        Compute the intensity of spectral lines of this element at LTE.

        Parameters
        ----------
        Te: float
            Electron temperature in units of kelvin.
        Ne: float
            Electron number density in units of :math:`{\rm cm}^{-3}`.
        n: float, optional
            Population of the element (default is 1).

        Returns
        -------
        ints: tuple
            Intensity of each line. This will be a tuple of :class:`Table` objects with two columns: `lambda` and `int`.

        """
        num = self._lte_fraction(Te, Ne, ntot)
        for sp, n in zip(self.species, num):
            sp._compute_lte_intensity(n, Te)
        return

    def _lte_spectrum(self, x: Any, res: int) -> Any:
        """ 
        Compute the spectrum at LTE.

        Parameters
        ----------
        x: array_like
            Wavelength values to compute intensity.
        res: int
            Resolution parameter to find width.

        Returns
        -------
        spec: :class:`Table`
            Computed spectrum of different species.
        
        """
        return Table(
                        {
                            f"{self.sym}-{s._id}": s._lte_spectrum(x, res) for s in self.species
                        }
                    )

    def _all_lines(self, ) -> Table:
        """ Spectroscopic data of all lines """
        out = Table()
        for sp in self.species:
            t = sp._all_lines()
            if not t.shape[1]:
                continue
            t.insert_column(np.repeat(sp._id, t.shape[0]), 0, 'sp', int)
            out = out + t
        return out

    def add_refline(self, _id: int, _lambda: float, tol: float = 0.001) -> bool:
        """
        Add a reference line. A reference line is used later for analysis of a plasma spectrum.

        Parameters
        ----------
        _id: int
            Id of the species to which the line belongs.
        _lambda: float
            Wavelength of the line.  
        tol: float, optional
            Tolerance for wavelength matching (default is 0.001).

        Returns
        -------
        res: bool
            True if successfully added reference line, else false.

        """
        return self.species[_id-1]._add_refline(_lambda, tol)

    def get_reflines(self, ) -> Table:
        """ Get the reference lines as a table """
        out = Table()
        for sp in self.species:
            t = sp._get_reflines()
            if not t.shape[1]:
                continue
            t.insert_column(np.repeat(sp._id, t.shape[0]), 0, 'sp', int)
            out = out + t
        return out

def loadElement(fname: str) -> Element:
    """ 
    Create an element by loading data from a JSON file. 

    Parameters
    ----------
    fname: str
        Filename (or path) for the data file.

    Returns
    -------
    elem: :class:`Element`
        Element object.

    Examples
    --------
    TODO

    """
    import json
    
    def _table_from_dict(_dict: dict):

        def _type(key: str):
            _map = {'float': float, 'int': int, 'str': str}
            return _map[key]

        return Table(_dict['data'], _dict['cols'], list(map(_type, _dict['types'])))

    def _create_species_from_dict(_dict: dict):
        s = Species(_dict['id'], _dict['Eion'], )
        s.set_linesTable(_table_from_dict(_dict['lines']))
        s.set_levelsTable(_table_from_dict(_dict['levels']))
        return s

    data = json.load(open(fname, 'r'))
    
    # create element object
    elem = Element(
                    z       = data['z'], 
                    mass    = data['mass'], 
                    sym     = data['symbol'],
                    species = list(map(_create_species_from_dict, data['species'])),
                )
    return elem

def saveElement(elem: Element, path: str = '.') -> None:
    """
    Save the data of the given element in JSON format. 
    
    Parameters
    ----------
    elem: :class:`Element`:
        Element whose data is to save.
    path: str, optional
        Path to an existing directory, to where the file saved. If it not exist, raise error. 

    """
    import json, os, re

    def _table_to_dict(tab: Table):

        def _typename(_t: type):
            return re.search(r"(?<=\')\w+", repr(_t)).group(0)

        def _build_value(val: Any, _t: type):
            return _t(val)

        def _build_row(row: tuple):
            return list(map(_build_value, row, tab.coltypes))

        return {
                    'cols' : tab.colnames,
                    'types': list(map(_typename, tab.coltypes)),
                    'data' : list(map(_build_row, list(zip(*tab.data))))
               }

    def _species_to_dict(sp: Species):
        return {
                    'id'    : sp._id,
                    'Eion'  : sp.eion,
                    'lines' : _table_to_dict(sp.linetab),
                    'levels': _table_to_dict(sp.leveltab),
               }

    def _element_to_dict(elem: Element):
        return {
                    'symbol'  : elem.sym,
                    'z'       : elem.z,
                    'mass'    : elem.mass,
                    'nspecies': elem.nspecies,
                    'species' : list(map(lambda _s: _s._to_dict(), elem.species)),
               }
    
    file = os.path.abspath(path)
    if not os.path.isdir(file):
        raise NotADirectoryError("the given path is not a directory or does not exist.")
    file = os.path.join(file, f'{elem.sym.lower()}.json')

    json.dump(_element_to_dict(elem), open(file, 'w'), indent = 4)
    return


class Plasma:
    r"""
    Object representing a plasma in local thermal equilibrium. The plasma contains one or more elements as its components.

    Parameters
    ----------
    components: dict
        A python dict mapping the component elements to their population (specified in terms of weight fraction or numbe fraction). 
    Te: float
        Plasma temperature in eV.
    Ne: float
        Electron number density in the plasma, in :math:`{\rm cm}^{-3}`.
    fmt: str, optional
        Format for specifying the population of an element. Its allowed values are `wt` (default) for weight fraction and `num` for number fraction. In the first case, composition will be concverted to a number fraction.

    Examples
    --------
    TODO

    """
    __slots__ = "comp", "Te", "Ne", "_updated"

    def __init__(self, components: dict, Te: float, Ne: float, fmt: str = 'wt') -> None:
        if not len(components):
            raise ValueError("no components specified")

        _comp = list(components.keys())
        if not min(map(lambda _x: isinstance(_x, Element), _comp)):
            raise TypeError("components must be 'Element' objects")

        _conc = list(components.values())
        if sum(_conc) != 1.:
            _conc[-1] = 1. - sum(_conc[:-1])
            if _conc[-1] < 0.:
                raise ValueError("got negative concentration")
        if fmt == 'wt':
            _conc = list(map(lambda _w, _e: _w / _e.mass, _conc, _comp))
            _sum  = sum(_conc) 
            _conc = list(map(lambda _w: _w / _sum, _conc))
        elif fmt != 'num':
            raise ValueError(f"unknown value for fmt: '{fmt}'")

        self.comp = dict(zip(_comp, _conc))
        self.Te   = Te
        self.Ne   = Ne

        self._updated = True

        self._compute_lte_intensity()

    def __repr__(self) -> str:
        comp = ", ".join(map(lambda _x, _y: f"{_x.sym}: {_y}", self.comp.keys(), self.comp.values()))
        return f"<Plasma composition = ({comp}), Te = {self.Te} eV, Ne = {self.Ne} cm^-3>"

    def setTemperature(self, value: float) -> None:
        """
        Set the plasma temperature. After setting the values, :meth:`update` should be called to make the changes in effect. 

        Parameters
        ----------
        value: float
            Temperature in eV.
        """
        self.Te = value
        self._updated = False
        return

    def setElectronDensity(self, value: float) -> None:
        """
        Set the electron number density. After setting the values, :meth:`update` should be called to make the changes in effect. 

        Parameters
        ----------
        value: float
            electron density in :math:`{\\rm cm}^{-3}`.
        """
        self.Ne = value
        self._updated = False
        return

    def composition(self, ) -> dict:
        """
        Plasma composition in terms of number fraction. Return value is a :class:`dict` with element symbols as keys and composition vector as values.
        """
        return {
                    elem.sym: elem._lte_fraction(self.Te, self.Ne, ntot) for elem, ntot in self.comp.items()
                }

    def update(self, ) -> None:
        """
        Update plasma conditions. This should be called after setting the plasma conditions, in order to make the changes effective. 
        """
        if self._updated:
            return
        self._compute_lte_intensity()
        self._updated = True
        return

    def _compute_lte_intensity(self, ) -> None:
        """
        Internal function to compute the line intensities at LTE.
        """
        for elem, conc in self.comp.items():
            elem._compute_lte_intensity(self.Te, self.Ne, conc)
        return
        
    def computeSpectrum(self, _from: float, _to: float, res: int = 500, ) -> Table:
        r"""
        Compute the plasma spectrum.

        Parameters
        ----------
        _from: float
            Minimum wavelength value. Must be greater than 100.
        _to: float
            Maximum wavelength value. Must be less than 1000.
        res: int, optional
            Resolution (number of samples will be 20 times this, qpprox.). Default is 200.

        Returns
        -------
        spectrum: :class:`Table`
            Plasma spectrum and its components as a table.

        """
        if _from < 100.:
            raise ValueError("wavelengths must be greater than 100")
        elif _to > 1000.:
            raise ValueError("wavelengths must be less than 1000")
        if res < 0:
            res = 200
        
        x   = np.linspace(_from, _to, int(20 * res))
        out = Table().join([elem._lte_spectrum(x, res) for elem in self.comp.keys()])
        out.add_column(sum(out.data), "sum", float)
        out.insert_column(x, 0, 'lambda', float)
        return out

    def allLines(self, _from: float = ..., _to: float = ...) -> Table:
        """
        Return all the lines of the plasma components, with their intensity as a table. 
        """
        out = Table()
        for elem in self.comp.keys():
            t = elem._all_lines()
            if not t.shape[1]:
                continue
            t.insert_column(np.repeat(elem.sym, t.shape[0]), 0, 'elem', str)
            out = out + t
        if _from is not Ellipsis:
            if _to is Ellipsis:
                _to = 1000.
            rows = np.where((_from <= out.c('lambda')) & (out.c('lambda') <= _to))
            return out.subtable(rows, ...)
        return out


class ElementSpecifier:
    """
    A value used to specify a species of an element
    """
    __slots__ = 'val', 'sep', '_sym', '_id'

    def __init__(self, val: Any, sep: str = '-') -> None:
        self.val = val
        self.sep = sep

        # parse the value:
        self._sym, self._id = self._parse_elemspec(val)
    
    def _parse_elemspec(self, es: Any) -> Any:
        """ Parse the specifier value """
        if isinstance(es, str):
            x = es.split(self.sep)
            if len(x) == 1:
                return x[0].capitalize(), None
            if len(x) == 2:
                return x[0].capitalize(), int(x[1])
        elif isinstance(es, tuple):
            if len(es) == 2:
                return str(es[0]).capitalize(), int(es[1])
        raise ValueError("invalid element specifier")

    def __repr__(self) -> str:
        _id = f'-{self._id}' if self._id is not None else ''
        return f"<ElementSpecifier '{self._sym}{_id}'>"

    def value(self, ) -> Any:
        """ 
        Parsed value as a (symbol, species id) pair. For un-specified species, only symbol is returned.
        """
        if self._id is None:
            return self._sym
        return self._sym, self._id

    def symbol(self, ) -> str:
        """ Get the element symbol. """
        return self._sym

    def id(self, ) -> Any:
        """ Get the species id integer, if exist, else None. """
        return self._id

    def is_element(self, ) -> bool:
        """ Return true if an element, else (if species) false """
        return (self._id is None)


# =================================================================================
# Test functions
# =================================================================================

def _load_species(lines_file: str, levels_file: list, eion: list) -> list:
    """
    Load species data from files. To use this function, the data files should be of a specific format. Otherwise, the result may be unwanted.

    Parameters
    ----------
    lines_file: str
        Path to the file containing the spectral line data.
    levels_file: list of str
        List of files containg the energy level data of each species. Files should be sorted in terms of the species charge. i.e., neutral species file should come the first, then single ionized etc. 
    eion: list of float
        List of ionisation energies (sorted in a similar way as energy level data files).

    Returns
    -------
    specs: list 
        List of species objects.

    Notes
    -----
    1. **lines file format**: The lines file assume columns as in the query output of the NIST database in CSV format. i.e., the first 9 columns should be element, species number, wavelength, Aki, accuracy, Ei, Ek, gi and gk. Of this, element and accuracy are dropped.
    2. **levels file format**: There should be two columns - g and E. If a column has no g value (specify an ionized level), that row is dropped. 
    3. Units of energy should be same in all files (eV is the preffered unit), for all elements.

    """
    from utils import Parser, readtxt

    nspec = len(levels_file)
    if len(eion) != nspec:
        raise ValueError("not enough data")
    levels = list(range(1, nspec+1))

    specs = [Species(_id = levels[i], eion = eion[i]) for i in range(nspec)]

    # load levels data
    for i, file in  enumerate(levels_file):
        x = readtxt(
                file, 
                transpose = 1, 
                parser = Parser(ignore_cols = [2, ], ignore_conds = {0: lambda _x: _x == ''})
            )
        specs[i].set_levelsTable(Table(tuple(x), ['g', 'E'], ))

    y = readtxt(
                lines_file, 
                transpose = 1, 
                parser = Parser(ignore_cols = [0, 4, ], max_cols = 9)
            )
    y = Table(tuple(y), ['sp', 'lambda', 'Aki', 'Ei', 'Ek', 'gi', 'gk'], )
    for i, lns in enumerate(y.split('sp', levels)):
        specs[i].set_linesTable(lns)

    return specs

def _load_test_elements_cu_sn() -> tuple:
    """
    Create two test elements: copper and tin, with first two species.
    """
    # create copper:
    try:
        copper = loadElement('data/cu.json')
    except FileNotFoundError:
        copper = Element(
                            z       = 29, 
                            mass    = 63.546, 
                            sym     = "Cu", 
                            species = _load_species(
                                            'data/cu_lines.csv', 
                                            ['data/cu-i_level.csv', 'data/cu-ii_level.csv'], 
                                            [7.726380,20.29239]
                                        )
                        )
        copper.save("./data")

    # create tin:
    try:
        tin = loadElement('data/sn.json')
    except FileNotFoundError:
        tin     = Element(
                            z       = 50, 
                            mass    = 118.71, 
                            sym     = "Sn", 
                            species = _load_species(
                                            'data/sn_lines.csv', 
                                            ['data/sn-i_level.csv', 'data/sn-ii_level.csv'], 
                                            [7.343918,14.63307]
                                        )
                        )
        tin.save("./data")
    
    return copper, tin

def _test_cu_sn_plasma(cu: float = 0.7, sn: float = 0.3, Te: float = 1., Ne: float = 1.e+17) -> Plasma:
    """
    Create a test plasma of a copper-tin compound.

    Parameters
    ----------
    cu: float
        Copper abundance (default is 0.7).
    sn: float
        Tin abundance (default is 0.3).
    Te: float
        Temperature in eV (default is 1).
    Ne: float
        Electron number density in :math:`cm^{-3}` (default is 1e+17).

    Returns
    -------
    p: Plasma
        Copper-tin plasma object.

    """
    copper, tin = _load_test_elements_cu_sn()
    return Plasma({copper: cu, tin: sn}, Te, Ne)

if __name__ == "__main__":
    p = _test_cu_sn_plasma()

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    plt.figure()
    y = p.computeSpectrum(350., 650., 1000)
    plt.plot(y.c('lambda'), y.c('sum'))
    plt.show()