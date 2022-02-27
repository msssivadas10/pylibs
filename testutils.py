#!/usr/bin/python3

# code for testings
from pylibs.spectrum import Spectrum, QAnalyser
from pylibs.objects import Species, Element, Plasma, loadElement
from pylibs.utils import Table 


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
    from pylibs.utils import Parser, readtxt

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

def loadTestElements_CuSn() -> tuple:
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

def testPlasma_CuSn(cu: float = 0.7, sn: float = 0.3, Te: float = 1., Ne: float = 1.e+17) -> Plasma:
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
    copper, tin = loadTestElements_CuSn()
    return Plasma({copper: cu, tin: sn}, Te, Ne)

def testSpectrum_CuSn(cu: float = 0.7, sn: float = 0.3, Te: float = 1., Ne: float = 1e+17, _from: float = 350., _to: float = 650., res: int = 500):
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
    p = testPlasma_CuSn(cu, sn, Te, Ne) # test plasma
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