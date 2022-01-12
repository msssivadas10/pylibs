#!/usr/bin/python3
 
import numpy as np
from objects import loadElement, Plasma
from spectrum import Spectrum, QAnalyser

def _load_test_elements_cu_sn() -> tuple:
    """
    Create two test elements: copper and tin, with first two species. 
    """
    # create copper:
    copper = loadElement('data/cu.json')

    # create tin:
    tin = loadElement('data/sn.json')
    
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

def _test_cu_sn_spectrum(cu: float = 0.7, sn: float = 0.3, Te: float = 1., Ne: float = 1e+17, _from: float = 350., _to: float = 650., res: int = 500) -> Spectrum:
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

if __name__ == "__main__":
    ref = {
        510.554 : 'cu-1',
        515.324 : 'cu-1',
        521.820 : 'cu-1',
        380.1011: 'sn-1',
        452.4734: 'sn-1', 
        533.2339: 'sn-2', 
        556.1910: 'sn-2', 
        558.8815: 'sn-2', 
        579.8860: 'sn-2', 
        645.3542: 'sn-2', 
        684.4186: 'sn-2',
    }

    s = _test_cu_sn_spectrum()
    q = QAnalyser(s, list(s.attrs['plasma'].comp.keys()))
    q.setReferenceLines(ref)

    q.searchPeaks(match=1, dist_ub=1.)

    q.computeBoltzCoords('saha', 1., 1e+16)

    t = q.getBoltzCoords()

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    plt.figure()
    for elem in q._elem_idx.keys():
        i = np.where(t.c('elem') == elem)[0]
        plt.plot(t.c('x')[i], t.c('y')[i], 'o')
    plt.show()