from testing import loadlines
from pylibs.plasma import plasma
from pylibs.analysis import boltzmann, optimizePlasma
from pylibs.misc import stockElement
import numpy as np

def main():
    lnt = loadlines("percistent_lines.txt" )

    t = [ stockElement[ 'cu' ].copy(), stockElement[ 'sn' ].copy() ]

    for ti in t:
        ti.setLines( lnt )

    cusn = plasma('cusn', t)

    a = cusn(70.0)

    a.setup(1.0, 1.0E+17)

    # s = a.getSpectrum(np.linspace(300.0, 600.0, 101), 500)

    a.lock()

    lnt = a.lines

    # boltzmann(lnt, 2, a)

    b = cusn(50.0)
    b.setup(0.5, 1.0E+17)
    b.lock()

    optimizePlasma( lnt, b,  )

    print( b.Te, b.composition, a.Te, a.composition )



if __name__ == '__main__':
    main()
