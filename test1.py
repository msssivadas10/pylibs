import numpy as np
from pylibs.objects import linestable, levelstable, loadtxt, elementTree
from pylibs.plasma import plasma
from pylibs.analysis import boltzmann, optimizePlasma

def loadlines(file: str, select: list = None):
    accMap = {
                'AAA': 0.3, 'AA': 1.0, 'A+': 2.0, 'A': 3.0, 'B+': 7.0, 'B': 10.0,
                'C+': 18.0, 'C': 25.0, 'D+': 40.0, 'D': 50.0, 'E': 100.0,
             }

    elem, sp, x, Aki, acc, Ei, Ek, gi, gk = loadtxt( 
                                                        file, 
                                                        regex = r'\s*'.join([
                                                                                r'([A-Za-z]{2})', 
                                                                                r'(\d+)', 
                                                                                r'\"(\d*\.\d*)\"',
                                                                                r'\"(\d*\.\d*[eE]?[+-]?\d*)\"',
                                                                                r'([A-E]{1,3}[+]?)',
                                                                                r'\"(\d*\.\d*)\"',
                                                                                r'\"(\d*\.\d*)\"',
                                                                                r'(\d*)',
                                                                                r'(\d*)',         
                                                                            ]),
                                                        convert = 'sdffsffff' 
                                                    )

    acc  = np.array([ accMap[key] for key in acc ])
    sp   = sp - 1
    elem = np.array( list( map( str.lower, elem ) ) )

    if select is not None:
        i = np.zeros(elem.shape, dtype = 'bool')
        for __e in select:
            i = i | ( elem == __e.lower() )

        x, Aki, Ek, gk, elem, sp, acc = x[i], Aki[i], Ek[i], gk[i], elem[i], sp[i], acc[i]

    return linestable( np.stack([x, Aki, Ek, gk, acc], axis = -1), sp, elem )

def levels(x: str):
    """ Energy levels """
    x, s = x.split('-')
    return levelstable(
                        loadtxt( 
                                    f'../data/levels/{x}-levels-{s}.txt', 
                                    regex   = r'(\d+)\s*\"\s*[\[\(]*(\s*\d*\.\d*)[\?\]\)]?\"', 
                                    convert = True 
                               )
                      )

def main():
    lines = loadlines("percistent_lines.txt", ['cu', 'sn'] )

    t = {
            'cu': {
                    'm'      : 63.546,
                    'species': [
                                    {
                                        'Vs'    : 7.726380,
                                        'levels': levels('cu-1'),
                                        'lines' : lines,
                                    },
                                    {
                                        'Vs'    : 20.29239,
                                        'levels': levels('cu-2'),
                                        'lines' : lines,
                                    },
                            ]
                },
            'sn': {
                    'm'      : 118.710,
                    'species': [
                                    {
                                        'Vs'    : 7.343918,
                                        'levels': levels('sn-1'),
                                        'lines' : lines,
                                    },
                                    {
                                        'Vs'    : 14.63307,
                                        'levels': levels('sn-2'),
                                        'lines' : lines,
                                    }
                            ]
                },
        }

    t = elementTree(t)

    cusn = plasma('cusn', t)

    a = cusn(70.0)

    a.setup(1.0, 1.0E+17)

    s = a.getSpectrum(np.linspace(300.0, 600.0, 101), 500)

    a.lock()

    lnt = a.lines

    # boltzmann(lnt, 2, a)

    # print(lnt.boltzY)

    b = cusn(50.0)
    b.setup(0.5, 1.0E+17)
    b.lock()

    optimizePlasma( lnt, b,  )

    print( b.Te, b.composition, a.Te, a.composition )

    

if __name__ == '__main__':
    main()

# import pylibs.objects as o