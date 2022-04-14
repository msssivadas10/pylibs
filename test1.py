import numpy as np
from pylibs.objects import linestable, levelstable, loadtxt, elementTree
from pylibs.plasma import plasma
from pylibs.analysis import boltzmann, optimizePlasma

def loadlines(file: str, select: list = None):
    accMap = {
                'AAA': 0.3, 'AA': 1.0, 'A+': 2.0, 'A': 3.0, 'B+': 7.0, 'B': 10.0,
                'C+': 18.0, 'C': 25.0, 'D+': 40.0, 'D': 50.0, 'E': 100.0,
             }

    data = loadtxt( 
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

    elem, sp, x, Aki, acc = data[:5]
    Ek, gk                = data[6], data[8]

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

def lines(x: str):
    """ Lines """
    return loadlines(f"../data/lines/{x}-lines.txt")

def main():
    lnt = loadlines("percistent_lines.txt", ['cu', 'sn'] )

    t = {
            'cu': {
                    'm'      : 63.546,
                    'species': [
                                    {
                                        'Vs'    : 7.726380,
                                        'levels': levels('cu-1'),
                                        'lines' : lnt,
                                    },
                                    {
                                        'Vs'    : 20.29239,
                                        'levels': levels('cu-2'),
                                        'lines' : lnt,
                                    },
                            ]
                },
            'sn': {
                    'm'      : 118.710,
                    'species': [
                                    {
                                        'Vs'    : 7.343918,
                                        'levels': levels('sn-1'),
                                        'lines' : lnt,
                                    },
                                    {
                                        'Vs'    : 14.63307,
                                        'levels': levels('sn-2'),
                                        'lines' : lnt,
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

def main2():
    from pylibs.objects import element
    from pylibs.misc import createElementLibrary

    x = [
            {
                'key'    : 'cu',
                'm'      : 63.546,
                'species': [
                                {
                                    'Vs'    : 7.726380,
                                    'levels': levels('cu-1'),
                                    'lines' : lines('cu'),
                                },
                                {
                                    'Vs'    : 20.29239,
                                    'levels': levels('cu-2'),
                                    'lines' : lines('cu'),
                                },
                        ]
            },
            {
                'key'    : 'sn',
                'm'      : 118.710,
                'species': [
                                {
                                    'Vs'    : 7.343918,
                                    'levels': levels('sn-1'),
                                    'lines' : lines('sn'),
                                },
                                {
                                    'Vs'    : 14.63307,
                                    'levels': levels('sn-2'),
                                    'lines' : lines('sn'),
                                },
                        ]
            },
            {
                'key'    : 'zn',
                'm'      : 65.38,
                'species': [
                                {
                                    'Vs'    : 9.394197,
                                    'levels': levels('zn-1'),
                                    'lines' : lines('zn'),
                                },
                                {
                                    'Vs'    : 17.96439,
                                    'levels': levels('zn-2'),
                                    'lines' : lines('zn'),
                                },
                        ]
            },
            {
                'key'    : 'fe',
                'm'      : 55.845,
                'species': [
                                {
                                    'Vs'    : 7.9024681,
                                    'levels': levels('fe-1'),
                                    'lines' : lines('fe'),
                                },
                                {
                                    'Vs'    : 16.19921,
                                    'levels': levels('fe-2'),
                                    'lines' : lines('fe'),
                                },
                        ]
            },
            {
                'key'    : 'al',
                'm'      : 26.981,
                'species': [
                                {
                                    'Vs'    : 5.985769,
                                    'levels': levels('al-1'),
                                    'lines' : lines('al'),
                                },
                                {
                                    'Vs'    : 18.82855,
                                    'levels': levels('al-2'),
                                    'lines' : lines('al'),
                                },
                        ]
            },
        ]

    createElementLibrary( 'stock_elem.dat', [ element(xi) for xi in x ] ) 

    return   

def main3():
    from pylibs.misc import stockElements

    print( stockElements[ 'cu' ] )


if __name__ == '__main__':
    main3()
