from pylibs_expt._tables import LinesTable, LevelsTable
from pylibs_expt.
import re
import numpy as np

def createLinesTable(file: str, select: list = None) -> LinesTable:
    def loadLines(file: str) -> list:
        """ load spectral lines from a file.  """
        ptn  = r'\s*'.join([
                                r'(?P<elem>[A-Za-z]{2})', 
                                r'(?P<sp>\d+)', 
                                r'\"(?P<x>\d*\.\d*)\"',
                                r'\"(?P<aki>\d*\.\d*[eE]?[+-]?\d*)\"',
                                r'(?P<acc>[A-E]{1,3}[+]?)',
                                r'\"(?P<ei>\d*\.\d*)\"',
                                r'\"(?P<ek>\d*\.\d*)\"',
                                r'(?P<gi>\d*)',
                                r'(?P<gk>\d*)',         
                            ])
        lines = []
        with open(file, 'r') as f:
            for __line in f.read().splitlines():
                m = re.search(ptn, __line)
                if m:
                    lines.append( list( m.groups() ) )
        return lines

    accMap = {
                'AAA': 0.3, 'AA': 1.0, 'A+': 2.0, 'A': 3.0, 'B+': 7.0, 'B': 10.0,
                'C+': 18.0, 'C': 25.0, 'D+': 40.0, 'D': 50.0, 'E': 100.0,
             }

    elem, sp, x, Aki, acc, Ei, Ek, gi, gk = np.array(loadLines(file)).T

    acc = np.array([accMap[key] for key in acc])

    if select is not None:
        elem = np.array( list( map( str.lower, elem ) ) )

        i = np.zeros(elem.shape, dtype = 'bool')
        for __e in select:
            i = i | ( elem == __e.lower() )

        return LinesTable(x[i], Aki[i], Ek[i], gk[i], elem[i], sp[i], acc[i])

    return LinesTable(x, Aki, Ek, gk, elem, sp, acc)

def createLevelsTable(file: str, elem: str = ..., sp: int = ...) -> LevelsTable:
    def loadLevels(file: str) -> list:
        """ load energy levels from a file """
        ptn = r'(\d+)\s*\"\s*[\[\(]*(\s*\d*\.\d*)[\?\]\)]?\"'

        levels = []
        with open(file, 'r') as f:
            for __line in f.read().splitlines():
                m = re.search(ptn, __line)
                if not m:
                    continue
                levels.append(m.groups())
        return np.array(levels, 'float').T

    g, E = loadLevels(file)
    return LevelsTable(g, E)


lines = createLinesTable("percistent_lines.txt", ['Cu', 'Sn'])

lvl_cu1 = createLevelsTable(f'../data/levels/cu-levels-1.txt')
lvl_cu2 = createLevelsTable(f'../data/levels/cu-levels-2.txt')
lvl_sn1 = createLevelsTable(f'../data/levels/sn-levels-1.txt')
lvl_sn2 = createLevelsTable(f'../data/levels/sn-levels-2.txt')
