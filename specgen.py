#!/usr/bin/python3

import numpy as np
import pandas as pd
import json
import os.path as path
from typing import Any


roman = {
            1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 
            6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X'
        } # first 10 roman numerals

class Profiles:

    def gaussian(x: Any, x0: float, w: float):

        t = (x - x0) / w
        return np.exp(-4*np.log(2.0)*t*t)

    def lorentzian(x: Any, x0: float, w: float):
        
        t = (x - x0) / (0.5*w)
        return 1. / (1. + t*t)

    def voigt(x: Any, x0: float, w: float, frac: float = 0.5):
        
        # pseudo-voigt
        return frac * Profiles.gaussian(x, x0, w) + (1-frac) * Profiles.lorentzian(x, x0, w)

class DataNotAvailableException(Exception):
    ...

class SpectroscopicDatabase:

    __slots__ = 'info', 'lines', 'levels', 

    def __init__(self, lines: pd.DataFrame, levels: pd.DataFrame, info: dict) -> None:

        # ==================================================
        # checking data availablity
        # ==================================================

        if not isinstance(lines, pd.DataFrame):
            raise TypeError("lines must be a 'pandas.DataFrame' object")
        if lines.shape[0] == 0:
            raise DataNotAvailableException("lines table cannot be empty")
        # check for important columns 
        for col in ['Element', 'SpecNum', 'Wavelen', 'Aki', 'Ei', 'Ek', 'gi', 'gk']:
            if col not in lines.columns:
                raise DataNotAvailableException(f"'{col}' data is not available in lines table")


        if not isinstance(levels, pd.DataFrame):
            raise TypeError("levels must be a 'pandas.DataFrame' object")
        if levels.shape[0] == 0:
            raise DataNotAvailableException("levels table cannot be empty")
        # check for important columns 
        for col in ['Element', 'SpecNum', 'g', 'Level']:
            if col not in levels.columns:
                raise DataNotAvailableException(f"'{col}' data is not available in levels table")

        # check info data format
        if not isinstance(info, dict):
            raise TypeError("info must be a 'dict'")
        
        for key, value in info.items():
            
            # check if lines and levels for this element is available
            if key not in lines['Element'].values:
                raise DataNotAvailableException(f"lines for '{key}' is not available")
            if key not in levels['Element'].values:
                raise DataNotAvailableException(f"levels for '{key}' is not available")

            # check values dict
            if not isinstance(value, dict):
                raise TypeError("Values in 'info' dict must be a 'dict'")
            for name in ["Z", "Mass", "IonizationEnergy"]:
                if name not in value.keys():
                    raise DataNotAvailableException(f"'{name}' is not available for '{key}'")

        self.lines  = lines
        self.levels = levels
        self.info   = info
  
    @classmethod
    def loadfrom(cls, lines_file: str, levels_file: str, info_file: str, lines_args: dict = {}, levels_args: dict = {}):

        lines  = pd.read_csv(lines_file, **lines_args)
        levels = pd.read_csv(levels_file, **levels_args)
        info   = json.load( open(info_file, 'r') )
        return cls(lines, levels, info)


_specDB = SpectroscopicDatabase.loadfrom(
            lines_file  = path.join(path.split(__file__)[0], 'data', 'lines.csv'),
            levels_file = path.join(path.split(__file__)[0], 'data', 'levels.csv'),
            info_file   = path.join(path.split(__file__)[0], 'data', 'info.json'),
            lines_args  = dict(
                                names=['Element', 'SpecNum','Wavelen' ,'Aki', 'Acc', 'Ei', 'Ek', 'gi', 'gk', 'Type'],
                                delimiter=',', comment='#', skipinitialspace=True, na_values=' '
                            ),
            levels_args = dict(
                                names=['Element', 'SpecNum', 'g', 'Level', 'Unceratinity'],
                                delimiter=',', comment='#', skipinitialspace=True, na_values=' '
                            )
        )

class SpectrumGenerator:

    __slots__ = 'Te', 'Ne', 'elementData', 'specData', 'lines', 

    def __init__(self, wt: dict, Te: float, Ne: float, db: SpectroscopicDatabase = _specDB) -> None:

        # check if composition and data availability        
        total = 0.0
        for key, value in wt.items():
            
            if key not in db.info.keys():
                raise KeyError(f'No data available for element `{key}`')

            if value < 0:
                raise ValueError('Weight of an element cannnot be negative')            
            total += value
        
        if abs(total - 1) > 1e-08:
            raise ValueError('Total weight must be 1')

        # set electron temperature and density
        if Te <= 0.:
            raise ValueError('Temperature `Te` cannot be zero or negative')
        self.Te = Te

        if Ne <= 0.:
            raise ValueError('Electron density `Ne` cannot be zero or negative')
        self.Ne = Ne

        # ====================================================
        # component elements data
        # ====================================================
        elementData = {
                            'Element': [], 'AtomicNum': [], 'Mass': [], 'Weight': []
                      }
        for key, value in wt.items():
            elementData['Element'].append( key.capitalize() ) # symbol
            elementData['AtomicNum'].append( db.info[key]['Z'] ) # atomic number
            elementData['Mass'].append( db.info[key]['Mass'] ) # atomic mass in u
            elementData['Weight'].append( value ) # weight fraction
        
        elementData = pd.DataFrame(elementData).set_index('Element')
        elementData.index.names = [None]

        # number fractions
        elementData['Number']  = elementData['Weight'] / elementData['Mass']
        elementData['Number'] /= elementData['Number'].sum()

        self.elementData = elementData

        # ======================================================
        # component species data
        # ======================================================
        specData = {
                        'SpecID': [], 'Element': [], 'SpecNum': [], 
                        'IonizationEnergy': [], 'PartitionFunction': [],
                   }
        
        for key in self.elementData.index:
            
            for i in range(1, 3):
                specID = '-'.join( [key, roman[i]] )

                specData['SpecID'].append(specID)
                specData['Element'].append(key)
                specData['SpecNum'].append(i)
                specData['IonizationEnergy'].append( db.info[key]['IonizationEnergy'][i-1] ) # ionization energy

                # calculate partition function
                lvdata = db.levels.query(  
                                            f"Element=='{key}' and SpecNum=={i}" 
                                        )[['g', 'Level']].dropna() # levels for this species
                pf     = np.sum( lvdata['g'] * np.exp(-lvdata['Level']/Te) )
                specData['PartitionFunction'].append(pf)
        
        specData = pd.DataFrame(specData).set_index('SpecID')
        specData.index.names = [None]

        # calculate fraction of each species
        const = 6.009e+21 * Te**1.5 / Ne

        pf = specData['PartitionFunction'].to_numpy()
        Ei = specData['IonizationEnergy'].to_numpy()
        f  = const * pf[1::2] / pf[::2] * np.exp(-Ei[::2]/Te)
        n0 = self.elementData['Number'].to_numpy() / (1. + f)
        
        specData['Number'] = np.stack([n0, f*n0]).T.flatten()

        self.specData = specData

        # ====================================================
        # lines for this combinations of elements
        # ====================================================
        self.lines = db.lines.query("Element in @self.elementData.index")
        self.lines = self.lines.reset_index().drop(['Type', 'index'], axis=1)
        self.lines['Intensity'] = np.nan

        self._calculateIntensities()

    def _calculateIntensities(self):

        j = self.lines['Element'] + '-' + self.lines['SpecNum'].map(roman) # row index

        # calculate upper level population
        nk = ( self.specData['Number'].loc[j] / self.specData['PartitionFunction'].loc[j] ).to_numpy() * self.lines['gk'] * np.exp(-self.lines['Ek']/self.Te)

        # calculate intensity
        self.lines['Intensity'] = nk * self.lines['Aki'] * (self.lines['Ek']-self.lines['Ei']) / (4*np.pi)

        return

    def generate(self, xrange: tuple, res: int = 1000, size: int = None, full_output: bool = False):

        if size is None:
            size = int(res*20)

        # create table for storing spectrum
        x    = np.linspace(xrange[0], xrange[1], size)
        spec = pd.DataFrame({'Wavelen': x, })
        for s in self.specData.index:
            spec[s] = np.zeros_like(x) # intensity for each component spectrum

        # calculate spectrum intensity
        specID = self.lines['Element'] + '-' + self.lines['SpecNum'].map(roman)
        for i in self.lines.index:
            x0 = self.lines.at[i,'Wavelen']
            y  = self.lines.at[i,'Intensity'] * Profiles.gaussian(x, x0, x0 / res)

            spec[ specID[i] ] += y
        
        spec['Sum'] = spec.drop('Wavelen', axis=1).sum(axis=1)

        if full_output:
            return spec
        return spec[['Wavelen', 'Sum']]



if __name__ == '__main__':


    import matplotlib.pyplot as plt
    import seaborn as sn
    plt.style.use('seaborn')

    sg = SpectrumGenerator({'Cu':0.7, 'Sn': 0.25, 'Zn': 0.05}, 1.5, 1e+17)
    spec = sg.generate(xrange=[300, 600], res=5000, full_output=False)

    fig, ax = plt.subplots(1, 1, figsize=[8, 3])
    sn.lineplot(spec, x='Wavelen', y='Sum')
    plt.show()

