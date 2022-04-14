#!/usr/bin/python3
import numpy as np
from pylibs.objects import linestable, levelstable, loadtxt

def loadlines(file: str):
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

    # if select is not None:
    #     i = np.zeros(elem.shape, dtype = 'bool')
    #     for __e in select:
    #         i = i | ( elem == __e.lower() )

    #     x, Aki, Ek, gk, elem, sp, acc = x[i], Aki[i], Ek[i], gk[i], elem[i], sp[i], acc[i]

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