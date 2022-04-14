r"""

Miscellanious Objects
======================

This module contains things that not belong to other categories. It includes some 
functions and data. Data are stored as dicts and include elements and spectra. There 
are also some pre-defined plasma type available.

"""

import struct
import numpy as np
from pylibs.objects import ElementNode, SpeciesNode, levelstable, linestable
from pylibs.plasma import plasma
from typing import Any, Iterable, Dict

def packElement(en: ElementNode) -> bytes:
    """ 
    Pack an element node into a `bytes` object. 
    """
    out, size = b'', 0

    if not isinstance( en, ElementNode ):
        raise TypeError("object must be an 'ElementNode'")

    key_size = min( len(en.key), 10 )
    key_rem  = max( 10 - key_size, 0 )

    out += struct.pack(
                            '10sdi',
                            bytes( en.key[ :key_size ] + ' ' * key_rem, 'utf-8' ),
                            en.m,
                            en.nspec
                      )
    
    size += struct.calcsize( '10sdi' )

    for sn in en.children():
        out  += struct.pack( 'dii', sn.Vs, sn.lines.nr, sn.levels.nr )
        size += struct.calcsize( 'dii' )
        
        for i in range( sn.lines.nr ):
            out  += struct.pack( 
                                    'dddid', 
                                    sn.lines.wavelen[i],
                                    sn.lines.aki[i],
                                    sn.lines.ek[i],
                                    int( sn.lines.gk[i] ),
                                    sn.lines.errAki[i],
                              )
            size += struct.calcsize( 'dddid' )

        for i in range( sn.levels.nr ):
            out  += struct.pack( 'id', int( sn.levels.g[i] ), sn.levels.value[i] )
            size += struct.calcsize( 'id' )

    return out 
    
def unpackElement(buf: bytes, interp: bool = True, T: Any = None) -> ElementNode:
    """ 
    Unpack a `bytes` object and create an element node from the data. 
    """
    start, end = 0, 0

    end           = start + struct.calcsize( '10sdi' )
    key, m, nspec = struct.unpack( '10sdi', buf[ start:end ] )
    key           = key.decode( 'utf-8' ).strip()
    start         = end

    en = ElementNode( key, m )

    for s in range( nspec ):
        end                 = start + struct.calcsize( 'dii' )
        Vs, nlines, nlevels = struct.unpack( 'dii', buf[ start:end ] )
        start               = end

        # lines ( cols: wavelen, aki, Ek, gk, err )
        fmt   = 'dddid' * nlines
        end   = start + struct.calcsize( fmt )
        lnt   = np.asfarray( struct.unpack( fmt, buf[ start:end ] ) ).reshape( ( nlines, 5 ) )
        start = end

        # levels ( cols: g, value )
        fmt   = 'id' * nlevels
        end   = start + struct.calcsize( fmt )
        lvt   = np.asfarray( struct.unpack( fmt, buf[ start:end ] ) ).reshape( ( nlevels, 2 ) )
        start = end

        en.addspecies(
                        SpeciesNode( 
                                        s, 
                                        Vs, 
                                        levelstable( lvt ), 
                                        linestable( 
                                                        lnt,
                                                        np.repeat( s,   nlines ),
                                                        np.repeat( key, nlines ),
                                                  ),
                                        interp, 
                                        T,
                                    )
                     )
    return en

def createElementLibrary(file: str, __elem: Iterable[ElementNode]) -> None:
    """ 
    Create a library of elements and save. 
    """
    if not len( __elem ):
        return
    elif False in map( lambda o: isinstance( o, ElementNode ) ):
        raise TypeError("object should be an 'ElementNode'")

    data, size = b'', []
    for xi in __elem:
        di = packElement( xi )

        size.append( len( di ) )
        data += di

    head = struct.pack( 
                            'i' * ( len(size) + 1 ), 
                            len( __elem ), 
                            *size
                      )
    
    with open( file, 'wb' ) as f:
        f.write( head + data )
    
    return

def loadElementLibrary(file: str) -> Dict[str, ElementNode]:
    """ 
    Read a file and load data from a library of elements. 
    """
    elem = {}

    with open(file, 'rb') as f:
        buf = f.read()

        start, end = 0, 4
        nelem,     = struct.unpack( 'i', buf[ start:end ] ) 
        start      = end 

        fmt        = 'i' * nelem
        end        = start + struct.calcsize( fmt )
        size       = struct.unpack( fmt, buf[ start:end ] )
        start      = end

        for i in range( nelem ):
            end   = start + size[ i ]

            x = unpackElement( buf[ start:end ] )

            elem[ x.key ] = x

            start = end 
    
    return elem

# =====================================================================
# Stock data
# ===================================================================== 

stockElement = loadElementLibrary( 'pylibs/misc/stock_elements.dat' )

stockPlasma  = {
                    'ctplasma' : plasma( 
                                            'ctplasma', 
                                            [ 
                                                stockElement[ 'cu' ],
                                                stockElement[ 'sn' ],
                                            ] 
                                        ),
                    'ctzplasma': plasma(
                                            'ctzplasma',
                                            [
                                                stockElement[ 'cu' ],
                                                stockElement[ 'sn' ],
                                                stockElement[ 'zn' ],
                                            ]
                                        ),
                }

stockSpectrum = {}