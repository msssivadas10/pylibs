import numpy as np
from pylibs.objects._tables import LinesTable, LevelsTable
from pylibs.objects._elemtree import SpeciesNode, ElementNode
from pylibs.objects.tree import Node
from typing import Any, Iterable, Sequence, Union


# =========================================================================================
# Functions to create special tables
# =========================================================================================
 
def linestable(a: Any, s: Any = None, elem: Any = None, columns: dict = None) -> LinesTable:
    """  
    Create a :class:`LinesTable` object from an array.

    Parameters
    ----------
    a: array_like
        A 2D array of at least 4 columns to get the required columns of the lines 
        table (`wavelen`, `aki`, `ek` and `gk`). Other columns will be mapped to 
        `errAki`, `I`, `boltzX` and `boltzY`. Column order can be specified by the 
        `columns` parameter. 
    s: array_like, None, optional
        Species column (type: `int`). 
    elem: array_like, None, optional
        Element columns (type: `str`).
    columns: dict, list, optional
        Column mapping. If a list given, then entries will correspond to the column 
        names and mapped to column of `a` specified by the index. If a dict, then 
        a key-value pair correspond to column name and index. 

    Returns
    -------
    t: LinesTable
        Table created from the input array(s).

    Examples
    --------
    todo

    """
    if columns is None:
        columns = ['wavelen', 'aki', 'ek', 'gk', 'errAki', 'I', 'boltzX', 'boltzY']

    if isinstance(columns, list):
        columns = dict( zip( columns, range( len( columns ) ) ) )
    elif not isinstance(columns, dict):
        raise TypeError("columns must be a 'list' or 'dict'")

    a = np.asfarray(a)
    if np.ndim(a) != 2:
        raise TypeError("array must be a 2D array")
    elif a.shape[1] < 4:
        raise TypeError("not enough columns in the input array")

    table = {}
    for key, i in columns.items():
        if i >= a.shape[1]:
            continue
        table[ key.lower() ] = a[:,i]
    
    table[ 's' ] = None
    if s is not None:
        s = np.asarray(s).astype('int')
        if np.ndim(s) != 1:
            raise TypeError("s must be a 1D array")
        elif s.shape[0] != a.shape[0]:
            raise TypeError("size mismatch for s array")
        table[ 's' ] = s

    table[ 'elem' ] = None
    if elem is not None:
        elem = np.asarray(elem).astype('str')
        if np.ndim(elem) != 1:
            raise TypeError("elem must be a 1D array")
        elif elem.shape[0] != a.shape[0]:
            raise TypeError("size mismatch for elem array")
        table[ 'elem' ] = elem

    if 'erraki' not in table.keys():
        table[ 'erraki' ] = None

    lt = LinesTable(
                        wavelen = table[ 'wavelen' ],
                        Aki     = table[ 'aki' ],
                        Ek      = table[ 'ek' ],
                        gk      = table[ 'gk' ],
                        elem    = table[ 'elem' ],
                        s       = table[ 's' ],
                        errAki  = table[ 'erraki' ],
                    )
    
    if 'i' in table.keys():
        lt.setLineIntensity( table[ 'i' ] )
    if 'boltzx' in table.keys() and 'boltzy' in table.keys():
        lt.setBoltzmannXY( table[ 'boltzx' ], table[ 'boltzy' ] )
    
    return lt

def levelstable(a: Any) -> LevelsTable:
    """
    Create a :class:`LevelsTable` object from an array of shape (n,2).

    Parameters
    ----------
    a: array_like
        A 2D array of of 2 columns. The first column correspond to g values and 
        the second column is the energy level values in eV.

    Returns
    -------
    t: LevelsTable
        Table created from the input array.

    Examples
    --------
    todo

    """
    a = np.asfarray(a)

    if np.ndim(a) != 2:
        raise TypeError("array must be a 2D array")
    elif a.shape[1] != 2:
        raise TypeError("array should have 2 columns")
    
    return LevelsTable( a[:,0], a[:,1] )


# =========================================================================================
# Functions to create special nodes and trees
# =========================================================================================

def element(key: str, m: float, nspec: int, Vs: Sequence[float], levels: Sequence[LevelsTable], lines: Union[LinesTable, Sequence[LinesTable]], interpolate: bool = True, T: Any = None) -> ElementNode:
    """ 
    Create a new element node with species. 

    Parameters
    ----------
    key: str
        Key used to specify the element.
    m: float
        Atomic mass in amu.
    nspec: int 
        Number of species of this element.
    Vs: sequence of float
        Ionization energy in eV. Must be a sequence of length `nspec`.
    levels: sequence of LevelsTable
        Energy levels of the species. Must be a sequence of length `nspec`.
    lines: LinesTable, sequence of LinesTable
        Spectral lines of this element. If a sequence is given, must have length 
        `nspec` and each table correspond to a species, in their order.
    interpolate: bool, optional
        If set true (default), use interpolation table to calculate partition 
        function.
    T: array_like, optional
        Array of temperature values to create interpolation table.

    Returns
    -------
    node: ElementNode
        A tree representing element, with each branch correspond to a species.
    
    """
    key = key.lower()

    if m < 0.0:
        raise ValueError("atomic mass cannot be negative")

    if nspec < 1:
        raise ValueError("there should be at least one species")

    if len(Vs) != nspec:
        raise ValueError("incorrect number of ionization energies, must be same as the number of species")
    Vs = np.asfarray(Vs)

    if len(levels) != nspec:
        raise ValueError("incorrect number of energy levels, must be same as the number of species")

    if isinstance(lines, LinesTable):
        if lines.s is None:
            raise ValueError("table should have species column ('s')")

        # check if the table has `elem` column
        if lines.elem is None:
            lines.elem = np.repeat(elem.key, lines.nr)

        _lines = []
        for s in range(nspec):
            _lines.append( lines.slice( ( lines.elem == key ) & ( lines.s == s ) ) )
        lines = _lines

    else:
        if len(lines) != nspec:
            raise ValueError("incorrect number of lines tables, must be same as the number of species")
        elif False in map(lambda o: isinstance(o, LinesTable), lines):
            raise TypeError("lines lust be an array of 'LinesTable'")

        for s, _lines in enumerate(lines):

            # check if the table has `elem` column else add
            if _lines.elem is None:
                _lines.elem = np.repeat(elem.key, _lines.nr)

            # check if the table `s` column else add
            if _lines.s is None:
                _lines.s = np.repeat(s, _lines.nr)

    elem = ElementNode(key, m)
    for s in range(nspec):
        elem.addspecies( SpeciesNode(s, Vs[s], levels[s], lines[s], interpolate, T) )

    return elem

def elementTree(__in: Any, interpolate: bool = True, T: Any = None):
    """
    Create a tree elements. Input may be a list of :class:`ElementNode` or a 
    dict of specific format. If it is a dict, then the top level key-value pairs 
    correspond to element key-data pairs. Each element data should be a dict with 
    fields `m` for atomic mass and `species` for species data. Species data shold 
    be a dict with fields `Vs`-ionization energy, `levels`-energy level table and 
    `lines`-spectral lines table.  

    Parameters
    ----------
    __in: list, dict
        Element data. If a list is given, entries should be :class:`ElementNodes`. If 
        the input is a :class:`Node` object, then check if it is a proper element tree 
        and return itself.
    interpolate: bool, optional
        Use interpolated partition function values (default). Used if a dict input is
        given.
    T: array_like, optional
        Temperature values uaed to interpolate. Default is 101 points in [0,5] interval. 
        Used if a dict input is given.
    """
    if isinstance(__in, ( list, tuple )):
        return elementTree_fromList(__in)
    elif isinstance(__in, dict):
        return elementTree_fromDict(__in, interpolate, T)
    elif isinstance(__in, Node):
        # check if a non-empty element tree
        fail = False 
        msg  = ""
        if len(__in.children()) == 0:
            msg  = "empty tree"
            fail = True
        else:
            for c in __in.children():
                if not isinstance(c, ElementNode):
                    fail = True
                    msg  = "child node is not an 'ElementNode'"
                    break
                for gc in c.children():
                    if not isinstance(gc, SpeciesNode):
                        fail = True
                        msg  = "child node of node '{}' is not a 'SpeciesNode'".format(c.key)
                        break
        if fail:
            raise TypeError("input is not a proper element tree: {}".format( msg ))
        return __in
    raise TypeError("input must be a 'dict' or 'list'")

def elementTree_fromList(__nodes: Iterable[ElementNode]):
    """
    Create a tree of elements. The input must be an array of :class:`ElementNode` 
    objects and returns a :class:`Node` object with each element as branches. 
    """
    if len(__nodes) == 0:
        raise TypeError("input cannot be empty")
    
    root = Node()
    for elem in __nodes:
        if not isinstance(elem, ElementNode):
            raise TypeError("input list must contain only 'ElementNode' objects")
        root.addchild( elem, key = elem.key )
    
    return root

def elementTree_fromDict(__dict: dict, interpolate: bool = True, T: Any = None) -> Node:
    """ 
    Create a tree of elements. Data are taken from the input dict. For this to work, 
    the input dict must be of special format. Top level key-value pairs correspond 
    to element key-data pairs. Each element data should be a dict with fields `m` 
    for atomic mass and `species` for species data. Species data shold be a dict with 
    fields `Vs`-ionization energy, `levels`-energy level table and `lines`-spectral 
    lines table.  

    Parameters
    ----------
    __dict: dict
        Element tree data. A dict in the specified format.
    interpolate: bool, optional
        Use interpolated partition function values (default).
    T: array_like, optional
        Temperature values uaed to interpolate. Default is 101 points in [0,5] interval.
    
    Returns
    -------
    rood: Node
        Root node of the element tree.

    """
    def species_fromDict(__o: dict, elem: str) -> SpeciesNode:
        """ A species node from dict. """
        data = {}
        for key, value in __o.items():
            if key == 'key':
                data[ 'key' ] = value
            elif key == 'Vs':
                if not isinstance( value, (float, int) ):
                    raise TypeError("ionisation energy 'Vs' must be a number")
                data[ 'Vs' ] = value
            elif key == 'levels':
                if not isinstance(value, LevelsTable):
                    try:
                        value = levelstable(value)
                    except Exception:
                        raise TypeError("levels must be a 'LevelsTable' or a 2D array")
                data[ 'levels' ] = value
            elif key == 'lines':
                if not isinstance(value, LinesTable):
                    raise TypeError("lines must be a 'LinesTable'")

                data[ 'lines' ] = value
            else:
                raise KeyError("invalid key: '{}'".format(key))

        # check if all field are present
        for key in [ 'key', 'Vs', 'levels', 'lines' ]:
            if key not in data.keys():
                raise KeyError("cannot find field: '{}'".format(key))

        # filter lines
        lines           = data[ 'lines' ]
        data[ 'lines' ] = lines.slice( ( lines.elem == elem ) & ( lines.s == data[ 'key' ] ) )

        data[ 'interpolate' ] = interpolate
        data[ 'T' ]           = T
        return SpeciesNode( **data )

    def element_fromDict(__o: dict) -> ElementNode:
        """ An element node from dict. """
        data = {}
        for key, value in __o.items():
            if key == 'key':
                data[ 'key' ] = value
            elif key == 'm':
                if not isinstance(value, (float, int)):
                    raise TypeError("atomic mass 'm' must be a number")
                data[ 'm' ] = value 
            elif key not in [ 'species', 'lines' ]:
                raise KeyError("invalid key: '{}'".format(key))
        # check if all field are present
        for key in [ 'key', 'm' ]:
            if key not in data.keys():
                raise KeyError("cannot find field: '{}'".format(key))

        # get species
        species = []
        if 'species' in __o.keys():
            for __s in __o[ 'species' ]:
                if not isinstance(__s, dict):
                    raise TypeError("entries in species must be 'dict'")
                species.append(__s)

        # if a lines table is given, extract lines of this specific element
        if 'lines' in __o.keys():
            lines = __o[ 'lines' ]
            if not isinstance(lines, LinesTable):
                raise TypeError("lines must be a 'LinesTable'")
            
            # get a subset of lines: lines of this element only
            if lines.elem is None:
                lines.elem = np.repeat( data[ 'key' ], lines.nr )
            else:
                lines = lines.slice( lines.elem == data[ 'key' ] )

            for s in range( len( species ) ):
                if 'lines' in species[s].keys():
                    continue
                if lines.s is None:
                    raise KeyError("lines should have species column: 's'")
                species[s][ 'lines' ] = lines.slice( lines.s == s )

        e = ElementNode( **data )
        for s in range( len( species ) ):
            species[s][ 'key' ] = s
            e.addspecies( species_fromDict( species[s], data[ 'key' ] ) )

        return e

    if not isinstance(__dict, dict):
        raise TypeError("input must be a 'dict'") 

    elem = []
    for key, value in __dict.items():
        if not isinstance(value, dict):
            raise TypeError("element data should be a dict")

        value[ 'key' ] = key 
        elem.append( element_fromDict( value ) )
    
    return elementTree_fromList( elem )

# =========================================================================================
# Functions for file reading
# =========================================================================================

def loadtxt(file: str, delim: str = ',', comment: str = '#', regex: str = None, ignore: bool = True, convert: Any = False) -> Any:
    """
    Load data from a text file. Can be used to read data from delimited text files or data 
    stored in a specific pattern. 

    Parameters
    ----------
    file: str
        Path to the file to read. 
    delim: str, optional
        Delimiter to use. Default is `,`.
    comment: str, optional
        Charecter used to comment. Default is `#`.
    regex: str, optional
        If a string is given use it as a regular expression pattern. This pattern is used to 
        parse lines in the file. 
    ignore: bool, optional
        If true, then ignore any non-matching lines or rows of incorrect size (default).
    convert: str, list, bool
        If true, try to convert to a float ndarray. If it is a list, then its size must be same 
        as the number of columns and each entry should be a typename. If different types, then 
        the output is transposed. If a string is used, its charecters will be mapped to a type 
        as `{f: float, d: int, s: str, c: complex}`.

    Returns
    -------
    data: list, numpy.ndarray
        If no conversion is used, return an array of type `str`. If converted to different types, 
        then a list of converted columns are returned. If all types are same, return an array of 
        that type.

    """
    data = []

    with open(file, 'r') as f:
        if regex is None:
            size = 0
            for __line in f.read().splitlines():
                __line = __line.strip()
                if __line.startswith( comment ):
                    continue
                __row = __line.split( delim )
                if size:
                    if len( data[-1] ) != len( __row ):
                        if ignore:
                            continue
                        raise ValueError("row size mismatch: {} and {}".format( len( data[-1] ), len( __row ) ))
                data.append( __row )
                size += 1
        else:
            import re 

            if not isinstance( regex, str ):
                raise TypeError("regex must be 'str'")
            for __line in f.read().splitlines():
                __line = __line.strip()
                if __line.startswith( comment ):
                    continue
                m = re.search( regex, __line )
                if not m:
                    if ignore:
                        continue
                    raise ValueError("cannot find a match to the pattern")
                data.append( m.groups() )

    data = np.array( data )
    
    if isinstance( convert, str ):
        convert = [{'f': 'float', 'd': 'int', 's': 'str', 'c': 'complex'}[c] for c in convert]

    if convert is True:
        data = data.astype( 'float' )
    elif isinstance( convert, list ):
        if len(convert) != data.shape[1]:
            raise TypeError("convert do not have enough entries: should have {}".format( data.shape[1] ))
        data = list( data.T )
        for i in range( len( convert ) ):
            try:
                data[i] = data[i].astype( convert[i] )
            except Exception:
                raise RuntimeError("error converting column {} to '{}'".format( i, convert[i] ))
        
        if len( set( convert ) ) == 1:
            data = np.array( data ).T
    else:
        raise TypeError("convert must be a 'bool', 'str' or 'list' of types")
        
    return data



    