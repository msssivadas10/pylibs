from typing import Any, Iterable, List, Type
from collections import namedtuple
import numpy as np, re, json

class TableError(Exception):
    """ Base class exceptions used by table objects. """
    ...

class Table:
    """ 
    A class for table objects. Tables are 2D data storage objects, storing the 
    data as columns. These columns can be accessed by the dot (`.`) operator 
    like the attributes or using `[]` operators as keys. Use the `subset` 
    property to get a subset of the table.

    This class is does not define the table methods. All the tables should be 
    created as subclasses of this class. Use the `table` function for creating 
    specialized tables. A table can be extended only appending rows to the end 
    of the table. To add a column to the table, a new class of tables must be 
    created.

    Parameters
    ----------
    *args, **kwargs: array_like, (key, array_like) pairs
        Columns of the table. These must be iterables. If keyword arguments are 
        used, then the keys should correspond to the column names. All these 
        columns should have same size and 1D.

    Examples
    --------
    All tables must be subclasses of table objects. Such a table can be created 
    using the `table` function. e.g., to create a table with two columns x and 
    y, called `my_table`:

    >>> my_table = table('my_table', ['x', 'y'])
    >>> t1       = my_table([1.0, 2.0, 3.0], [0.9, 2.1, 3.2])
    >>> t1
    <table 'my_table': cols=('x', 'y'), shape=(3, 2)>

    """
    __slots__ = '_nr', '_nc', '_subset_getter', 
    __name__  = ''

    def __init__(self, *args, **kwargs) -> None:
        self._nr: int = ...
        self._nc: int = ...
        self._subset_getter: TableSubsetGetter = ...

    @property
    def shape(self) -> tuple:
        """ Shape of the table. """
        return self._nr, self._nc

    @property
    def nr(self) -> int:
        """ Number of rows in the table. """
        return self._nr

    @property
    def nc(self) -> int:
        """ Number of rows in the table. """
        return self._nc

    @property
    def colnames(self) -> tuple:
        """ Names of columns in this table. """
        ...
    
    @property
    def subset(self) -> object:
        """ 
        An object to get a subset of the table. This object will be an instance 
        of :class:`TableSubsetGetter` and a table with the required rows and 
        columns can be extracted using its `__getitem__()` method or `[]`
        operator. If the key is:

        1. A 2-tuple, it will be of the form (rows, columns).
        2. An integer, slice or boolean array, specify the required rows.
        3. A string or list of string, specify the columns.

        If row (or column) key is not provided, then the subset will have all the 
        rows (or columns) in the previous table.

        Examples
        --------
        >>> t1.subset['x']
        <table 'my_table': cols=('x',), shape=(3, 1)>
        >>> t1.subset[:2]
        <table 'my_table': cols=('x', 'y'), shape=(2, 2)>
        >>> t1.subset[:2, ['x']]
        <table 'my_table': cols=('x',), shape=(2, 1)>
        
        """
        ...

    def append(self, *args, **kwargs) -> None:
        """ Append a row to the table. """
        ...

    def hascolumn(self, __key: str) -> bool:
        """ Check if the table has a column with name `__key`.  """
        ...

    def typeof(self, __key: str, __type: Any = ...) -> Any:
        """ 
        Get or set the type of column. If a second argument is given, then convert 
        the column to that type. Otherwise, return the type of that column.

        Parameters
        ----------
        __key: str
            Column name, whose type is needed (to change).
        __type: type, str specifying the type, optional
            Type to which the column is converted.

        Returns
        -------
        __type: numpy.dtype, None
            Type of the column (if not `__type` argument is given), else None.
        """
        ...

    def r(self, __i: int) -> tuple:
        """ 
        Get a specified row in the table.  
        
        Parameters
        ----------
        __i: int
            Index of the row.

        Returns
        -------
        row: tuple
            A namedtuple containing the values in the row. Its field names 
            correspond to the column names.
        """
        ...

    def print(self, fmt: Iterable[str], sep: str = ' ', lnchr: str = '-') -> None:
        """ Print the table. """
        ...

class TableSubsetGetter:
    """ 
    Instances of this class are used by table objects to get subsets of a table. 
    """
    __slots__ = '_table', 

    def __init__(self, table: Table) -> None:
        self._table = table

    def __getitem__(self, __key: Any) -> Any:
        if isinstance(__key, tuple):
            if len(__key) == 1:
                __cols  = self._table.colnames 
                __rows, = __key
            elif len(__key) == 2:
                __rows, __cols = __key
                if isinstance(__cols, str):
                    __cols = [__cols, ]
                elif isinstance(__cols, list):
                    if not min(map(lambda __o: isinstance(__o, str), __cols)):
                        raise TypeError("column names must be a 'str'")
                else:
                    raise KeyError("column key must be 'str' or a 'list' of 'str'")
            else:
                raise KeyError("too many indices for table")
        else:
            if isinstance(__key, str):
                __rows, __cols = slice(None), [__key, ]
            elif isinstance(__key, list):
                if not min(map(lambda __o: isinstance(__o, str), __key)):
                    raise TypeError("column names must be a 'str'")
                __rows, __cols = slice(None), __key
            else:
                __rows, __cols  = __key, self._table.colnames

        cls = table(self._table.__name__, __cols)
        return cls(**{__col: self._table[__col][__rows] for __col in __cols})

class JSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for table objects.
    """

    def default(self, o: Any) -> Any:
        if not isinstance(o, Table):
            return super().default(o)
        
        # json format of the table is {"key": ["type", *data], ...}
        _dict = {}
        for key in o.colnames:
            value = o[key]

            # get column type name:
            tp_name  = value.dtype.name
            tp, size =re.search(r'([a-z]+)(\d*)', tp_name).groups()
            if tp not in ['bool', 'int', 'float', 'complex', 'str']:
                raise TypeError(f"type '{tp_name}' is not supported")
            
            _dict[key] = [tp_name, *list(value)]
        return _dict

def table(name: str, cols: Iterable[str]) -> Type[Table]:
    """ 
    Create a table type with given column fields. These tables are subclasses 
    of :class:`Table`.

    Parameters
    ----------
    name: str
        Typename for the new table subclass.
    cols: list of str
        Column names for the table. Must be non-empty and cannot have duplicates.

    Returns
    -------
    table_t: type, subclass of :class:`Table`
        A table type with given column fields.

    Examples
    --------
    >>> my_table = table('my_table', ['x', 'y'])
    >>> t1       = my_table([1.0, 2.0, 3.0], [0.9, 2.1, 3.2])
    >>> t1
    <table 'my_table': cols=('x', 'y'), shape=(3, 2)>

    """
    _ncols = len(cols)
    if not _ncols:
        raise TableError("cols cannot be empty")
    elif not min(map(lambda __o: isinstance(__o, str), cols)):
        raise TableError("cols must be an iterable of 'str'")
    elif len(cols) != len(set(cols)):
        raise TableError("column names must be unique")

    table_row = namedtuple('table_row', cols)

    def _init(self: Table, *args, **kwargs) -> None:
        self._nr, self._nc = 0, _ncols
        args = dict(zip(self.__slots__, args))
        for __name in self.__slots__:
            __value = []
            if __name in args.keys():
                if __name in kwargs.keys():
                    raise TableError("got multiple values for column '{}'".format(__name))
                __value = args[__name]
            elif __name in kwargs.keys():
                __value = kwargs[__name]
            __value = np.asarray(__value)
            if __value.ndim != 1:
                raise TableError("each column must be one-dimensional arrays")
            if not self._nr:
                self._nr = __value.shape[0]
            elif __value.shape[0] != self._nr:
                raise TableError("all columns must have the same size")
            setattr(self, __name, __value)

        self._subset_getter = TableSubsetGetter(self)

    def _repr(self: Table) -> str:
        return f"<table '{self.__name__}': cols={self.colnames}, shape={self.shape}>"

    def _append(self: Table, *args, **kwargs) -> None:
        """ append a row at the end of the table. """
        args = dict(zip(self.colnames, args))
        for __name in self.colnames:
            if __name in args.keys():
                if __name in kwargs.keys():
                    raise TableError("got multiple values for column '{}'".format(__name))
                __value = args[__name]
            elif __name in kwargs.keys():
                __value = kwargs[__name]
            else:
                raise TableError("missing value for column '{}'".format(__name))
            setattr(self, __name, np.append(getattr(self, __name), __value))
        self._nr += 1

    def _hascolumn(self: Table, __key: str) -> bool:
        """ check if a table has the specified column. """
        if not isinstance(__key, str):
            raise TableError("key must be a string")
        elif __key in self.colnames:
            return True
        return False

    def _getitem(self: Table, __key: str) -> Any:
        """ get the specified column. """
        if self.hascolumn(__key):
            return getattr(self, __key)
        raise TableError("no column called '{}'".format(__key))

    def _typeof(self: Table, __key: str, __type: Any = ...) -> Any:
        """ get or set the type of a column. """
        if self.hascolumn(__key):
            if __type is ... : 
                return getattr(self, __key).dtype
            setattr(self, __key, getattr(self, __key).astype(__type))
        raise TableError("no column called '{}'".format(__key))

    def _subset(self: Table) -> tuple:
        """ get the specified row. """
        return self._subset_getter

    def _colnames(self: Table) -> tuple:
        """ get the column names. """
        return self.__slots__

    def _r(self: Table, __i: int) -> tuple:
        """ get a specified row in the table. """
        return self.table_row(
                                **{
                                    __col: self[__col][__i] for __col in self.colnames
                                  }
                             )

    def _print(self: Table, fmt: List[str], sep: str = ' ', lnchr: str = '-') -> None:
        """ print the table. """
        __fmt_pattern = r'(?<={:)([\^\<\>\s]*)([\+\-\s]*)(\d*)([\.,]*)(\d*)([bcdoxXneEfFgGs\s]*)(?=})'

        if not isinstance(fmt, list):
            raise TypeError("fmt must be a list")
        elif len(fmt) != self.nc:
            raise TableError("not enough format specifiers")
        __val_fmt, __head_fmt = [], []
        for __fmt in fmt:
            if not isinstance(__fmt, str):
                raise TypeError("format specifier must be a 'str'")
            __match = re.search(__fmt_pattern, __fmt)
            if not __match:
                raise TableError(f"invalid format specifier '{__fmt}'")
            _align, _sign, _width, _dsep, _prec, _pres = __match.groups()
            __val_fmt.append(__fmt)
            __head_fmt.append(''.join(['{:{fill}', _align if _align else '>', _width, 's}']))

        val_fmt, head_fmt = sep.join(__val_fmt), sep.join(__head_fmt)

        # print column names:
        print(head_fmt.format(*self.colnames, fill = ''))

        # print a line:
        print(head_fmt.format(*[''] * self.nc, fill = lnchr))

        # print rows:
        for i in range(self.nr):
            print(val_fmt.format(*self.r(i)))
        
    return type(
                    name, 
                    (Table, ),
                    {
                        '__slots__'   : tuple(cols),
                        '__name__'    : name,
                        'table_row'   : table_row,
                        '__init__'    : _init,
                        '__repr__'    : _repr,
                        '__getitem__' : _getitem,
                        'append'      : _append,
                        'hascolumn'   : _hascolumn,
                        'typeof'      : _typeof,
                        'subset'      : property(_subset, ),
                        'colnames'    : property(_colnames, ),
                        'r'           : _r, 
                        'print'       : _print,
                    }
                )

def extent(old: Type[Table], cols: Iterable[str], name: str = ..., ) -> Type[Table]:
    """
    Extent an existing table type with new columns. This can be used to get the 
    functionality equivalent to adding new columns. Since a table type does not 
    support adding new columns to it, a new table type is created with columns 
    from the old table and the extra columns given. 

    Parameters
    ----------
    old: type, subclass of :class:`Table`
        Table type to be extented.
    cols: list or str
        Names of the extra columns. Must be different from the columns in the old.
    name: str, optional
        Type name for the new table type. If not given, use the same name as the 
        previous.

    Returns
    -------
    table_t: type, subclass of :class:`Table`
        A table type extented from the old.

    Examples
    --------
    >>> old_table = table('old_table', ['x', 'y']) # old table
    >>> new_table = extent(old_table, ['z', 'w'])  # new table
    >>> 
    >>> # create a table of 'new_table' format:
    >>> t = new_table([0., 1.], [0., 1.], [0., 1.], [0., 1.])
    >>> t
    <table 'old_table': cols=('x', 'y', 'z', 'w'), shape=(2, 4)>

    """
    if not issubclass(old, Table):
        raise TypeError("old must be a subclass of 'Table'")
    if not cols:
        raise ValueError("cols cannot be empty")

    name = old.__name__ if name is ... else name
    cols = [_name for _name in [*old.__slots__, *cols] if _name not in Table.__slots__]
    return table(name, cols)

def create(table_t: Type[Table], *args, **kwargs) -> Table:
    """
    Create a table of given type and fill values to it. These values are taken 
    from the other tables (given as positional arguments) and arrays (given as 
    keyword arguments, where the key is the column name). This can be used to 
    initialise a table extented from an existing one. For example,

    >>> # create atable with two columns x and y
    >>> my_table = table('my_table', ['x', 'y'])
    >>> t1       = my_table([1.0, 2.0, 3.0], [0.9, 2.1, 3.2])
    >>> # extent the table by adding another column z
    >>> my_table2 = extent(my_table, ['z'])
    >>> # fill the table with data from the old table and an z array
    >>> t2 = create(my_table2, t1, z = [0.3, 0.1, 0.2])
    >>> t2
    <table 'my_table': cols=('x', 'y', 'z'), shape=(3, 3)>

    """
    if not issubclass(table_t, Table):
        raise TypeError("table_t must be a subclass of 'Table'")
    
    # get column names:
    cols  = {_name: None for _name in table_t.__slots__ if _name not in Table.__slots__}
    nrows = 0

    # get columns from the given tables:
    for _tb in args:
        if not isinstance(_tb, Table):
            raise TypeError("argument must be a 'Table' object")
        for key in _tb.colnames:
            # check if this column is present in the output table
            if key not in cols.keys():
                continue

            # check this column is already set, meaning a previous table 
            # has a column with this name, which is not allowed 
            if cols[key] is not None:
                raise TableError(f"multiple tables has column '{key}'")
            cols[key] = _tb[key]

            if not nrows:
                nrows = _tb.nr
            elif nrows != _tb.nr:
                raise TableError("all tables must have the same number of rows")

    # get other columns:
    for key, value in kwargs.items():
        if key not in cols.keys():
            raise TableError(f"no column called '{key}' in the table")
        if not nrows:
            nrows = len(value)
        elif nrows != len(value):
            raise TableError("length of the array must be equal to the number of rows")
        
        cols[key] = value

    # check if all columns are set:
    for key, value in cols.items():
        if value is None:
            raise TableError(f"no values are given for column '{key}'")

    return table_t(**cols) # make the table
                



if __name__ == "__main__":
    my_table = table('my_table', ['x', 'y'])
    t1       = my_table([1.0, 2.0, 3.0], [0.9, 2.1, 3.2])

    my_table2 = extent(my_table, ['z'])

    t2 = create(my_table2, t1, z = [0.3, 0.1, 0.2])
    print(json.dumps(t2, cls = JSONEncoder, indent = 4))

    