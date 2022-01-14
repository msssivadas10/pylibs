#!/usr/bin/python3

from typing import Any
import numpy as np
import re

class TableError(Exception):
    """ Exception used by :class:`Table` class """
    ...

class Table:
    r"""
    A table object is a 2D container of data as columns. 

    Parameters
    ----------
    data: dict, tuple, array_like
        Data to make the table. If it is a :class:`dict`, then the keys are used as column 
        labels and values are used as columns. If it is a :class:`tuple`, then it should be a 
        tuple of columns. If it is a :class:`numpy.ndarray`, then columns are used as columns 
        of the table.

    colnames: list
        Column labels. If not given, a label of the form `c{index}` is given. It should be 
        given if data is not given.

    coltypes: list
        Data type of the columns. If not given, it is infered from the data. It should be 
        given if data is not given.

    Raises
    ------
    :class:`TableError`

    Examples
    --------
    From a :class:`dict`, 

    >>> x = Table({'x': [1., 2., 3.], 'y': [2., 4., 6.]})
    >>> x
    <Table shape = [3, 2], columns = [x: float, y: float]>

    and, from a :class:`tuple`,

    >>> x = Table(([1., 2., 3.], [2., 4., 6.]), colnames = ['x', 'y'])
    >>> x
    <Table shape = [3, 2], columns = [x: float, y: float]>

    and, from a numpy array (or list),

    >>> x = Table(np.array([[1., 2.], [2., 4.], [3., 6.]]), colnames = ['x', 'y'], coltypes = [float, float])
    >>> x
    <Table shape = [3, 2], columns = [x: float, y: float]>

    All these will create the same table. If generating from a list or numpy array, it is 
    recommended to specify the column types.

    """
    __slots__ = 'colnames', 'coltypes', 'data', 'shape'

    def __init__(self, data: Any = ..., colnames: list = ..., coltypes: list = ...) -> None:
        if data is Ellipsis:
            # no data: column names should given
            if colnames is Ellipsis:
                # raise TableError("names should be given if no data")
                colnames = []
            if coltypes is Ellipsis:
                coltypes = [float for _ in colnames]
            data = [list() for _ in colnames]
        elif isinstance(data, dict):
            # dict: keys are column names, values are data
            colnames = list(data.keys())
            data     = list(data.values())
        else:
            if isinstance(data, (np.ndarray, list)):
                # numpy array: column names given or computed
                data = np.asarray(data)
                if data.ndim != 2:
                    raise TableError("`data` should be a 2 dimensional array")
                data = list(data.T)
            elif isinstance(data, tuple):
                # list: column names given or computed
                lengths = list(map(len, data))
                if min(lengths) != max(lengths):
                    raise TableError("columns of different sizes")
                data = list(data)
            else:
                raise TypeError("invalid value for `data`")

            # set/compute column names
            if colnames is Ellipsis:
                colnames = [f'c{i}' for i in range(len(data))]
            else:
                if len(colnames) != len(data):
                    raise TableError("number of columns and number of names not match")
            
        ncols = len(data)
        nrows = min(map(len, data)) if ncols else 0
        if coltypes is Ellipsis:
            coltypes = list(map(self.__get_type, data))
            if None in coltypes:
                raise TableError("table entries should be numeric or string types")
        elif len(coltypes) != ncols:
            raise TableError("number of columns and number of types not match")

        self.colnames = colnames
        self.coltypes = coltypes
        self.data     = list(map(np.array, data, coltypes))
        self.shape    = [nrows, ncols]         

    def __get_type(self, _data: Any) -> Any:
        _types = [int, float, str, None]

        def __type_id(_item):
            if isinstance(_item, (int, np.int64, np.int32)):
                return 0
            elif isinstance(_item, (float, np.float64, np.float32)):
                return 1
            elif isinstance(_item, str):
                return 2
            return 3                

        return _types[max(map(__type_id, _data))]   

    def __compute_shape(self, ) -> None:
        ncols      = len(self.data)
        nrows      = len(self.data[0]) if ncols else 0
        self.shape = [nrows, ncols]

    def __repr__(self) -> str:
        def __col_sign(key, tp):
            _tp = re.search(r"(?<=\')\w+", repr(tp)).group(0)
            return f"{key}: {_tp}"
        x = ', '.join(map(__col_sign, self.colnames, self.coltypes))
        return f"<Table shape = {self.shape}, columns = [{x}]>"

    def copy(self, ):
        """ Return a copy of the table """
        return Table({key: self.c(key).copy() for key in self.colnames}, coltypes = self.coltypes)

    def print(self) -> None:
        """
        Print the formatted table.
        """
        nrows = min(100, self.shape[0])

        # create the template strings ... 
        tmp = " ".join(
                    [
                        "".join(["[{:", f"{len(str(nrows))}d", "}]"]), 
                        " ".join(map(lambda t: "{:>10s}" if t is str else "{:10.3g}", self.coltypes))
                    ]
                ) # ... for rows
        hln = " ".join(
                    [
                        "".join(["{:^", f"{len(str(nrows)) + 2}", "}"]).format(''), 
                        " ".join([f"{'':=^10}"] * self.shape[1])
                    ]
                ) # ... for line
        col = " ".join(
                    [
                        "".join(["{:^", f"{len(str(nrows)) + 2}", "}"]).format(''), 
                        " ".join(map(lambda x: f"{x[:min(len(x), 8)]:>10}", self.colnames))
                    ]
                ) # ... for column heading 


        # print table
        print()
        print(col)
        print(hln)
        for i, row in enumerate(zip(*self.data)):
            if i > nrows:
                break
            print(tmp.format(i, *row))
        return   

    def typeof(self, key: str) -> type:
        """
        Get the type of the specified column.

        Parameters
        ----------
        key: str
            Name of the column to get the type.

        Returns
        -------
        dtype: type
            Type of the column.

        Examples
        --------
        >>> x = Table({'x': [1., 2., 3.], 'y': [2., 4., 6.]})
        >>> x.typeof('x')
        <class 'float'>

        """
        if key not in self.colnames:
            raise TableError(f"no column called `{key}`")
        return self.coltypes[self.colnames.index(key)]

    def set_type(self, key: str, dtype: type) -> None:
        """
        Set the type of the specified column.

        Parameters
        ----------
        key: str
            Name of the column to set the type.

        dtype: type
            Type of the column.

        Examples
        --------
        >>> x = Table({'x': [1., 2., 3.], 'y': [2., 4., 6.]})
        >>> x.typeof('x')
        <class 'float'>
        >>> x.set_type('x', int)
        >>> x.typeof('x')
        <class 'int'>

        """
        if key not in self.colnames:
            raise TableError(f"no column called `{key}`")
        i = self.colnames.index(key)
        if dtype not in [int, float, str]:
            _tp = re.search(r"(?<=\')\w+", repr(dtype)).group(0)
            raise TableError(f"invalid value for dtype, '{_tp}'")
        self.data[i].astype(dtype)
        self.coltypes[i] = dtype
        return

    def c(self, key: str) -> Any:
        """
        Get the specified column from the table.

        Parameters
        ----------
        key: str
            Name of the column.

        Returns
        -------
        retval: array_like
            Specified column.

        Examples
        --------
        >>> x = Table({'x': [1., 2., 3.], 'y': [2., 4., 6.]})
        >>> x.c('x')
        array([1., 2., 3.])
        """
        if key not in self.colnames:
            raise TableError(f"no column called `{key}`")
        return self.data[self.colnames.index(key)]

    def r(self, i: int) -> tuple:
        """
        Get the specified row from the table.

        Parameters
        ----------
        i: int
            Index of the row.

        Returns
        -------
        retval: tuple
            Specified row.

        Examples
        --------
        >>> x = Table({'x': [1., 2., 3.], 'y': [2., 4., 6.]})
        >>> x.r(1)
        (2.0, 4.0)
        """
        if not isinstance(i, int):
            raise TypeError("index should be an integer")
        if i < 0:
            i += self.shape[0]
        if i < 0 or i > self.shape[0]-1:
            raise IndexError("index out of bound")
        return list(zip(*self.data))[i]

    def set(self, key: str, i: int, value: Any) -> None:
        """
        Set the value of the specified cell.

        Parameters
        ----------
        key: str
            Name of the column.

        i: int
            Index of the row.
        
        value: int, float, str
            Value to set.

        Examples
        --------
        >>> x = Table({'x': [1., 2., 3.], 'y': [2., 4., 6.]})
        >>> x.set('x', 0, 1.23) # replace row-0, column-'x' with 1.23

        """
        if key not in self.colnames:
            raise TableError(f"no column called `{key}`")
        key = self.colnames.index(key)
        if not isinstance(i, int):
            raise TypeError("index should be an integer")
        if i < 0:
            i += self.shape[0]
        if i < 0 or i > self.shape[0]-1:
            raise IndexError("index out of bound")
        try:
            value = self.coltypes[key](value)
        except Exception:
            raise TypeError("cannot convert values to column types")
        self.data[key][i] = value
        return

    def get(self, key: str, i: int) -> Any:
        """
        Get the value of the specified cell.

        Parameters
        ----------
        key: str
            Name of the column.

        i: int
            Index of the row.
        
        Returns
        -------
        retval: int, float, str
            Value at the specified location.

        Examples
        --------
        >>> x = Table({'x': [1., 2., 3.], 'y': [2., 4., 6.]})
        >>> x.get('x', 0) # get row-0, column-'x'
        1.0
        
        """
        if key not in self.colnames:
            raise TableError(f"no column called `{key}`")
        key = self.colnames.index(key)
        if not isinstance(i, int):
            raise TypeError("index should be an integer")
        if i < 0:
            i += self.shape[0]
        if i < 0 or i > self.shape[0]-1:
            raise IndexError("index out of bound")
        return self.data[key][i]

    def insert_column(self, data: Any, i: int = ..., name: str = ..., dtype: type = ...) -> None:
        """
        Insert a column at the given position.

        Parameters
        ----------
        data: array_like
            Column values. It should have the same size as other columns.

        i: int, optional
            Position to insert the column. If not given, insert to the end of the table.

        name: str, optional
            Name of the column. If not specified, the name `c{pos}` is used.

        dtype: type, optional
            Type of the column. If not given, it is inferred from the values.

        Examples
        --------
        >>> x = Table({'x': [1., 2., 3.], 'y': [2., 4., 6.]})
        >>> x.insert_column([5., 10., 15.], 0, 'z')
        >>> x
        <Table shape = [3, 3], columns = [z: float, x: float, y: float]>

        """
        if name in self.colnames:
            raise TableError(f"already have column with name `{name}`")
        if i is Ellipsis:
            i = self.shape[1]
        dtype = self.__get_type(data) if dtype is Ellipsis else dtype
        if dtype is None:
            raise TableError("table entries should be numeric or string types")
        name = f"c{self.shape[1]}" if name is Ellipsis else name
        data = np.asarray(data, dtype = dtype).flatten()
        if data.shape[0] != self.shape[0] and self.shape[1]:
            raise TableError(f"incorrect size ({data.shape[0]} instead of {self.shape[0]})")
        self.data.insert(i, data)
        self.colnames.insert(i, name)
        self.coltypes.insert(i, dtype)
        if not self.shape[1]:
            self.shape[0] = len(data)
        self.shape[1] += 1
        return  list.insert

    def add_column(self, data: Any, name: str = ..., dtype: type = ...) -> None:
        """
        Add a new column to the end of the table.

        Parameters
        ----------
        data: array_like
            Column values. It should have the same size as other columns.

        name: str, optional
            Name of the column. If not specified, the name `c{pos}` is used.

        dtype: type, optional
            Type of the column. If not given, it is inferred from the values.

        Examples
        --------
        >>> x = Table({'x': [1., 2., 3.], 'y': [2., 4., 6.]})
        >>> x.add_column([5., 10., 15.], 'z')
        >>> x
        <Table shape = [3, 3], columns = [x: float, y: float, z: float]>
        
        """
        if name in self.colnames:
            raise TableError(f"already have column with name `{name}`")
        dtype = self.__get_type(data) if dtype is Ellipsis else dtype
        if dtype is None:
            raise TableError("table entries should be numeric or string types")
        name = f"c{self.shape[1]}" if name is Ellipsis else name
        data = np.asarray(data, dtype = dtype).flatten()
        if data.shape[0] != self.shape[0] and self.shape[1]:
            raise TableError(f"incorrect size ({data.shape[0]} instead of {self.shape[0]})")
        self.data.append(data)
        self.colnames.append(name)
        self.coltypes.append(dtype)
        if not self.shape[1]:
            self.shape[0] = len(data)
        self.shape[1] += 1
        return

    def add_row(self, row: list) -> None:
        """
        Add a new row to the end of the table.

        Parameters
        ----------
        row: array_like
            Row values. It should have the same size as other rows.

        Examples
        --------
        >>> x = Table({'x': [1., 2., 3.], 'y': [2., 4., 6.]})
        >>> x.add_row([4., 8.])
        >>> print(x)
        <Table shape = [4, 2], columns = [x: float, y: float]>
        
        """
        if len(row) != self.shape[1]:
            raise TableError("incorrect number of values")
        if False in map(isinstance, row, self.coltypes):
            try:
                row = tuple(map(lambda v, __t: __t(v), row, self.coltypes))
            except Exception:
                raise TypeError("cannot convert values to column types")
        for i in range(self.shape[1]):
            self.data[i] = np.append(self.data[i], row[i])
        self.shape[0] += 1
        return

    def replace_column(self, key: str, data: Any) -> None:
        """
        Replace a column in the table.

        Parameters
        ----------
        key: str
            Name of the column to replace.

        data: array_like
            New values of the column. Should have the same size as other columns.

        Examples
        --------
        >>> x = Table({'x': [1., 2., 3.], 'y': [2., 4., 6.]})
        >>> x.replace_column('x', [1.1, 2.2, 3.3])
        >>> x.c('x')
        array([1.1, 2.2, 3.3])

        """
        if key not in self.colnames:
            raise TableError(f"no column called '{key}`")
        try:
            data = np.asarray(data, dtype = self.typeof(key)).flatten()
        except Exception:
            raise TableError("cannot convert values to column type")
        if data.shape[0] != self.shape[0]:
            raise TableError("incorrect size for column")
        self.data[self.colnames.index(key)] = data
        return

    def replace_row(self, i: int, data: list) -> None:
        """
        Replace a row in the table.

        Parameters
        ----------
        i: int
            Index of the row to replace.

        data: array_like
            New values of the row. Should have the same size as other rows.

        Examples
        --------
        >>> x = Table({'x': [1., 2., 3.], 'y': [2., 4., 6.]})
        >>> x.replace_row(2, [3.3, 6.1])
        >>> x.r(2)
        (2.0, 4.0)

        """
        if len(data) != self.shape[1]:
            raise TableError("incorrect number of values")
        if False in map(isinstance, data, self.coltypes):
            try:
                data = tuple(map(lambda v, __t: __t(v), data, self.coltypes))
            except Exception:
                raise TypeError("cannot convert values to column types")
        if not isinstance(i, int):
            raise TypeError("index should be an integer")
        if i < 0:
            i += self.shape[0]
        if i < 0 or i > self.shape[0]-1:
            raise IndexError("index out of bound")
        for c in range(self.shape[1]):
            self.data[c][i] = data[c]
        return

    def remove_column(self, key: str) -> None:
        """
        Remove a column from the table.

        Parameters
        ----------
        key: str
            Name of the column to remove.

        Examples
        --------
        >>> x = Table({'x': [1., 2., 3.], 'y': [2., 4., 6.]})
        >>> x.remove_column('x')
        >>> print(x)
        <Table shape = [3, 1], columns = [y: float]>

        """
        if key not in self.colnames:
            raise TableError(f"no column called '{key}`")
        i = self.colnames.index(key)
        _ = self.coltypes.pop(i)
        _ = self.colnames.pop(i)
        _ = self.data.pop(i)
        self.shape[1] -= 1
        return
    
    def remove_row(self, i: int) -> None:
        """
        Remove a row from the table.

        Parameters
        ----------
        i: int
            Index of the row to remove.

        Examples
        --------
        >>> x = Table({'x': [1., 2., 3.], 'y': [2., 4., 6.]})
        >>> x.remove_row(0)
        >>> print(x)
        <Table shape = [2, 2], columns = [x: float, y: float]>
        
        """
        if not isinstance(i, int):
            raise TypeError("index should be an integer")
        if i < 0:
            i += self.shape[0]
        if i < 0 or i > self.shape[0]-1:
            raise IndexError("index out of bound")
        for c in range(self.shape[1]):
            self.data[c] = np.delete(self.data[c], i)
        self.shape[0] -= 1
        return

    def rename_column(self, key: str, new_key: str) -> None:
        """
        Rename a column.

        Parameters
        ----------
        key: str
            Name of the column to rename.

        new_key: str
            New name of the column.

        Examples
        --------
        >>> x = Table(np.random.random((10, 3)), colnames = ['x', 'y', 'z'])
        >>> x.rename_column('x', 'a')
        >>> x.colnames
        ['a', 'y', 'z']

        """
        if key == new_key:
            return
        if new_key in self.colnames:
            raise TableError(f"a column with name `{new_key} already present")
        self.colnames[self.colnames.index(key)] = new_key
        return
        
    def has_columns(self, keys: list) -> bool:
        """
        Check if there are columns with the given keys.

        Parameters
        ----------
        keys: list
            Keys to check.
            
        Returns
        -------
        retval: bool
            True if all keys are present, else false.

        Examples
        --------
        >>> x = Table(np.random.random((10, 3)), colnames = ['x', 'y', 'z'])
        >>> x.has_columns(['x', 'z'])
        True
        >>> x.has_columns(['a', ])
        False

        """
        return min(map(lambda _key: _key in self.colnames, keys))

    def subtable(self, rows: Any, keys: list = ...):
        """
        Return a part of the table with the given rows and columns.

        Parameters
        ----------
        rows: array_like
            Rows in the new table.

        keys: list, optional
            Columns in the new table. If not given, all columns are used.

        Returns
        -------
        table: :class:`Table`
            New table with given columns and rows.

        Examples
        --------
        >>> x = Table({'x': [1, 2, 3, 4, 5, 6, 7], 'y': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]})
        >>> y = x.subtable([1, 2, 3])
        >>> y.c('x')
        array([2, 3, 4])

        """
        if keys is Ellipsis:
            keys = self.colnames
        elif not self.has_columns(keys):
            raise TableError("`keys` has coulmns not in the table")
        rows        = [rows, ] if isinstance(rows, int) else rows
        table       = Table(colnames = list(keys), coltypes = list(map(self.typeof, keys)))
        table.data  = list(map(lambda _key: self.c(_key)[rows], keys))
        table.__compute_shape()
        return table

    def split(self, key: str, levels: Any) -> tuple:
        """
        Split the table by levels in the specified column.

        Parameters
        ----------
        key: str
            Column name. This column should be of type :class:`int` or :class:`str`.

        levels: array_like
            Levels to split the table. If a value is not present in the column, 
            :class:`TableError` is raised.

        Returns
        -------
        tables: tuple
            Tables created by splitting this table. Number tables will be same as the size of 
            levels list. 

        Examples
        --------
        >>> t = Table({'i': [1, 1, 2, 1, 2], 'x': [1, 2, 3, 4, 5], 'y': np.random.random(5)})
        >>> x1, x2 = t.split('i', [1, 2])
        >>> x1.c('x'), x2.c('x') # check 'x' column 
        array([1, 2, 4]), array([3, 5])

        """
        if key not in self.colnames:
            raise TableError(f"no column called `{key}`")
        keys  = [_key for _key in self.colnames if _key != key]
        types = list(map(self.typeof, keys))

        def __table_from_slice(_lv):
            table       = Table(colnames = keys, coltypes = types)
            rows        = np.where(self.c(key) == _lv)[0]
            if not len(rows):
                raise TableError(f"column `{key}` has no value `{_lv}`")
            table.data  = list(map(lambda _key: self.c(_key)[rows], keys))
            table.shape = [len(rows), self.shape[1] - 1]
            return table

        return tuple(map(__table_from_slice, levels))

    def saveas(self, file: str, delim: str = ',', comment: str = '#', head: bool = True, specify_type: bool = True) -> None:
        """
        Save the table to a text file.

        Parameters
        ----------
        file: str
            Name or path of the file to write.
        
        delim: str, optional
            Charecter to use as delimiter. Default is `,`.

        comment: str, optional
            Charecter to use for comments. Default is `#`.
        
        head: bool, optional
            If True (default), write the column names to the first comment line.
        
        specify_type: bool, optional
            If True (default), include the type to the column names.

        Examples
        --------
        To save the table as a CSV file,

        >>> x = Table(np.random.random((10, 3)), colnames = ['x', 'y', 'z'])
        >>> x.saveas(file = 'x.csv', delim = ',')

        """
        comment, delim = comment + ' ', delim   + ' '

        def __typename(_t: type):
            return re.search(r"(?<=\')\w+", repr(_t)).group(0)

        with open(file, 'w') as f:
            if head:
                if specify_type:
                    f.write(
                            comment + delim.join(
                                map(
                                    lambda _key, _t: ":".join([_key, __typename(_t)]), 
                                    self.colnames, 
                                    self.coltypes
                                )
                            ) + "\n"
                        )
                else:
                    f.write(comment + delim.join(self.colnames) + "\n")

            for row in zip(*self.data):
                f.write(delim.join(map(str, row)) + "\n")

    def join(self, tables: tuple, copy: bool = False):
        """
        Join the given tables to a copy of the current table and return. 

        Parameters
        ----------
        tables: tuple
            Tables to join. Each entry should be a :class:`Table ` object.

        copy: bool, optional
            If true, use copies of the input tables. Default is false.

        Returns
        -------
        table: :class:`Table`
            Table created.

        Examples
        --------
        >>> t = Table({'i': [1, 1, 2, 1, 2], 'x': [1, 2, 3, 4, 5], })
        >>> s = Table({'j': [3, 4, 5, 4, 3], 'y': [4, 2, 5, 6, 8]})
        >>> Table().join((t, s))
        <Table shape = [5, 4], columns = [i: int, x: int, j: int, y: int]>
        
        """
        table = self.copy()
        for _tbl in tables:
            if not isinstance(_tbl, Table):
                raise TypeError("object is not a 'Table'")
            if copy:
                _tbl = _tbl.copy()
            for data, name, tp in zip(_tbl.data, _tbl.colnames, _tbl.coltypes):
                table.add_column(data, name, tp)
        # table.__compute_shape()
        return table
    
    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple):
            # given column and row keys
            if len(key) == 1:
                key = (key[0], ...)
            elif len(key) != 2:
                raise KeyError("incorrect number of keys")

            keys, rows = key
            if rows is Ellipsis:
                rows = list(range(self.shape[0]))
            if isinstance(keys, str):
                keys = [keys, ]
            return self.subtable(rows, keys)
        else:
            # given only column key(s)
            return self.__getitem__((key, ...))

    def __add__(self, other):
        """ Add two tables """
        if not isinstance(other, Table):
            return NotImplemented
        if self.shape[1] != other.shape[1]:
            if not self.shape[1]: # this table is empty
                return other.copy()
            elif not other.shape[1]: # other table is empty
                return self.copy()
            raise TableError("cannot join tables with different number of columns")
        if not self.has_columns(other.colnames):
            raise TableError("some columns has no equivalent column in the other table")
        t = Table()
        for key in self.colnames:
            t.add_column(np.append(self.c(key), other.c(key)), key, )
        return t
      

# ===================================================================================
# File read / write functions
# ===================================================================================

class Parser:
    """
    A parser object to parse data from a line. This used by the `readtxt` function to read data from a file.

    Parameters
    ----------
    ignore_cols: list, optional
        List of column indices to ignore.

    ignore_conds: dict, optional
        A dictionary of conditions to ignore a row. Its keys are the column indices and 
        values are the conditions to check.

    max_cols: int
        Maximum number of columns to read.

    parser: callable, optional
        Function to parse a single value.

    ignore_val: object, optional
        Return value for an ignored row. Default is `None`.

    """
    __slots__ = "ignore_cols", "ignore_conds", "parser", "ignore_val", "max_cols"

    def __init__(self, ignore_cols: list = [], ignore_conds: dict = {}, max_cols: int = 0, parser: object = None, ignore_val: object = None) -> None:
        if not isinstance(ignore_cols, list):
            raise TypeError("`ignore_cols` should be a list")
        if len(ignore_cols):
            if not min(map(lambda _x: isinstance(_x, int), ignore_cols)):
                raise TypeError("`ignore_cols` should be a list of integers")
        self.ignore_cols  = ignore_cols

        if not isinstance(ignore_conds, dict):
            raise TypeError("`ignore_conds` should be a dict")
        if len(ignore_conds):
            if not min(map(lambda _x: callable(_x), ignore_conds.values())):
                raise TypeError("values in `ignore_conds` should be a callable")
        self.ignore_conds = ignore_conds

        if not isinstance(max_cols, int):
            raise ValueError("`max_cols` should be an integer")
        self.max_cols = max_cols

        self.parser = self.__default_parser if parser is None else parser

        self.ignore_val = ignore_val # ignore row if this value returned

    def __default_parser(self, _val: str) -> Any:
        """ Parse a string into integer, float or string types """
        try:    
            return eval(_val)
        except Exception:
            return _val.strip()

    def __call__(self, vals: list) -> Any:
        """ Parser function """
        out = list(map(self.parser, vals)) # parse the values

        if self.max_cols and self.max_cols <= len(out):
            out = out[:self.max_cols]

        # check conditions for ignoring this row
        for i, cond in self.ignore_conds.items():
            if i not in range(len(out)):
                continue
            if cond(out[i]):
                return self.ignore_val

        # remove unwanted columns
        return [_x for i, _x in enumerate(out) if i not in self.ignore_cols]


def readtxt(file: str, delim: str = ',', comment: str = '#', parser: Parser = Parser(), transpose: bool = False) -> list:
    """
    A general function to read data from a text file. The file contain the data in a tabular 
    format, and each entry in a row is seperated by the delimiter charecter.

    Parameters
    ----------
    file: str
        Name or path of the input file.

    delim: str, optional
        Delimiter charecter. By default, comma (`,`) is used, which correspond to CSV files.

    comment: str, optional
        Charecter used for commenting (default is `#`).

    parser: :class:`Parser`, optional
        Parser object used to parse a line from the file. If not given, a default parser is 
        used.

    transpose: bool, optional
        If true (default is false), transpose the table, so that the return value is a list 
        of columns. 

    Returns
    -------
    data: list
        Data read from the file (either as list of rows or columns as specified).

    Examples
    --------
    TODO

    """
    if not isinstance(parser, Parser):
        raise TypeError("`parser` should be a 'Parser' object")

    ncols = 0
    data  = []
    with open(file, 'r') as f:
        for _line in f.read().splitlines():
            _line = _line.strip()
            if _line.startswith(comment):
                continue
            row = parser(_line.split(delim))
            if row == parser.ignore_val:
                continue
            if ncols and len(row) != ncols:
                raise ValueError("column numbers mismatch")
            data.append(row)
            ncols = len(row)

    if transpose:
        tdata = [[] for _ in range(ncols)]
        for row in data:
            for i in range(ncols):
                tdata[i].append(row[i])
        data = tdata
    return data
            

# =====================================================================================
# Shape functions
# =====================================================================================

class Shape:
    """
    A library of line shape functions. All the shape functions here accept four arguments: 
    `x` is the point to evaluate the function, with three parameters `x0` the central 
    wavelength, `y0` the line height and `w` the line width (FWHM). Availabel functions are 

    1. `gaussian`: general Gaussian function.
    2. `lorentzian`: general Lorentzian function.

    One can call the functions as `Shape.func` or use the `shape_` object with the function 
    key as `shape_[key]`. For example, both `Shape.gaussian` and `shape_['gaussian']` call 
    the Gaussian function.

    """
    available = "gaussian", "lorentzian", 
    
    def gaussian(x: Any, x0: float, y0: float, w: float) -> Any:
        r"""
        Gaussian shape function, 

        .. math::
            y = y_0 \exp \left[ -4 \ln 2 \left( \frac{x - x_0}{w} \right)^2 \right]

        where :math:`y_0` is the intensity, :math:`x_0` the center value and :math:`w` the 
        width.

        Parameters
        ----------
        x: array_like
            X values.

        y0: float
            Height or peak line intensity.

        x0: float
            Centre value, correspond to central wavelength. 

        w: float
            Line width.

        Returns
        -------
        y: array_like
            Line profile.
            
        """
        return y0 * np.exp(-2.772588722239781 * ((x - x0) / w)**2)

    def lorentzian(x: Any, x0: float, y0: float, w: float) -> Any:
        r"""
        Lorentzian shape function, 

        .. math::
            y = y_0 \left[ 1 + 4 \left( \frac{x - x_0}{w} \right)^2 \right]^{-1}

        where :math:`y_0` is the intensity, :math:`x_0` the center value and :math:`w` the 
        width.

        Parameters
        ----------
        x: array_like
            X values.

        y0: float
            Height or peak line intensity.

        x0: float
            Centre value, correspond to central wavelength. 

        w: float
            Line width.

        Returns
        -------
        y: array_like
            Line profile.
            
        """
        return y0  / (1. + 4. * ((x - x0) / w)**2)

    def __getitem__(self, key) -> Any:
        """ Get a shape from key """
        if key == "gaussian":
            return Shape.gaussian
        elif key == "lorentzian":
            return Shape.lorentzian
        raise KeyError(f"unknown shape {key}")

shape_ = Shape() # to access functions by keys
