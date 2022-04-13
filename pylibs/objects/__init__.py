#!/usr/bin/python3
r"""

Objects Module
==============

The `pylibs.objects` module contains various objects for data storage, such as tables and 
trees. These are not optimized objects, but simple. :class:`Table` and :class:`Node` are 
the basic objects. The first can be used as a base class for tables and the other can be 
used as the base for creating nodes, which make up a tree (nodes connected as branches). 
There are also some functions to make the creation of these objects easier.

"""

from pylibs.objects.special import LinesTable, LevelsTable, SpeciesNode, ElementNode
from pylibs.objects.helpers import element, elementTree, elementTree_fromList, elementTree_fromDict
from pylibs.objects.helpers import linestable, levelstable
from pylibs.objects.helpers import loadtxt

__all__ = ['table', 'tree', ]