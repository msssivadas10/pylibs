#!/usr/bin/python3
r"""

`pylibs.objects`: Special Data Objects
======================================

The objects module provides specialized objects to store data related to 
spectrum analysis. This include special tables to store the spectroscopic 
data of lines and energy levels of species. It also include functions to 
represent a mixture of multi-species elements as trees. Main use of these 
functionalities are in the internal plasma spectrum calculator and analysis.

`pylibs.objects.table`: Tables
------------------------------

This module contains the :class:`Table` base class. Other special tables 
can be created as subclasses to this. For creating simple tables, use `table` 
function, which creates a subclass of :class:`Table`.

`pylibs.objects.tree` : Trees
-----------------------------

This module can be used to create trees, using instances of :class:`Node` 
or its subclasses.

"""

from pylibs.objects._tables import LinesTable, LevelsTable
from pylibs.objects._elemtree import SpeciesNode, ElementNode
from pylibs.objects._helpers import element, elementTree, elementTree_fromList, elementTree_fromDict
from pylibs.objects._helpers import linestable, levelstable
from pylibs.objects._helpers import loadtxt

__all__ = ['table', 'tree', ]