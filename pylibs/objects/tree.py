from typing import Any, Iterable, Type, Union

class NodeError(Exception):
    """ Base class of exceptions used by nodes. """
    ...

class Node:
    """
    A node of a tree. Every node will have a set of attributes and some child nodes. 
    The base class :class:`Node` do not have any attributes. To add attributes to a 
    node, make subclass of node with those attributes. This can be done simply by 
    using the `node` function. Attributes can be accessed using the dot `.` operator 
    and child nodes can be accessed using the `Node.child` method.

    Parameters
    ----------
    *args, **kwargs: 
        Values of the attributes of a node.

    Examples
    --------
    To create a node type called `node1` with fields `x` and `y`, use the `node` function:

    >>> node1 = node( 'node1', [ 'x', 'y' ] )
    >>> root  = node1( x = 0.0, y = 0.0 )

    To add child node, `n1` using the `Node.addchild` method:

    >>> root.addchild( node1( x = 2.0, y = 5.0 ) )

    To access the first child node,

    >>> root.child( 0 )
    node1(x=2.0, y=5.0, children=0)
    
    """
    __slots__ = '_child', '_childkey'
    __name__  = 'Node'

    def __init__(self, *args, **kwargs) -> None:
        self._child    = []
        self._childkey = [] 
    
    def __repr__(self) -> str:
        return '{}({})'.format(
                                    self.__name__,
                                    ', '.join(
                                                list(
                                                        map(
                                                                lambda i: f'{i}={getattr(self, i)}', 
                                                                self.keys()
                                                           )
                                                    ) + [ f'children={len(self._child)}' ]
                                              )
                              )
    
    def __getitem__(self, __key: str) -> Any:
        if __key not in self.keys():
            raise NodeError("node do not have an attribute '{}'".format(__key))
        return getattr(self, __key)

    def __setitem__(self, __key: str, __value: Any) -> Any:
        if __key not in self.keys():
            raise NodeError("node do not have an attribute '{}'".format(__key))
        return setattr(self, __key, __value)

    def copy(self):
        copy = type( self )()
        for key in self.keys():
            setattr( copy, key, getattr( self, key ) )
        
        copy._childkey = [ key for key in self._childkey ]
        copy._child    = [ ch.copy() for ch in self._child ]
        return copy

    def keys(self) -> tuple:
        """ Return a tuple containing attribute names. """
        return tuple()

    def addchild(self, child: object, key: str = None) -> None:
        """ 
        Add a child node. If given `key` is used to access the child node. 
        Otherwise, an integer index is used. 
        """
        if key is None:
            key = len(self._child)
        if not isinstance(child, Node):
            raise TypeError("child must be a 'Node'")
        self._child.append(child)
        self._childkey.append(key)

        if True in map( lambda o: isinstance( o, str ), self._childkey ):
            self._childkey = list( map( str, self._childkey ) )

    def child(self, key: Union[int, str] = 0) -> object:
        """ Get a child node. If no keys are given, get the first """
        if isinstance(key, str):
            if key not in self._childkey:
                raise NodeError("invalid node key: '{}'".format(key))
            key = self._childkey.index(key)
        elif not isinstance(key, int):
            raise TypeError("key must be 'str' or 'int'")
        if not -1 < key < len(self._child):
            return None
        return self._child[key]

    def children(self) -> list:
        """ Get all the children. """
        return self._child

    @property
    def nchildren(self) -> int:
        """ Number of children. """
        return len(self._child)

def node(name: str, attrs: Iterable[str], namespace: dict = {}) -> Type[Node]:
    """
    Create a subclass of :class:`Node` to store specific values.

    Parameters
    ----------
    name: str
        Name of the new :class:`Node` type.
    attrs: sequence of str
        A non-empty sequence of attribute names. 
    namespace: dict
        Extra things to add to the namespace of new node.

    Returns
    -------
    node: Node
        New node type.

    Examples
    --------
    >>> node1 = node( 'node1', [ 'x', 'y' ] )
    >>> root  = node1( x = 0.0, y = 0.0 )
        
    """
    if len(attrs) < 1:
        raise TypeError("attrs cannot be empty")

    def _init(self: Node, *args, **kwargs) -> None:
        args = dict(zip(self.keys(), args))
        for __name in self.keys():
            __value = None
            if __name in args.keys():
                if __name in kwargs.keys():
                    raise TypeError("got multiple values for argument '{}'".format(__name))
                __value = args[__name]
            elif __name in kwargs.keys():
                __value = kwargs[__name]
            else:
                raise TypeError("missing value for argument '{}'".format(__name))
            setattr(self, __name, __value)

        Node.__init__(self)
    
    def _keys(self: Node) -> tuple:
        return self.__slots__

    return type(
                    name,
                    (Node, ),
                    {
                        '__init__' : _init,
                        '__slots__': attrs,
                        '__name__' : name,
                        'keys'     : _keys,
                        **namespace 
                    }
                )

