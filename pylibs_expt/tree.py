from typing import Any, Iterable, Type, Union

class NodeError(Exception):
    """ Base class of exceptions used by nodes. """
    ...

class Node:
    """
    A node of a tree. Every node will have a set of attributes and some 
    child nodes.
    """
    __slots__ = '_child', '_childkey'
    __name__  = 'Node'

    def __init__(self, *args, **kwargs) -> None:
        self._child    = []
        self._childkey = [] 
    
    def __repr__(self) -> str:
        return '{}({}, children={})'.format(
                                                self.__name__,
                                                ', '.join(
                                                            map(lambda i: f'{i}={getattr(self, i)}', self.keys())
                                                         ),
                                                len(self._child),
                                           )
    
    def __getitem__(self, __key: str) -> Any:
        if __key not in self.keys():
            raise NodeError("node do not have an attribute '{}'".format(__key))
        return getattr(self, __key)

    def __setitem__(self, __key: str, __value: Any) -> Any:
        if __key not in self.keys():
            raise NodeError("node do not have an attribute '{}'".format(__key))
        return setattr(self, __key, __value)

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

    @property
    def nchildren(self) -> int:
        """ Number of children. """
        return len(self._child)

def node(name: str, attrs: Iterable[str], namespace: dict = {}) -> Type[Node]:
    """
    Create a subclass of :class:`Node` to store specific values.
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
