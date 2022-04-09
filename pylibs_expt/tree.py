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
        return 'Node(children={})'.format(len(self._child))

    def keys(self) -> tuple:
        """ Return a tuple containing attribute names. """
        ...

    def addchild(self, child: object, key: str = None) -> None:
        """ Add a child node. """
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

    def _repr(self: Node) -> str:
        return '{}({}, children={})'.format(
                                                self.__name__,
                                                ', '.join(
                                                            map(lambda i: f'{i}={getattr(self, i)}', self.keys())
                                                         ),
                                                len(self._child),
                                           )

    return type(
                    name,
                    (Node, ),
                    {
                        '__init__' : _init,
                        '__slots__': attrs,
                        '__name__' : name,
                        '__repr__' : _repr,
                        'keys'     : _keys,
                        **namespace 
                    }
                )
                


