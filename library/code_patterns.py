# noinspection PyMissingConstructor
class AttDict(dict):
    """
    Object that adds to a regular dict the ability to refer to the keys as if they were attributes.
    """
    """
    Doctest:
    >>> d_ = {'orange': 'acid', \
              'pear':'sweet'}
    >>> d = AttDict(d_)
    >>> d['chocolate'] = 'super_yummy!'
    >>> d.chocolate
    'super_yummy!'
    >>> d['chocolate']
    'super_yummy!'
    >>> {k:v for k, v in d.items()}
    {'orange': 'acid', 'pear': 'sweet', 'chocolate': 'super_yummy!'}
    """
    def __init__(self, d, **kwargs):
        super().__init__(**kwargs)
        for att_name in d.keys():
            super().__setitem__(att_name, d[att_name])

    def __getattr__(self, item):
        return self[item]  # if item in self.keys() else dict.__getattr__(item)


class Singleton(object):
    """
    Clase singleton que puede heredarse por otras clases para otorgarles comportamiento singleton.
    """
    def __new__(cls, *p, **k):
        if '_the_instance' not in cls.__dict__:
            cls._the_instance = object.__new__(cls)
        return cls._the_instance


def memodict(f):
    """
    Decorador de funciones que cachea el resultado para funciones con un sólo argumento.
    """
    class MemoDict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return MemoDict().__getitem__


# def lazy_property(fn):
#     """
#     Decorator that makes a property lazy-evaluated.
#     """
#     attr_name = '_lazy_' + fn.__name__
#
#     @property
#     def _lazy_property(self):
#         if not hasattr(self, attr_name) or getattr(self, attr_name) is None:
#             setattr(self, attr_name, fn(self))
#         return getattr(self, attr_name)
#     return _lazy_property

class lazy_property(object):
    """A decorator that converts a function into a lazy property.  The
    function wrapped is called the first time to retrieve the result
    and then that calculated result is used the next time you access
    the value::
        class Foo(object):
            @cached_property
            def foo(self):
                # calculate something important here
                return 42
    The class has to have a `__dict__` in order for this property to
    work.
    """

    # implementation detail: this property is implemented as non-data
    # descriptor.  non-data descriptors are only invoked if there is
    # no entry with the same name in the instance's __dict__.
    # this allows us to completely get rid of the access function call
    # overhead.  If one choses to invoke __get__ by hand the property
    # will still work as expected because the lookup logic is replicated
    # in __get__ for manual invocation.

    _missing = '_missing'

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, self._missing)
        if value is self._missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value
