
import sys, warnings, traceback

from collections.abc import Generator, Iterable, Sequence, MutableSequence
from types import ModuleType
import numbers
from array import array, typecodes as array_typecodes

from .__pixelmaps__ import operandzip, Stop, Repeat, Previous


ARRAY_CODES = dict(Int8=dict(code='b', min=-0x80, max=0x7f),
                   UInt8=dict(code='B', min=0, max=0xff),
                   Int16=dict(code='i', min=-0x8000, max=0x7fff),
                   UInt16=dict(code='I', min=0, max=0xffff),
                   Int32=dict(code='l', min=-0x80000000, max=0x7fffffff),
                   UInt32=dict(code='L', min=0, max=0xffffffff),
                   Int64=dict(code='q',
                              min=-0x8000000000000000,
                              max=0x7fffffffffffffff),
                   UInt64=dict(code='Q',
                               min=0,
                               max=0xffffffffffffffff),
                   Float32=dict(code='f', min=None, max=None),
                   Float64=dict(code='d', min=None, max=None))


class Number(object):

    _Minumum = None
    _Maximum = None
    _TypeCode = None

    @classmethod
    def _Overflow(cls, value):
        if cls._Minumum is not None and value < cls._Minumum:
            warnings.warn("{} value {} was clamped to minimum {}"
                          .format(cls.__name__, value, cls._Minumum))
            value = cls._Minumum

        elif cls._Maximum is not None and value > cls._Maximum:
            warnings.warn("{} value {} was clamped to maximum {}"
                          .format(cls.__name__, value, cls._Maximum))
            value = cls._Maximum
        return value

    def __new__(cls, value):
        try:
            return super().__new__(cls, cls._Overflow(value))
        except TypeError as te:
            print("cls.__name__", cls.__name__)
            print("type(cls._Overflow).__name__", type(cls._Overflow).__name__)
            traceback.print_exc(file=sys.stdout)
            raise te
            #traceback.print_stack()


def number_type_factory(name, code=None, min=None, max=None, bases=None):
    if name in ARRAY_CODES:
        details = ARRAY_CODES[name]
        if (code is not None and code != details["code"]) \
                or (min is not None and min != details["min"]) \
                or (max is not None and max != details["max"]):
            raise ValueError("Cannot vary definition for predefined type "
                             + name)
        code, min, max = (details[k] for k in ('code', 'min', 'max'))
    if bases is None:
        if code is not None and code in 'bBuhHiIlLqQ':
            bases = (Number, int)
        elif code is not None and code in 'fd':
            bases = (Number, float)
        else:
            raise ValueError("bases are required when not using a code")
    elif not any(isinstance(b, Number) for b in bases):
        bases = (Number,) + tuple(bases)
    attrs = dict(_Minumum=min, _Maximum=max, _TypeCode=code)
    cls = type(name, bases, attrs)
    assert issubclass(cls, numbers.Number)
    return cls


class NumberTypesModule(ModuleType):

    __file__ = __file__
    __path__ = __file__[:__file__.rindex('__types__.py')-1]
    __package__ = "numtyped.numbertypes"

    def __getattr__(self, name):
        if name in ARRAY_CODES:
            #print("creating {}".format(name))
            setattr(self, name, number_type_factory(name))
            return getattr(self, name)
        raise AttributeError("module 'numbertypes' has no attribute '{}'"
                             .format(name))


numbertypes = NumberTypesModule("numtyped.numbertypes")
sys.modules['numtyped.numbertypes'] = numbertypes


class TypedArray(array, MutableSequence):

    _TypeCode = None
    _ItemType = None

    def __new__(cls, initializer, *args, length=None, **kwargs):
        """
        Creates a TypedArray.

        If length is None or not given then initializer must be an
        Iterable of numeric values, the number of items consumed will
        be the length.  If length is an int then initializer may be any
        numeric value or Iterable of them, and the TypedArray will be
        initialized to the length given.  TypedArray itself is abstract
        and must be subclassed and implement both "_TypeCode" and
        "_ItemType", where _TypeCode is an array code and _ItemType is
        either a number type or a function returning a numeric value.

        Arguments:
            initializer (any:)  The initialization values.
            length (int|None:)  The length of the TypedArray (optional)
        """
        #print("TypedArray.__new__", cls, "initializer", args, kwargs)
        if isinstance(length, int):
            itemtype = cls._ItemType
            if not isinstance(initializer, Iterable):
                initializer = [initializer]
            values = (v
                      for (i, v) in operandzip(range(length),
                                               initializer,
                                               defaults=(Stop, Repeat)))
        elif length is None:
            values = initializer
        else:
            raise ValueError("length must be an int or None, but found {}"
                             .format(length))
        return array.__new__(cls, cls._TypeCode, values)

    @property
    def itemtype(self):
        return self._ItemType

    def __init__(self, *args, **kwargs):
        #print("TypedArray.__init__", self, "initializer", args, kwargs)
        return array.__init__(self)

    def __key__(self, key):
        if key is None:
            key = len(self)
        elif key is Ellipsis:
            key = slice(None)
        if isinstance(key, (int, slice, Iterable)):
            return key
        raise IndexError("expects int, None, Ellipses, slice or "
                         "iterable, but found {}"
                         .format(type(key).__name__))

    def __getitem__(self, key):
        indexes = self.__key__(key)
        cls = self.__class__
        if isinstance(indexes, slice):
            return cls(array.__getitem__(self, indexes))
        if isinstance(indexes, Iterable):
            return cls(array.__getitem__(self, i) for i in indexes)
        return array.__getitem__(self, indexes)

    def __delitem__(self, key):
        indexes = self.__key__(key)
        if isinstance(indexes, Iterable):
            for i in reversed(indexes):
                array.__delitem__(self, i)
        return array.__delitem__(self, indexes)

    def __setitem__(self, key, values):
        itemtype = self.itemtype
        indexes = self.__key__(key)
        if isinstance(indexes, int) and not isinstance(values, Iterable):
            return array.__setitem__(self, itemtype(values))
        if isinstance(indexes, slice) and isinstance(values, Iterable):
            if not isinstance(values, array):
                values = array(self.typecode,
                               (itemtype(v) for v in values))
            return array.__setitem__(self, indexes, values)
        self.fill(values, key=indexes)

    def __repr__(self):
        return "{}([{}])".format(type(self).__name__,
                                 ', '.join(repr(i) for i in self))

    def fill(self, value, *, key=slice(None)):
        if isinstance(key, slice):
            key = range(*key.indices(len(self)))
        if not isinstance(key, Iterable):
            raise ValueError("key must be a slice or Iterable")
        if isinstance(value, Iterable):
            for (i, v) in operandzip(key, value,
                                     defaults=(Stop, Repeat)):
                array.__setitem__(self, i, v)
        else:
            for i in key:
                super().__setitem__(i, value)


def typed_array_factory(name, itemtype=None, code=None, bases=()):
    itemname = name[:-5]
    if name.endswith("Array") and itemname in ARRAY_CODES:
        details = ARRAY_CODES[itemname]
        if itemtype is None:
            itemtype = getattr(numbertypes, name)
        if (code is not None and code != details["code"]):
            raise ValueError("Cannot vary definition for predefined type "
                             + name)
        code = details['code']
    elif code is None:
        if not isinstance(itemtype, Number) \
                or itemtype._TypeCode not in array_typecodes:
            raise ValueError("Invalid type code {}".format(code))
        code = itemtype._TypeCode
    if not any(isinstance(b, TypedArray) for b in bases):
        bases = (TypedArray,) + tuple(bases)
    attrs = dict(_ItemType=itemtype, _TypeCode=code)
    #print("type(name, bases, attrs)", name, bases, attrs)
    return type(name, bases, attrs)


class TypedArrayModule(ModuleType):

    __file__ = __file__
    __path__ = __file__[:__file__.rindex('__types__.py')-1]
    __package__ = "numtyped.typedarrays"

    def __getattr__(self, name):
        itemname = name[:-5]
        #print(name, itemname)
        if itemname in ARRAY_CODES:
            #print("creating {}".format(name))
            itemtype = getattr(numbertypes, itemname)
            #print("itemtype", itemtype)
            setattr(self, name, typed_array_factory(name, itemtype))
            return getattr(self, name)
        raise AttributeError("module 'typedarrays' has no attribute '{}'"
                             .format(name))


typedarrays = TypedArrayModule("numtyped.typedarrays")
sys.modules['numtyped.typedarrays'] = typedarrays

