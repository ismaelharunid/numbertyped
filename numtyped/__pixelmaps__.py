
from sys import float_info

import math

try:
    from collections.abc import Generator, Iterable, Sequence, MutableSequence
except ImportError:
    from collections import Generator, Iterable, Sequence, MutableSequence

from itertools import product as iterprod
from functools import reduce

from array import array

import operator

from PIL.Image import open as ImageOpen


NoneType = type(None)
Previous = type("PreviousType", (), {})()
Repeat = type("RepeatType", (), {})()
Stop = type("StopType", (), {})()

def operandzip(*args, defaults=Repeat):
    n_args = len(args)
    defaults = tuple(defaults) if isinstance(defaults, Sequence) else \
            (defaults,) * n_args
    if n_args > len(defaults):
        defaults += (None,) * (n_args - len(defaults))
    elif n_args < len(defaults):
        raise ValueError("too many defaults, {} expected, but there were {}"
                         .format(n_args, len(defaults)))
    stopped = [False] * n_args
    stacks = tuple([] for i in range(n_args))
    it_rows = tuple(iter(it if isinstance(it, Iterable) else (it,))
                    for it in args)
    i_rows = 0
    while True:
        row = []
        for i in range(n_args):
            if not stopped[i]:
                try:
                    value = it_rows[i].__next__()
                    row.append(value)
                    stacks[i].append(value)
                    continue
                except StopIteration:
                    stopped[i] = True
            defval = defaults[i]
            if defval is Repeat:
                value = stacks[i].pop(0)
            elif defval is Previous:
                value = stacks[i].pop()
            elif defval is Stop:
                stopped = [True] * n_args
                break
            elif isinstance(defval, type) and issubclass(defval, Exception):
                raise defval("Looks like this is the end at {}"
                             .format(i_rows))
            elif callable(defval):
                value = defval(i, i_rows)
            else:
                value = defval
            row.append(value)
            stacks[i].append(value)
        if all(stopped):
            return
        yield row
        i_rows += 1


def prod(a):
    return reduce(Ops.mul, a, 1)


def lerp(a, b, t):
    return a * (1 - t) + b * t


def dot(a, b, common=None):
    ac, bc = len(a), len(b)
    if common is None:
        if hasattr(a, "shape"):
            aheight, awidth = a.shape
            if hasattr(a, "flatten"):
                a = a.flatten()
        else:
            aheight, awidth = (ac, 1)
    elif type(common) is int:
        aheight, awidth = (ac // common, common)
    else:
        raise ValueError("ashape expects a 2tuple of ints but found {}"
                         .format(type(ashape).__name__))
    if type(awidth) is not int or awidth <= 0 \
            or type(aheight) is not int or aheight <= 0:
        raise ValueError("awidth and aheight must be ints but found {} and {}"
                         .format(type(awidth).__name__,
                                 type(aheight).__name__))
    if hasattr(b, "shape"):
        bheight, bwidth = b.shape
        if hasattr(b, "flatten"):
            b = b.flatten()
    else:
        bheight, bwidth = (awidth, bc // awidth)
    if type(bwidth) is not int or bwidth <= 0 \
            or type(bheight) is not int or bheight <= 0:
        raise ValueError("bwidth and bheight must be ints but found {} and {}"
                         .format(type(bwidth).__name__,
                                 type(bheight).__name__))
    if type(aheight) is not int or type(bwidth) is not int:
        raise ValueError("aheight and bwidth must be ints but found {} and {}"
                         .format(type(aheight).__name__,
                                 type(bwidth).__name__))
    #print("ashape", awidth, aheight, "bshape", bwidth, bheight)
    if awidth != bheight:
        raise ValueError("awidth must equal bheight, not true for "
                         "a[{}, {}] and b[{}, {}]"
                         .format(awidth, aheight, bwidth, bheight))
    result = []
    for i in range(aheight):
        for j in range(bwidth):
            result.append(sum(a[i*awidth+k] * b[k*bwidth+j]
                              for k in range(awidth)))
    return result, (aheight, bwidth)


CONSTANTS = dict(epsilon = float_info.epsilon,
                 tau = math.tau,
                 pi = math.pi,
                 halfpi = 0.5 * math.pi,
                 e = math.e,
                 inf = math.inf,
                 nan = math.nan)

Constants = type("Constants", (object,), CONSTANTS)

OPERATOR_FN_NAMES = ('__abs__', '__add__', '__contains__', '__eq__',
                     '__floordiv__', '__ge__', '__gt__', '__inv__',
                     '__invert__', '__le__', '__lshift__', '__lt__',
                     '__mod__', '__mul__', '__ne__', '__neg__', '__pos__',
                      '__pow__', '__rshift__', '__sub__', '__truediv__',
                      '__xor__', 'abs', 'add', 'and_', 'contains', 'countOf',
                      'eq', 'floordiv', 'ge', 'gt', 'indexOf', 'inv',
                      'invert', 'is_', 'is_not', 'le', 'lshift', 'lt', 'mod',
                      'mul', 'ne', 'neg', 'not_', 'or_', 'pos', 'pow',
                      'rshift', 'sub', 'truediv', 'truth', 'xor')

MATH_FN_NAMES = ('acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh',
                 'ceil', 'copysign', 'cos', 'cosh', 'degrees', 'erf', 'erfc',
                 'exp', 'expm1', 'fabs', 'factorial', 'floor', 'fmod',
                 'frexp', 'fsum', 'gamma', 'gcd', 'hypot', 'isclose',
                 'isfinite', 'isinf', 'isnan', 'ldexp', 'lgamma', 'log',
                 'log1p', 'log10', 'log2', 'modf', 'pow', 'radians',
                 'remainder', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'trunc')


OTHER_FUNCTIONS = dict(prod = prod,
                       lerp = lerp,
                       )


ITEM_OPERATORS = dict(OTHER_FUNCTIONS)
ITEM_OPERATORS.update((name, getattr(operator, name))
                      for name in OPERATOR_FN_NAMES)
ITEM_OPERATORS.update((name, getattr(math, name))
                      for name in MATH_FN_NAMES)

Ops = type("ItemOperators", (object,), ITEM_OPERATORS)


def userseqop(operator, *args, shape=None,
              itemtype=True, rtype=Previous, **kwargs):
    if type(operator) is str and operator in USERSEQOP_LUT:
        operator = USERSEQOP_LUT[operator]
    if shape is None and hasattr(args[0], "shape"):
        shape = args[0].shape
    if itemtype is True:
        itemtype = args[0].itemtype if hasattr(args[0], "itemtype") else None
    elif itemtype is not None and not callable(itemtype):
        raise ValueError("itemtype must be a type or callable, but found {}"
                         .format(itemtype))
    if rtype is Previous:
        rtype = type(args[0])
    print("userseqop", operator, args, shape, itemtype, rtype, kwargs)
    if callable(operator):
        if itemtype is None:
            gen = (operator(*items, **kwargs)
                   for items in operandzip(*args))
        else:
            gen = (itemtype(operator(*items, **kwargs))
                   for items in operandzip(*args))
        #print("rtype", rtype)
        if rtype is None:
            return gen
        if shape is None:
            return rtype(gen)
        return rtype(gen, shape=shape)
    raise ValueError("operator must be callable")


def create_seqop(name, func, doc=None):
    seqop = lambda *args, **kwargs: userseqop(func, *args, **kwargs)
    seqop.__name__ = name
    seqop.__doc__ = func.__doc__ if doc is None else doc
    return seqop


SEQUENCE_OPERATORS = dict((name, create_seqop(name, meth))
                          for (name, meth) in ITEM_OPERATORS.items())


SeqOps = type("SequenceOperators", (object,), SEQUENCE_OPERATORS)


class TypedArray(array, MutableSequence):

    _TypeCode = None

    @property
    def TypeCode(cls):
        return cls._TypeCode

    _ItemType = None

    @property
    def ItemType(cls):
        return cls._ItemType

    def __new__(cls, initializer, *args, **kwargs):
        print("UInt8Array.__new__", cls, "initializer", args, kwargs)
        if isinstance(initializer, (int, float)):
            count = initializer
            initializer = (0 for i in range(count))
        return array.__new__(cls, cls._TypeCode, initializer)

    @property
    def itemtype(self):
        return self._ItemType

    def __init__(self, initializer, *args, **kwargs):
        print("UInt8Array.__init__", self, "initializer", args, kwargs)
        return array.__init__(self)


class NDimItems(object):
    
    def __new__(cls, *args, shape=None, **kwargs):
        print("NDimItems.__new__", cls, args, shape, kwargs)
        if len(args) == 0 and isinstance(shape, Sequence) \
                and all(isinstance(i, (int, float)) for i in shape):
            args = (prod(shape),)
        self = super().__new__(cls, *args, **kwargs)
        return self

    @classmethod
    def fromimagefile(cls, imagepath):
        im = ImageOpen(imagepath)
        imdata = im.getdata()
        shape = tuple(reversed(imdata.size)) + (imdata.bands,)
        print(cls.__name__, shape)
        return cls((v for p in im.getdata() for v in p), shape=shape)

    _shape = None

    @property
    def shape(self):
        return self._shape

    _strides = None

    @property
    def strides(self):
        return self._strides

    _base = None

    def __init__(self, *args, shape=None, **kwargs):
        print("NDimItems.__init__", self, args, shape, kwargs)
        if len(args) == 0 and isinstance(shape, Sequence) \
                and all(isinstance(i, (int, float)) for i in shape):
            args = (prod(shape),)
        super().__init__(*args, **kwargs)
        length = len(self)
        #print("length", length)
        if shape is None:
            width = round(length ** (1/2))
            shape = (width, width, None)
        elif not isinstance(shape, Sequence) \
                or 1 >= len(shape) or len(shape) > 4 \
                or not all(isinstance(i, (int, float, NoneType))
                           for i in shape):
            raise ValueError("Bad shape {}".format(shape))
        p = prod(1 if i is None else int(i) for i in shape)
        self._shape = tuple(int(length // p if i is None else i)
                            for i in shape)
        print("self._shape", self._shape)
        p = prod(1 if i is None else int(i) for i in self._shape)
        if p != length:
            raise ValueError("Shape {} does doesn't match length {}"
                             .format(self._shape, length))
        try:
            self._strides = tuple(prod(self._shape[i+1:])
                                 for i in range(len(self._shape)))
        except TypeError as te:
            print(self._shape)
            raise te
        cls = self.__class__
        bases = tuple(c for c in cls.__bases__
                      if c not in (cls, NDimItems)
                      and hasattr(c, "__getitem__"))
        #print(bases)
        self._base = bases[-1]

    def __expandkeys__(self, keys):
        #print("keys", type(keys), keys)
        if type(keys) is not tuple:
            keys = (keys,)
        shape = self._shape
        ndims, nkeys = len(shape), len(keys)
        if nkeys > ndims:
            raise IndexError(("too many indices for {} is {}-dimensional, "
                              "but {} were indexed")
                             .format(type(self).__name__, ndims, nkeys))
        keys = tuple(keys) + (Ellipsis,) * (ndims - nkeys)
        indexes, mins, maxs = [], [], []
        for ikeys in range(ndims):
            stride = self._strides[ikeys]
            key = self.__key__(keys[ikeys])
            #print(stride, key)
            if isinstance(key, int):
                mins.append(key)
                maxs.append(key + 1)
                indexes.append((key * stride,))
            elif isinstance(key, slice):
                args = tuple(range(*key.indices(shape[ikeys])))
                #print("args", args)
                mins.append(min(args))
                maxs.append(max(args) + 1)
                indexes.append(tuple(i * stride
                                for i in args))
            elif isinstance(key, Iterable):
                args = tuple(key)
                mins.append(min(args))
                maxs.append(max(args) + 1)
                indexes.append(i * stride
                               for i in args)
            else:
                raise IndexError(("invalid at indice {}, expected Iterable "
                                  "or int, but found {}")
                                 .format(ikeys, key))
        reshape = tuple(b - a for (a, b) in zip(mins, maxs))
        indexes = tuple(sum(i) for i in iterprod(*indexes))
        #print("indexes", indexes)
        return indexes, reshape

    def __key__(self, key):
        if key is Ellipsis:
            key = slice(None)
        if isinstance(key, slice):
            return key
        if isinstance(key, (int, NoneType)):
            return self.__index__(key)
        if isinstance(key, Iterable):
            return key
        raise IndexError("expects int, None, Ellipses, slice or "
                         "iterable, but found {}"
                         .format(type(key).__name__))

    def __index__(self, key):
        length = self.__len__()
        if key is None:
            return length
        index = int(key)
        if type(key) is int:
            if index < 0:
                index += length
            if 0 <= index and index < length:
                return index
            raise IndexError("index {} out of range"
                              .format(index))
        raise IndexError("expects None or int, but found {}"
                          .format(type(key).__name__))

    def __getitem__(self, keys):
        indexes, reshape = self.__expandkeys__(keys)
        base = self._base
        return type(self)((base.__getitem__(self, i) for i in indexes),
                          shape=reshape)

    def __delitem__(self, keys):
        indexes, reshape = self.__expandkeys__(keys)
        base = self._base
        return type(self)((base.__detitem__(self, i)
                           for i in sorted(indexes, reversed=True)),
                          shape=reshape)

    def __setitem__(self, keys, values):
        indexes, reshape = self.__expandkeys__(keys)
        base = self._base
        for (i, v) in operandzip(indexes, values):
            base.__setitem__(self, i, v)

    def __repr__(self):
        return "{}([{}], shape={})".format(type(self).__name__,
                                 ', '.join(repr(i) for i in self),
                                 self.shape)


UInt8Map = type("UInt8Map", (SeqOps, Constants, NDimItems, TypedArray),
                 dict(_TypeCode="B", _ItemType=int))

UInt16Map = type("UInt16Map", (SeqOps, Constants, NDimItems, TypedArray),
                  dict(_TypeCode="I", _ItemType=int))

UInt32Map = type("UInt32Map", (SeqOps, Constants, NDimItems, TypedArray),
                  dict(_TypeCode="L", _ItemType=int))

UInt64Map = type("UInt64Map", (SeqOps, Constants, NDimItems, TypedArray),
                  dict(_TypeCode="Q", _ItemType=int))

Int8Map = type("Int8Map", (SeqOps, Constants, NDimItems, TypedArray),
                 dict(_TypeCode="b", _ItemType=int))

Int16Map = type("Int16Map", (Ops, Constants, NDimItems, TypedArray),
                  dict(_TypeCode="i", _ItemType=int))

Int32Map = type("Int32Map", (Ops, Constants, NDimItems, TypedArray),
                  dict(_TypeCode="l", _ItemType=int))

Int64Map = type("Int64Map", (Ops, NDimItems, TypedArray),
                  dict(_TypeCode="q", _ItemType=int))

Float32Map = type("Float32Map", (Ops, Constants, NDimItems, TypedArray),
                  dict(_TypeCode="f", _ItemType=float))

Float64Map = type("Float64Map", (Ops, Constants, NDimItems, TypedArray),
                  dict(_TypeCode="d", _ItemType=float))



"""
from PIL import Image
import pixelmaps as pm
sbm = pm.UInt8Map.fromimagefile('thumbnail.png')
tshape = (sbm.shape[0] * 2 + 1, sbm.shape[1] * 2 + 1, sbm.shape[2])
tbm = pm.UInt8Map(shape=tshape)
tbm[:] = (0, 255, 255)
tbm[1::2,1::2] = sbm
tim = Image.frombytes("RGB", (tbm.shape[1], tbm.shape[0]), bytes(tbm))
tim.save('thumbnail-dispersed.png')
"""
