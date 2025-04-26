import numbers
from sympy import *

from . import pyengine

tree = pyengine.tree

from . import builder


def lower(B, y):
    if y.is_Number or isinstance(y, numbers.Number):
        return y
    elif y.is_Symbol:
        return tree.Var(str(y))
    elif y.is_Relational:
        return relational(B, y)
    elif y.is_Boolean:
        return boolean(B, y)
    elif y.is_Piecewise:
        return piecewise(B, y)
    else:
        return lower_tree(B, y)


def lower_tree(B, y):
    if y.is_Add:
        return lower_add(B, y)
    elif y.is_Mul:
        return lower_mul(B, y)
    elif y.is_Pow:
        return lower_pow(B, y)
    elif y.is_Function:
        if len(y.args) == 1:
            return B.call(str(y.func), lower(B, y.args[0]))
        elif len(y.args) == 2:
            return B.call(str(y.func), lower(B, y.args[0]), lower(B, y.args[1]))
    raise ValueError(f"unrecognized expression: {y}")


def lower_add(B, y):
    assert y.is_Add
    args = [lower(B, arg) for arg in y.args]
    return B.fadd(*args)


def lower_mul(B, y):
    assert y.is_Mul
    args = [lower(B, arg) for arg in y.args]
    return B.fmul(*args)


def lower_pow(B, y):
    assert y.is_Pow
    return B.pow(lower(B, y.args[0]), lower(B, y.args[1]))


def relational(B, y):
    f = y.func
    a = lower(B, y.args[0])
    b = lower(B, y.args[1])

    if f == LessThan:
        return B.lt(a, b)
    elif f == StrictLessThan:
        return B.leq(a, b)
    elif f == GreaterThan:
        return B.gt(a, b)
    elif f == StrictGreaterThan:
        return B.geq(a, b)
    elif f == Equality:
        return B.eq(a, b)
    elif f == Unequality:
        return B.neq(a, b)
    else:
        raise ValueError("unrecognized relational operator")


def boolean(B, y):
    f = y.func
    a = lower(B, y.args[0])

    if f == Not:
        return B.not_(a)

    b = lower(B, y.args[1])

    if f == And:
        return B.and_(a, b)
    elif f == Or:
        return B.or_(a, b)
    elif f == Xor:
        return B.xor(a, b)
    else:
        raise ValueError("unrecognized boolean operator")


def piecewise(y):
    cond = y.args[0][1]
    x1 = y.args[0][0]

    if len(y.args) == 1:
        return lower(B, x1)
    if len(y.args) == 2:
        x2 = y.args[1][0]
    else:
        x2 = piecewise(*y.args[1:])

    return B.select(lower(B, cond), lower(B, x1), lower(B, x2))
