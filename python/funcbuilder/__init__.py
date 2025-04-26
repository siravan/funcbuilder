from . import pyengine
from . import builder
from . import lowering


def FuncBuilder(*states):
    if pyengine.can_compile():
        B = builder.Builder(*states)
        return B, B.states
    else:
        raise ValueError("unsupported platform")


def compile_func(states, eqs):
    states = [str(s) for s in states]

    if not isinstance(eqs, list):
        B, _ = FuncBuilder(*states)
        lowering.lower(B, eqs)
        return B.compile()

    fs = []
    for eq in eqs:
        B, _ = FuncBuilder(*states)
        lowering.lower(B, eq)
        fs.append(B.compile())

    return lambda *xs: [f(*xs) for f in fs]
