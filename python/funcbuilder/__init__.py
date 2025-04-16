from . import pyengine
from . import builder

def FuncBuilder(*states):
    if pyengine.can_compile():
        B = builder.Builder(*states)
        return B, B.states
    else:
        raise ValueError("unsupported platform")
