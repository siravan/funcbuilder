from sympy import *
from funcbuilder import compile_func

x = symbols("x")

x0 = 0.0001

print(f"depth\trbuilder\t\tlambdify")

for i in range(12):
    e = x**2 + x

    for _ in range(i):
        e = e**2 + e

    ed = e.diff(x)

    fb = compile_func([x], ed)
    fl = lambdify([x], ed)

    print(f"{i}\t{fb(x0):.12f}\t{fl(x0):.12f}")
