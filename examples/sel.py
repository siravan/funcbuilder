from random import random
from funcbuilder import FuncBuilder

B, [x, y] = FuncBuilder("x", "y")

r1 = B.gt(x, y)
r2 = B.select(r1, 1.0, 0.0)

f = B.compile(r2)

print(f.compiler.dumps())

for _ in range(10):
    X = random()
    Y = random()
    print(f"is {X:.3f} >= {Y:.3f}?\t", f(X, Y))
