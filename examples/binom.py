import math
from funcbuilder import FuncBuilder

B, [] = FuncBuilder()


def binom(B, n, k):
    if k == 0 or k == n:
        return 1.0
    else:
        r1 = binom(B, n - 1, k)
        r2 = binom(B, n - 1, k - 1)
        r3 = B.fadd(r1, r2)
        return r3


n = 15
k = 8

r = binom(B, n, k)
f = B.compile(r)

# print(f.compiler.dumps())

print(f"f({n}, {k}) =\t", f())
print(f"binom({n}, {k}) =\t", math.comb(n, k))
