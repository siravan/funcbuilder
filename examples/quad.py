import sys
import ctypes
from math import pi
from scipy.integrate import quad

from funcbuilder import FuncBuilder

B, [x] = FuncBuilder('x')

# Ahmed's Integral (Inside Interesting Integrals, 6.2)
# f(x) = atan(sqrt(2 + x**2)) / ((1 + x**2) * sqrt(2 + x**2))

r1 = B.square(x)
r2 = B.fadd(r1, 2.0)
r3 = B.sqrt(r2)
r4 = B.atan(r3)
r5 = B.fadd(r1, 1.0)
r6 = B.fmul(r5, r3)
r7 = B.fdiv(r4, r6)

f = B.compile(r7, sig=[ctypes.c_double, ctypes.c_int, ctypes.c_double])

sol = quad(f, 0.0, 1.0)

print('quad =\t\t', sol[0])
print('analytic = \t', 5 * pi**2 / 96)
