import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from funcbuilder import FuncBuilder

B, [a, b] = FuncBuilder("a", "b")

re = B.phi(0.0)
im = B.phi(0.0)
n = B.phi(50)

c = B.complex(a, b)
x = B.complex(re, im)

B.set_label("loop")

norm = B.cnorm2(x)
c1 = B.gt(norm, 4.0)
B.cbranch(c1, "done")

x2 = B.cmul(x, x)
r1 = B.cadd(x2, c)

re.add_incoming(r1.re)
im.add_incoming(r1.im)

r2 = B.fsub(n, 1)
n.add_incoming(r2)
r3 = B.gt(r2, 0)

B.cbranch(r3, "loop")

B.set_label("done")

r4 = B.fsub(50, n)
f = B.compile(r4)

# print(f.compiler.dumps())

A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))
H = np.zeros_like(A)

F = np.vectorize(f)
H = F(A, B)

plt.imshow(H)
plt.show()
