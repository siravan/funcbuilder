import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from funcbuilder import FuncBuilder

B, [a, b] = FuncBuilder('a', 'b')

x = B.phi()
y = B.phi()
n = B.phi()

x.add_incoming(0.0)
y.add_incoming(0.0)
n.add_incoming(50)

B.set_label('loop')

x2 = B.square(x)
y2 = B.square(y)

c1 = B.gt(x2, 4.0)
B.cbranch(c1, 'done')

c2 = B.gt(y2, 4.0)
B.cbranch(c2, 'done')

r1 = B.fsub(x2, y2)
r2 = B.fadd(r1, a)

xy = B.fmul(x, y, 2.0)
r3 = B.fadd(xy, b)

x.add_incoming(r2)
y.add_incoming(r3)

r4 = B.fsub(n, 1)
n.add_incoming(r4)
r5 = B.gt(r4, 0)

B.cbranch(r5, 'loop')

B.set_label('done')

r6 = B.fsub(50, n)
f = B.compile(r6)

A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))
H = np.zeros_like(A)

F = np.vectorize(f)
H = F(A, B)

plt.imshow(H)
plt.show()
