import sys
import time
import numpy as np
import matplotlib.pyplot as plt



def f(a, b):
    c = a + 1j * b
    x = np.zeros_like(c)
    n = np.zeros(c.shape, dtype=int)
    
    for i in range(50):
        x = x * x + c
        n += np.abs(x) < 2.0
        
    return n        

A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))
H = f(A, B)

plt.imshow(H)
plt.show()
