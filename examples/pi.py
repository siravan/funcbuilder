import math

from funcbuilder import FuncBuilder

def arctan_series(B, x0):
    s = B.init(x0)
    x = B.init(x0)
    x2 = B.square(x)
    
    for i in range(1, 23):
        r1 = B.fmul(x, x2)
        B.assign(x, r1)
        coef = -(1+2*i) if (i & 1 == 1) else 1+2*i
        r2 = B.fdiv(x, coef)
        r3 = B.fadd(s, r2)
        B.assign(s, r3)
        
    return s        
        

B, [] = FuncBuilder()

s1 = arctan_series(B, 1/2)
s2 = arctan_series(B, 1/3)

s = B.fadd(s1, s2)
p = B.fmul(s, 4.0)

f = B.compile(p)

print('func = \t', f())
print('math = \t', math.pi)

