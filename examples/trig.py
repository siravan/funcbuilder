from funcbuilder import FuncBuilder

B, [x] = FuncBuilder('x')

r1 = B.sin(x)
r2 = B.square(r1)
r3 = B.cos(x)
r4 = B.square(r3)
r5 = B.fadd(r2, r4)

f = B.compile(r5)

print(f(0.12345))
