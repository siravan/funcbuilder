from funcbuilder import FuncBuilder

B, [x, y] = FuncBuilder('x', 'y')

r1 = B.fadd(x, y)
r2 = B.fmul(x, y)
r3 = B.fdiv(r1, r2)
r4 = B.pow(r3, 4)
r5 = B.fadd(r4, 1.0)

f = B.compile(r5)

print(f.compiler.dumps())

print(f(2.0, 3.0))
