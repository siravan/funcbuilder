from funcbuilder import FuncBuilder

B, [x] = FuncBuilder('x')

r1 = B.init(1.0)

B.set_label('loop')
r2 = B.fmul(r1, x)
B.assign(r1, r2)
r3 = B.fsub(x, 1.0)
B.assign(x, r3)
r4 = B.geq(x, 1.0)
B.cbranch(r4, 'loop')

f = B.compile(r1)

print(f(10))

