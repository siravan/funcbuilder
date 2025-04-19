from funcbuilder import FuncBuilder

B, [x] = FuncBuilder('x')

p = B.phi()
p.add_incoming(1.0)

n = B.phi()
n.add_incoming(x)

B.set_label('loop')
r1 = B.fmul(p, n)
p.add_incoming(r1)
r2 = B.fsub(n, 1.0)
n.add_incoming(r2)
r3 = B.geq(n, 1.0)
B.cbranch(r3, 'loop')

f = B.compile(p)

print(f(10))

