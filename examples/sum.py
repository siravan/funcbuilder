from funcbuilder import FuncBuilder


B, X = FuncBuilder("x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9")

s = B.phi()
s.add_incoming(0.0)

for i in range(10):
    r = B.fadd(s, X[i])
    s.add_incoming(r)

f = B.compile(s)

# print(f.compiler.dumps())

print(f(*[i for i in range(10)]))
