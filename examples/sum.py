from funcbuilder import FuncBuilder

N = 12

B, X = FuncBuilder(*[f'x{i}' for i in range(N)])

s = B.phi(0.0)

for i in range(N):
    r = B.fadd(s, X[i])
    s.add_incoming(r)

f = B.compile(s)

# print(f.compiler.dumps())

print(f(*[i for i in range(N)]))
