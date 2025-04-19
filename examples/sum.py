from funcbuilder import FuncBuilder

B, X = FuncBuilder('x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9')

s = B.init(0.0)

for i in range(10):
    r = B.fadd(s, X[i])
    B.assign(s, r)
    
f = B.compile(s)

print(f(*[i for i in range(10)]))    


