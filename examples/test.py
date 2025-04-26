from funcbuilder import FuncBuilder

B, [x, y] = FuncBuilder("x", "y")
a = B.fadd(x, y)
f = B.compile(a)

print(f(1.0, 2.0))
