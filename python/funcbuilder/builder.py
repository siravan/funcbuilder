import numbers

from . import pyengine

tree = pyengine.tree

class Builder:
    def __init__(self, *states):
        self.states = [tree.Var(x) for x in states]
        self.obs = []  # observables (intermediate variables)
        self.eqs = []  # equations
        self.hits = {}  # number of times each observable is used

    def new_var(self):
        """Creates a new observable"""
        k = len(self.obs)
        v = tree.Var(f"${k}")
        self.obs.append(v)
        self.hits[v] = 0
        return v

    def prep(self, a):
        """Called on each observable on the right hand side of an equation"""
        if isinstance(a, numbers.Number):
            return tree.Const(a)
        if a in self.obs:
            self.hits[a] += 1
        return a

    def append_unary(self, op, a):
        a = self.prep(a)
        v = self.new_var()
        self.eqs.append(tree.Eq(v, tree.Unary(op, a)))
        return v

    def append_binary(self, op, a, b):
        a = self.prep(a)
        b = self.prep(b)
        v = self.new_var()
        self.eqs.append(tree.Eq(v, tree.Binary(op, a, b)))
        return v

    def append_ifelse(self, cond, a, b):
        cond = self.prep(cond)
        a = self.prep(a)
        b = self.prep(b)
        v = self.new_var()
        self.eqs.append(tree.Eq(v, tree.IfElse(cond, a, b)))
        return v

    def add(self, *a):
        if len(a) == 1:
            return self.prep(a[0])
        elif len(a) > 2:
            t = self.add(*a[1:])
            return self.append_binary("plus", a[0], t)
        else:
            return self.append_binary("plus", a[0], a[1])

    def sub(self, a, b):
        return self.append_binary("minus", a, b)

    def mul(self, *a):
        if len(a) == 1:
            return self.prep(a[0])
        elif len(a) > 2:
            t = self.add(*a[1:])
            return self.append_binary("times", a[0], t)
        else:
            return self.append_binary("times", a[0], a[1])

    def div(self, a, b):
        return self.append_binary("divide", a, b)

    def pow(self, a, power):
        if power == 2:
            return self.append_unary("square", a)
        elif power == 3:
            return self.append_unary("cube", a)
        elif power == -1:
            return self.append_unary("recip", a)
        elif power == -2:
            t = self.append_unary("recip", a)
            return self.append_unary("square", t)
        elif power == -3:
            t = self.append_unary("recip", a)
            return self.append_unary("cube", t)
        elif power == 0.5:
            return self.append_unary("root", a)
        elif power == -0.5:
            t = self.append_unary("root", a)
            return self.append_unary("recip", t)
        elif power == 1.5:
            t = self.append_unary("root", a)
            return self.append_unary("cube", t)
        else:
            return self.append_binary("power", a, b)

    def exp(self, a):
        return self.append_unary("exp", a)

    def log(self, a):
        return self.append_unary("log", a)

    def sqrt(self, a):
        return self.append_unary("root", a)

    def square(self, a):
        return self.append_unary("square", a)

    def cube(self, a):
        return self.append_unary("cube", a)

    def recip(self, a):
        return self.append_unary("recip", a)

    def sin(self, a):
        return self.append_unary("sin", a)

    def cos(self, a):
        return self.append_unary("cos", a)

    def tan(self, a):
        return self.append_unary("tan", a)

    def sinh(self, a):
        return self.append_unary("sinh", a)

    def cosh(self, a):
        return self.append_unary("cosh", a)

    def tanh(self, a):
        return self.append_unary("tanh", a)

    def asin(self, a):
        return self.append_unary("arcsin", a)

    def acos(self, a):
        return self.append_unary("arccos", a)

    def atan(self, a):
        return self.append_unary("arctan", a)

    def asinh(self, a):
        return self.append_unary("arcsinh", a)

    def acosh(self, a):
        return self.append_unary("arccosh", a)

    def atanh(self, a):
        return self.append_unary("arctanh", a)

    def lt(self, a, b):
        return self.append_binary("lt", a, b)

    def leq(self, a, b):
        return self.append_binary("leq", a, b)

    def gt(self, a, b):
        return self.append_binary("gt", a, b)

    def geq(self, a, b):
        return self.append_binary("geq", a, b)

    def eq(self, a, b):
        return self.append_binary("eq", a, b)

    def neq(self, a, b):
        return self.append_binary("neq", a, b)

    def logical_and(self, a, b):
        return self.append_binary("and", a, b)

    def logical_or(self, a, b):
        return self.append_binary("or", a, b)

    def logical_xor(self, a, b):
        return self.append_binary("xor", a, b)

    def ifelse(self, cond, a, b):
        return self.append_ifelse(cond, a, b)

    def compile(self, y):
        try:
            eqs = self.arborize(y)

            model = tree.Model(                
                self.states,  # states
                self.obs,  # obs
                eqs,  # eqs
            )

            idx = self.obs.index(y)
            compiler = pyengine.PyCompiler(model, y, ty="native")
            func = compiler.func
            # to prevent compiler to deallocate before func, 
            # since compiler holds the actual code, func is just a wrapper
            func.compiler = compiler    
            return func
        except:
            raise ValueError(f"return variable {y} not found")

    def merge(self, eqs, y, v):
        """Moves an observable v and merges it into the
        destination tree if it is used only once.
        y is the output observable and cannot be moved.
        """
        try:
            if v != y and self.hits[v] == 1:
                idx = self.obs.index(v)
                u = eqs[idx].rhs
                eqs[idx] = None
                return u
        except:
            return v
        return v

    def arborize(self, y):
        """Merges expression trees if possible"""
        eqs = []

        for eq in self.eqs:
            if isinstance(eq.rhs, tree.Unary):
                eqs.append(
                    tree.Eq(
                        eq.lhs, tree.Unary(eq.rhs.op, self.merge(eqs, y, eq.rhs.arg))
                    )
                )
            elif isinstance(eq.rhs, tree.Binary):
                eqs.append(
                    tree.Eq(
                        eq.lhs,
                        tree.Binary(
                            eq.rhs.op,
                            self.merge(eqs, y, eq.rhs.left),
                            self.merge(eqs, y, eq.rhs.right),
                        ),
                    )
                )
            elif isinstance(eq.rhs, tree.IfElse):
                eqs.append(
                    tree.Eq(
                        eq.lhs,
                        tree.IfElse(
                            self.merge(eqs, y, eq.rhs.cond),
                            self.merge(eqs, y, eq.rhs.true_val),
                            self.merge(eqs, y, eq.rhs.false_val),
                        ),
                    )
                )
            else:
                eqs.append(tree.Eq(eq.lhs, eq.rhs))
        # the equations of the observables that are merged
        # is set to None in the merge function and
        # are removed here
        return [eq for eq in eqs if eq is not None]



