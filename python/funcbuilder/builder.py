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
        
    def invalidate_hits(self):
        for v in self.hits:
            self.hits[v] = 2            

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

    def append_select(self, cond, a, b):
        cond = self.prep(cond)
        a = self.prep(a)
        b = self.prep(b)
        v = self.new_var()
        self.eqs.append(tree.Eq(v, tree.Select(cond, a, b)))
        return v
        
    def init(self, a):
        a = self.prep(a)
        v = self.new_var()
        self.eqs.append(tree.Eq(v, a))
        return v
        
    def assign(self, a, b):
        b = self.prep(b)
        self.eqs.append(tree.Eq(a, b))
        return a

    def fadd(self, *a):
        if len(a) == 1:
            return self.prep(a[0])
        elif len(a) > 2:
            t = self.add(*a[1:])
            return self.append_binary("plus", a[0], t)
        else:
            return self.append_binary("plus", a[0], a[1])

    def fsub(self, a, b):
        return self.append_binary("minus", a, b)

    def fmul(self, *a):
        if len(a) == 1:
            return self.prep(a[0])
        elif len(a) > 2:
            t = self.add(*a[1:])
            return self.append_binary("times", a[0], t)
        else:
            return self.append_binary("times", a[0], a[1])

    def fdiv(self, a, b):
        return self.append_binary("divide", a, b)
        
    def fneg(self, a):
        return self.append_unary("neg", a)            

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
        
    def call(fn, args):
        if fn == "pow":
            return self.exp(args[0], args[1])
        elif fn == "exp":
            return self.exp(args[0])     
        elif fn == "log":
            return self.log(args[0])
        elif fn == "root":
            return self.root(args[0])
        elif fn == "square":
            return self.square(args[0])
        elif fn == "cube":
            return self.cube(args[0])
        elif fn == "recip":
            return self.recip(args[0])
        elif fn == "sin":
            return self.sin(args[0])    
        elif fn == "cos":
            return self.cos(args[0])
        elif fn == "tan":
            return self.tan(args[0])
        elif fn == "sinh":
            return self.sinh(args[0])    
        elif fn == "cosh":
            return self.cosh(args[0])
        elif fn == "tanh":
            return self.tanh(args[0])
        elif fn == "asin":
            return self.asin(args[0])    
        elif fn == "acos":
            return self.acos(args[0])
        elif fn == "atan":
            return self.atan(args[0])
        elif fn == "asinh":
            return self.asinh(args[0])    
        elif fn == "acosh":
            return self.acosh(args[0])
        elif fn == "atanh":
            return self.atanh(args[0])            
        else:
            raise ValueError(f"undefined function name {fn}")

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
        
    def fcmp_ordered(op, a, b):
        if op == "<":
            return self.lt(a, b)
        elif op == "<=":
            return self.leq(a, b)
        elif op == ">":
            return self.gt(a, b)
        elif op == ">=":
            return self.geq(a, b)
        elif op == "==":
            return self.eq(a, b)
        elif op == "!=":
            return self.neq(a, b)
        else:
            raise ValueError(f"undefined comparison {op}")

    def and_(self, a, b):
        return self.append_binary("and", a, b)

    def or_(self, a, b):
        return self.append_binary("or", a, b)

    def xor(self, a, b):
        return self.append_binary("xor", a, b)
        
    def not_(self, a, b):
        return self.append_binary("not", a, b)

    def select(self, cond, a, b):
        return self.append_select(cond, a, b)    
        
    def set_label(self, label):
        self.eqs.append(tree.Label(label))
        
    def branch(self, label):
        self.eqs.append(tree.Branch(True, label))
        
    def cbranch(self, cond, true_label, false_label=None):
        cond = self.prep(cond)
        self.eqs.append(tree.Branch(cond, true_label, false_label))

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
            if isinstance(eq, tree.Label):
                eqs.append(eq)
            elif isinstance(eq, tree.Branch):
                eqs.append(
                    tree.Branch(
                        self.merge(eqs, y, eq.cond),
                        eq.true_label,
                        eq.false_label
                    )
                )
            elif isinstance(eq.rhs, tree.Unary):
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
            elif isinstance(eq.rhs, tree.Select):
                eqs.append(
                    tree.Eq(
                        eq.lhs,
                        tree.Select(
                            self.merge(eqs, y, eq.rhs.cond),
                            self.merge(eqs, y, eq.rhs.true_val),
                            self.merge(eqs, y, eq.rhs.false_val),
                        ),
                    )
                )
            else:
                eqs.append(eq)
        # the equations of the observables that are merged
        # is set to None in the merge function and
        # are removed here
        return [eq for eq in eqs if eq is not None]



