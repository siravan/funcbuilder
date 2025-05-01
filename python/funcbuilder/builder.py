import numbers

from . import pyengine

tree = pyengine.tree


class Block:
    def __init__(self, label):
        self.label = label
        self.eqs = []  # equations
        self.hits = {}  # number of times each observable is used in this block
        self.closure = None

    def list_merges(self, obs):
        """
            merge logic:
            
                1. if a variable is created in a block and is used only once,
                    it can be merged.
                2. if a variable is created in a block and is used more than once,
                    it needs stack storage.
                3. if a variable is created in a block and used in another block,
                    it needs stack storage.
                    
            obs is a set of variables that need stack stotage. It has three sourcs:                
            
                1. variables created in a block and used outside (from Builder.externals)
                2. variables created in a block and used more than once (in this function)
                3. the argument to compile function
        """
        merges = {}
        for eq in self.eqs:
            v = eq.lhs
            if v not in obs:
                if self.hits[v] == 1: 
                    merges[v] = eq.rhs
                elif self.hits[v] > 1:
                    obs.add(v)                    
        return merges

    def arborize(self, obs):
        """Merges expression trees if possible"""
        merges = self.list_merges(obs)
        eqs = []

        for eq in self.eqs:
            lhs, rhs = eq.lhs, eq.rhs

            if isinstance(rhs, tree.Unary):
                p = tree.Unary(rhs.op, merges.get(rhs.arg, rhs.arg))
            elif isinstance(rhs, tree.Binary):
                p = tree.Binary(
                    rhs.op,
                    merges.get(rhs.left, rhs.left),
                    merges.get(rhs.right, rhs.right),
                )
            elif isinstance(eq.rhs, tree.Var):
                p = merges.get(rhs, rhs)
            else:
                p = rhs

            if lhs in merges:
                merges[lhs] = p
            else:
                eqs.append(tree.Eq(lhs, p))

        if self.closure is not None:
            p = self.closure
            eqs.append(
                tree.Branch(merges.get(p.cond, p.cond), p.true_label, p.false_label)
            )

        return eqs
        
    def add_eq(self, lhs, rhs):
        self.hits[lhs] = 0
        self.eqs.append(tree.Eq(lhs, rhs))
        
    def hit(self, v):
        if v in self.hits:
            self.hits[v] += 1
            return True
        else:
            return False


class Phi:
    def __init__(self, parent, var):
        self.parent = parent
        self.var = var
        self.name = var.name

    def add_incoming(self, a):
        self.parent.add_block()  # needed based on the mandelbrot example
        a = self.parent.prep(a)
        self.parent.block.eqs.append(tree.Eq(self.var, a))
        

class Complex:
    def __init__(self, a, b=0.0):
        if isinstance(a, Complex):
            self.re = a.re
            self.im = a.im
        else:
            self.re = a
            self.im = b


class Builder:
    def __init__(self, *states):
        self.states = [tree.Var(x) for x in states]
        self.blocks = [Block("@0")]
        self.block = self.blocks[0]
        self.externals = set()
        self.count_vars = 0

    def new_var(self, rhs):
        """Creates a new observable"""
        v = tree.Var(f"${self.count_vars}")
        self.count_vars += 1
        self.block.add_eq(v, rhs)
        return v

    def add_block(self, label=None):
        if label is None:
            label = f"@{len(self.blocks)}"
        self.blocks.append(Block(label))
        self.block = self.blocks[-1]

    def prep(self, a):
        """Called on each observable on the right hand side of an equation"""
        if isinstance(a, numbers.Number):
            return tree.Const(a)

        if isinstance(a, Phi):
            a = a.var

        # if `a` is not defined in the current block, then it is an external
        if not self.block.hit(a):
            self.externals.add(a)

        return a

    def append_unary(self, op, a):
        a = self.prep(a)
        rhs = tree.Unary(op, a)
        return self.new_var(rhs)

    def append_binary(self, op, a, b):
        a = self.prep(a)
        b = self.prep(b)
        rhs = tree.Binary(op, a, b)
        return self.new_var(rhs)

    def phi(self, a=None):
        if a is None:
            a = tree.Const(0.0)
        else:
            a = self.prep(a)
        return Phi(self, self.new_var(a))           

    def fadd(self, *a):
        if len(a) == 1:
            return self.prep(a[0])
        elif len(a) > 2:
            t = self.fadd(*a[1:])
            return self.append_binary("plus", a[0], t)
        else:
            return self.append_binary("plus", a[0], a[1])

    def fsub(self, a, b):
        return self.append_binary("minus", a, b)

    def fmul(self, *a):
        if len(a) == 1:
            return self.prep(a[0])
        elif len(a) > 2:
            t = self.fmul(*a[1:])
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
            return self.call("pow", a, power)

    def sqrt(self, a):
        return self.append_unary("root", a)

    def square(self, a):
        return self.append_unary("square", a)

    def cube(self, a):
        return self.append_unary("cube", a)

    def recip(self, a):
        return self.append_unary("recip", a)

    def exp(self, a):
        return self.call("exp", a)

    def log(self, a):
        return self.call("log", a)

    def sin(self, a):
        return self.call("sin", a)

    def cos(self, a):
        return self.call("cos", a)

    def tan(self, a):
        return self.call("tan", a)

    def sinh(self, a):
        return self.call("sinh", a)

    def cosh(self, a):
        return self.call("cosh", a)

    def tanh(self, a):
        return self.call("tanh", a)

    def asin(self, a):
        return self.call("asin", a)

    def acos(self, a):
        return self.call("acos", a)

    def atan(self, a):
        return self.call("atan", a)

    def asinh(self, a):
        return self.call("asinh", a)

    def acosh(self, a):
        return self.call("acosh", a)

    def atanh(self, a):
        return self.call("atanh", a)

    def call(self, fn, *args):
        self.add_block()
        args = [self.prep(arg) for arg in args]
        rhs = tree.Call(fn, args)
        v = self.new_var(rhs)
        self.add_block()
        return v

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
        a = self.append_binary("select_if", cond, a)
        b = self.append_binary("select_else", cond, b)
        return self.or_(a, b)

    def set_label(self, label):
        if len(self.block.eqs) == 0:
            self.block.label = label
        else:
            self.add_block(label)

    def branch(self, label):
        self.block.closure = tree.Branch(True, label)
        self.add_block()

    def cbranch(self, cond, true_label, false_label=None):
        cond = self.prep(cond)
        self.block.closure = tree.Branch(cond, true_label, false_label)
        self.add_block()

    def complex(self, a, b):
        return Complex(a, b)

    def cadd(self, a, b):
        r1 = self.fadd(a.re, b.re)
        r2 = self.fadd(a.im, b.im)

        return Complex(r1, r2)

    def csub(self, a, b):
        r1 = self.fsub(a.re, b.re)
        r2 = self.fsub(a.im, b.im)

        return Complex(r1, r2)

    def cmul(self, a, b):
        r1 = self.fmul(a.re, b.re)
        r2 = self.fmul(a.im, b.im)
        r3 = self.fsub(r1, r2)

        r4 = self.fmul(a.re, b.im)
        r5 = self.fmul(a.im, b.re)
        r6 = self.fadd(r4, r5)

        return Complex(r3, r6)

    def cdiv(self, a, b):
        r1 = self.square(b.re)
        r2 = self.square(b.im)
        r3 = self.fadd(r1, r2)

        r4 = self.fmul(a.re, b.re)
        r5 = self.fmul(a.im, b.im)
        r6 = self.fadd(r1, r2)
        r7 = self.dfiv(r6, r3)

        r8 = self.fmul(a.im, b.re)
        r9 = self.fmul(a.re, b.im)
        r10 = self.fsub(r8, r9)
        r11 = self.dfiv(r10, r3)

        return Complex(r7, r11)
        
    def cnorm2(self, a):
        r1 = self.square(a.re)
        r2 = self.square(a.im)
        r3 = self.fadd(r1, r2)
        return r3

    def compile(self, y, sig=None):
        try:
            if isinstance(y, Phi):
                y = y.var

            eqs, obs = self.coalesce(y)

            model = tree.Model(
                self.states,  # states
                obs,  # obs
                eqs,  # eqs
            )

            compiler = pyengine.PyCompiler(model, y, ty="native", sig=sig)
            func = compiler.func
            # to prevent compiler to deallocate before func,
            # since compiler holds the actual code, func is just a wrapper
            func.compiler = compiler
            return func
        except:
            raise ValueError(f"return variable {y} not found")

    def coalesce(self, y):
        eqs = []
        obs = self.externals | {y}

        for b in self.blocks:
            eqs.append(tree.Label(b.label))
            eqs.extend(b.arborize(obs))

        return eqs, obs
