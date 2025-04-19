import numbers

from . import pyengine

tree = pyengine.tree

class Block:
    def __init__(self, label):
        self.label = label
        self.eqs = []   # equations
        self.hits = {}  # number of times each observable is used in this block
        self.closure = None
        
    def list_merges(self, externals):
        merges = {}         
        for eq in self.eqs:
            v = eq.lhs
            if self.hits.get(v) == 1 and v not in externals:
                merges[v] = eq.rhs                
        return merges
        
    def arborize(self, externals):
        """Merges expression trees if possible"""
        merges = self.list_merges(externals)
        eqs = []
        
        for eq in self.eqs:
            lhs, rhs = eq.lhs, eq.rhs
            
            if isinstance(rhs, tree.Unary):
                p = tree.Unary(
                    rhs.op, 
                    merges.get(rhs.arg, rhs.arg)
                )                
            elif isinstance(rhs, tree.Binary):
                p = tree.Binary(
                    rhs.op,
                    merges.get(rhs.left, rhs.left),
                    merges.get(rhs.right, rhs.right)                    
                )
            elif isinstance(eq.rhs, tree.Select):
                p = tree.Select(
                    merges.get(rhs.cond, rhs.cond),
                    merges.get(rhs.true_val, rhs.true_val),
                    merges.get(rhs.false_val, rhs.false_val),                    
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
                tree.Branch(
                    merges.get(p.cond, p.cond), 
                    p.true_label, 
                    p.false_label
                )   
            )
                        
        return eqs       


class Phi:
    def __init__(self, parent):
        self.parent = parent
        self.var = None
        
    def add_incoming(self, a):        
        a = self.parent.prep(a)
        
        if self.var is None:
            self.var = self.parent.new_var(a)
            self.parent.externals.add(self.var)
        else:        
            self.parent.block.eqs.append(tree.Eq(self.var, a))
        

class Builder:
    def __init__(self, *states):
        self.states = [tree.Var(x) for x in states]
        self.obs = []  # observables (intermediate variables)
        self.blocks = [Block("@0")]
        self.block = self.blocks[0]        
        self.externals = set()

    def new_var(self, rhs):
        """Creates a new observable"""
        k = len(self.obs)
        v = tree.Var(f"${k}")
        self.block.hits[v] = 0        
        self.block.eqs.append(tree.Eq(v, rhs))
        self.obs.append(v)
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
            if a.var is None:
                raise ValueError("cannot use an uninitiated Phi. Every Phi should have at least one incoming link.")
            a = a.var            
            
        if a in self.block.hits:
            self.block.hits[a] += 1
        else:
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

    def append_select(self, cond, a, b):
        cond = self.prep(cond)
        a = self.prep(a)
        b = self.prep(b)
        rhs = tree.Select(cond, a, b) 
        return self.new_var(rhs)        
        
    def phi(self):
        return Phi(self)

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
            return self.call("power", a, b)
            
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
        return self.call("arcsin", a)

    def acos(self, a):
        return self.call("arccos", a)

    def atan(self, a):
        return self.call("arctan", a)

    def asinh(self, a):
        return self.call("arcsinh", a)

    def acosh(self, a):
        return self.call("arccosh", a)

    def atanh(self, a):
        return self.call("arctanh", a)
        
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
        return self.append_select(cond, a, b)    
                
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
        self.externals.add(cond)
        self.block.closure = tree.Branch(cond, true_label, false_label)
        self.add_block()

    def compile(self, y):
        try:            
            if isinstance(y, Phi):
                y = y.var
                
            eqs = self.coalesce(y)

            model = tree.Model(                
                self.states,  # states
                self.obs,  # obs
                eqs,  # eqs
            )

            compiler = pyengine.PyCompiler(model, y, ty="native")
            func = compiler.func
            # to prevent compiler to deallocate before func, 
            # since compiler holds the actual code, func is just a wrapper
            func.compiler = compiler    
            return func
        except:
            raise ValueError(f"return variable {y} not found")

    def coalesce(self, y):
        eqs = []
        externals = self.externals | {y}
        
        for b in self.blocks:
            eqs.append(tree.Label(b.label))            
            eqs.extend(b.arborize(externals))            
                
        return eqs                




