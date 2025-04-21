import sys
import numbers
from sympy import *

COUNT_TEMP_XMM = 3 if sys.platform == "win32" else 13


class Unary:
    def __init__(self, op, arg):
        self.op = op
        self.arg = arg

    def __repr__(self):
        return f"Unary('{self.op}', {self.arg})"

    def compile(self, dst, prog, mem, vt):
        self.arg.compile(dst, prog, mem, vt)

        if self.op == "neg":
            prog.neg(dst)
        elif self.op == "not":
            prog.not_(dst)                        
        elif self.op == "abs":
            prog.abs(dst)
        elif self.op == "root":
            prog.root(dst)
        elif self.op == "square":
            prog.square(dst)
        elif self.op == "cube":
            prog.cube(dst)
        elif self.op == "recip":
            prog.recip(dst)
        else:
            raise ValueError(f"unary op {self.op} not defined")


class Binary:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Binary('{self.op}', {self.left}, {self.right})"

    def compile(self, dst, prog, mem, vt):
        sp = mem.push()        

        if sp < prog.count_shadows:
            self.right.compile(prog.first_shadow + sp, prog, mem, vt)
        else:
            self.right.compile(dst, prog, mem, vt)
            prog.save_mem(dst, self.idx)

        self.left.compile(dst, prog, mem, vt)

        if sp < prog.count_shadows:
            r = prog.first_shadow + sp
        else:
            prog.load_mem(1, self.idx)
            r = 1

        mem.pop()            
        
        if self.op == "plus":
            prog.plus(dst, r)
        elif self.op == "minus":
            prog.minus(dst, r)
        elif self.op == "times":
            prog.times(dst, r)
        elif self.op == "divide":
            prog.divide(dst, r)
        elif self.op == "power":
            prog.call_binary(dst, r, vt.find("pow"))
        elif self.op == "gt":
            prog.gt(dst, r)
        elif self.op == "geq":
            prog.geq(dst, r)
        elif self.op == "lt":
            prog.lt(dst, r)
        elif self.op == "leq":
            prog.leq(dst, r)
        elif self.op == "eq":
            prog.eq(dst, r)
        elif self.op == "neq":
            prog.neq(dst, r)
        elif self.op == "and":
            prog.and_(dst, r)
        elif self.op == "or":
            prog.or_(dst, r)
        elif self.op == "xor":
            prog.xor(dst, r)        
        elif self.op == "select_if":
            prog.select_if(dst, r)
        elif self.op == "select_else":
            prog.select_else(dst, r)                        
        else:
            raise ValueError(f"binary op {self.op} not defined")


class Call:
    def __init__(self, fn, args):
        self.fn = fn
        self.args = args
        
    def __repr__(self):
        return f"Call({self.fn},...)"
        
    def compile(self, dst, prog, mem, vt):
        args = self.args
        
        if len(args) == 1:
            assert(isinstance(args[0], Var) or isinstance(args[0], Const))
            args[0].compile(0, prog, mem, vt)
            prog.call_unary(dst, vt.find(self.fn))
        elif len(self.args) == 2:
            assert(isinstance(args[0], Var) or isinstance(args[0], Const))
            assert(isinstance(args[1], Var) or isinstance(args[1], Const))
            arg[0].compile(0, prog, mem, vt)
            arg[1].compile(1, prog, mem, vt)
            prog.call_binary(dst, vt.find(self.fn))
        else:
            raise ValueError("multi-variable call is not implemented yet")            
            

class Const:
    def __init__(self, val):
        self.val = float(val)        

    def __repr__(self):
        return f"Const({self.val})"

    def compile(self, dst, prog, mem, vt):
        prog.load_const(dst, mem.constant(self.val))


class Var:
    def __init__(self, name):
        self.name = str(name)

    def __repr__(self):
        return f"Var('{self.name}')"

    def compile(self, dst, prog, mem, vt):
        prog.load_mem(dst, mem.index(self.name))


class Label:
    def __init__(self, label):
        self.label = label        

    def __repr__(self):
        return f"Label('{self.label}')"

    def compile(self, dst, prog, mem, vt):
        prog.set_label(self.label)
        
        
class Branch:
    def __init__(self, cond, true_label, false_label=None):
        self.cond = cond
        self.true_label = true_label
        self.false_label = false_label

    def __repr__(self):
        return f"Branch({self.true_label}))"

    def compile(self, dst, prog, mem, vt):
        if self.cond == True:
            prog.branch(self.true_label)
            return

        self.cond.compile(dst, prog, mem, vt)                    
        if self.false_label is None:
            prog.branch_if(dst, self.true_label)
        else:
            prog.branch_if_else(dst, self.true_label, self.false_label)            
            

class Eq:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"{self.lhs} = {self.rhs}"

    def compile(self, dst, prog, mem, vt):
        self.rhs.compile(dst, prog, mem, vt)
        prog.save_mem(dst, mem.index(self.lhs.name))


class Model:
    def __init__(self, states, obs, eqs):
        self.states = states
        self.obs = obs
        self.eqs = eqs

    def __repr__(self):
        return f"""Model(
            states: {self.states}
            obs: {self.obs}
            eqs: {self.eqs})
        )"""

    def compile(self, idx, prog, mem, vt):
        for eq in self.eqs:
            eq.compile(0, prog, mem, vt)

        prog.append_epilogue(idx)
        prog.append_text_section(mem.consts, vt.vt())
        # we need to prepend and not append prologue
        # because we don't know the stack size until
        # after compiling the body of the function        
        prog.prepend_prologue()




