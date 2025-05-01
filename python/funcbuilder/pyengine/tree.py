import sys
import numbers
from sympy import *


class Cache:
    def __init__(self, prog):
        self.regs = [None] * (prog.first_shadow + prog.count_shadows)        
        
    def reset(self):
        self.regs = [None] * len(self.regs)        
        
    def assign(self, dst, var):
        for i in range(len(self.regs)):
            if self.regs[i] == var:
                self.regs[i] = None
        self.regs[dst] = var
        
    def find(self, var):
        return self.regs.index(var)
        

class Unary:
    def __init__(self, op, arg):
        self.op = op
        self.arg = arg
        self.reg = 0

    def __repr__(self):
        return f"Unary[reg={self.reg}]('{self.op}', {self.arg})"

    def alloc(self, lefty):
        self.reg = self.arg.alloc(True)
        return self.reg

    def compile(self, prog, mem, vt, cache):
        dst = self.arg.compile(prog, mem, vt, cache)
        cache.assign(dst, None)

        assert dst != 1 and dst < prog.first_shadow + prog.count_shadows

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
            try:
                if dst != 0:
                    prog.fmov(0, dst)
                    cache.assign(dst, None)
                prog.call_unary(0, vt.find(self.op))
                return 0
            except:
                raise ValueError(f"Unary operator {self.op} not found")       

        return dst        


class Binary:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right
        self.reg = 0

    def __repr__(self):
        return f"Binary[reg={self.reg}]('{self.op}', {self.left}, {self.right})"

    def alloc(self, lefty):
        """
        alloc performs the first pass of the Sethi–Ullman algorithm.
        See https://en.wikipedia.org/wiki/Sethi-Ullman_algorithm.
        In this pass, each node of the AST received a logical
        register number from 0 onward.

        alloc returns the logical register.
        """
        l = self.left.alloc(True)
        r = self.right.alloc(False)
        self.reg = l + 1 if l == r else max(l, r)
        return self.reg

    def compile(self, prog, mem, vt, cache):
        """
        compile performs the second pass of the Sethi–Ullman algorithm.

        In this pass, physical registers are assigned and the actual
        machine code is generated.

        compile returns the physical register.

        We use the following conventinos:

        1. Registers 0 and 1, r(0) and r(1), i.e., (XMM0/XMM1 in AMD
            and D0/D1 in aarch64) are scratch registers.

        2. Registers r(prog.first_shadow) to
            r(prog.first_shadow+prog.count_shadows-1) store the
            intermediate values.

        3. Logical registers 0 to prog.count_shadows-1 correspond
            to the intermediate physical registers.

        4. Logical registers with index >= prog.count_shadows map to r(0)
            and are spilled to the stack.

        5. r(1) also holds intermediate numerical constants for some unary
            operations.
        """
        t = mem.spill(self.reg - prog.count_shadows)

        # last accessible physical register
        last = prog.first_shadow + prog.count_shadows - 1

        # the physical register holding the result of the current operation
        dst = prog.first_shadow + self.reg if self.reg < prog.count_shadows else 0
        cache.assign(dst, None)

        if self.right.reg == self.left.reg:        
            l = self.left.compile(prog, mem, vt, cache)

            # we need to spill the result of the left limb if it is in the
            # last physical reg to open space for the result of the right limb
            if l == 0 or l == last:
                prog.save_mem(l, t)
                l = 0
            else:
                prog.fmov(l + 1, l)  # to prevent collision with r
                l = l + 1

            r = self.right.compile(prog, mem, vt, cache)

            if l == 0:
                prog.load_mem(1, t)
                l = 1
        elif self.right.reg > self.left.reg:
            r = self.right.compile(prog, mem, vt, cache)

            # we don't need to check with last or move to a higher register
            # because r is either 0 or strictly greater than l
            if r == 0:
                prog.save_mem(r, t)

            l = self.left.compile(prog, mem, vt, cache)

            if r == 0:
                prog.load_mem(1, t)
                r = 1  # 1 not 0 because l can be 0
        else:  # self.right.reg < self.left.reg:
            l = self.left.compile(prog, mem, vt, cache)

            if l == 0:
                prog.save_mem(l, t)

            r = self.right.compile(prog, mem, vt, cache)

            if l == 0:
                prog.load_mem(1, t)
                l = 1        
        
        return self.emit(dst, l, r, prog, mem, vt)

    def emit(self, dst, l, r, prog, mem, vt):
        if self.op == "plus":
            prog.plus(dst, l, r)
        elif self.op == "minus":
            prog.minus(dst, l, r)
        elif self.op == "times":
            prog.times(dst, l, r)
        elif self.op == "divide":
            prog.divide(dst, l, r)
        elif self.op == "gt":
            prog.gt(dst, l, r)
        elif self.op == "geq":
            prog.geq(dst, l, r)
        elif self.op == "lt":
            prog.lt(dst, l, r)
        elif self.op == "leq":
            prog.leq(dst, l, r)
        elif self.op == "eq":
            prog.eq(dst, l, r)
        elif self.op == "neq":
            prog.neq(dst, l, r)
        elif self.op == "and":
            prog.and_(dst, l, r)
        elif self.op == "or":
            prog.or_(dst, l, r)
        elif self.op == "xor":
            prog.xor(dst, l, r)
        elif self.op == "select_if":
            prog.select_if(dst, l, r)
        elif self.op == "select_else":
            prog.select_else(dst, l, r)
        else:
            try:
                if l != 0:
                    prog.fmov(0, l)
                    cache.assign(0, None)
                prog.call_binary(0, r, vt.find(self.op))
                return 0
            except:
                raise ValueError(f"Binary operator {self.op} not found")

        return dst


class Call:
    def __init__(self, fn, args):
        self.fn = fn
        self.args = args

    def __repr__(self):
        return f"Call[reg=?]({self.fn},...)\n"

    def alloc(self, lefty):
        return 0

    def compile(self, prog, mem, vt, cache):
        args = self.args

        if len(args) == 1:
            a = args[0]

            if isinstance(a, Var):
                prog.load_mem(0, mem.index(a.name))
            elif isinstance(a, Const):
                prog.load_const(0, mem.constant(a.val))
            else:
                raise ValueError("call arguments should be Var or Const")

            prog.call_unary(0, vt.find(self.fn))
        elif len(self.args) == 2:
            a = args[0]
            b = args[1]

            if isinstance(a, Var):
                prog.load_mem(0, mem.index(a.name))
            elif isinstance(a, Const):
                prog.load_const(0, mem.constant(a.val))
            else:
                raise ValueError("call arguments should be Var or Const")

            if isinstance(b, Var):
                prog.load_mem(1, mem.index(b.name))
            elif isinstance(b, Const):
                prog.load_const(1, mem.constant(b.val))
            else:
                raise ValueError("call arguments should be Var or Const")

            prog.call_binary(0, 1, vt.find(self.fn))
        else:
            raise ValueError("multi-variable call is not implemented yet")

        return 0


class Const:
    def __init__(self, val):
        self.val = float(val)
        self.reg = 0

    def __repr__(self):
        return f"Const[reg={self.reg}]({self.val})"

    def alloc(self, lefty):
        self.reg = 0 if lefty else 1
        return self.reg

    def compile(self, prog, mem, vt, cache):
        dst = prog.first_shadow + self.reg
        prog.load_const(dst, mem.constant(self.val))
        cache.assign(dst, self.val)
        return dst        


class Var:
    def __init__(self, name):
        self.name = str(name)
        self.reg = 0

    def __repr__(self):
        return f"Var[reg={self.reg}]('{self.name}')"

    def alloc(self, lefty):
        self.reg = 0 if lefty else 1
        return self.reg

    def compile(self, prog, mem, vt, cache):            
        dst = prog.first_shadow + self.reg      

        try:
            l = cache.find(self.name)            
            if l == dst:
                return dst
            prog.fmov(dst, l)
        except:
            prog.load_mem(dst, mem.index(self.name))
            
        cache.assign(dst, self.name)
        return dst        


class Label:
    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return f"Label[reg=?]('{self.label}')\n"

    def alloc(self, lefty):
        return 0

    def compile(self, prog, mem, vt, cache):
        cache.reset()
        prog.set_label(self.label)
        return 0


class Branch:
    def __init__(self, cond, true_label, false_label=None):
        self.cond = cond
        self.true_label = true_label
        self.false_label = false_label
        self.reg = 0

    def __repr__(self):
        return f"Branch[reg=?]({self.true_label}))\n"

    def alloc(self, lefty):
        return 0

    def compile(self, prog, mem, vt, cache):
        if self.cond == True:
            prog.branch(self.true_label)
            return

        dst = self.cond.compile(prog, mem, vt, cache)        

        if self.false_label is None:
            prog.branch_if(dst, self.true_label)
        else:
            prog.branch_if_else(dst, self.true_label, self.false_label)
            
        return dst            


class Eq:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"{self.lhs} = {self.rhs}\n"

    def alloc(self, lefty):
        return self.rhs.alloc(True)

    def compile(self, prog, mem, vt, cache):
        dst = self.rhs.compile(prog, mem, vt, cache)        
        prog.save_mem(dst, mem.index(self.lhs.name))
        cache.assign(dst, self.lhs.name)   
        return dst          


class Model:
    def __init__(self, states, obs, eqs):
        self.states = states
        self.obs = obs
        self.eqs = eqs
        self.reg = 0

    def __repr__(self):
        return f"""Model(
            states: {self.states}
            obs: {self.obs}
            eqs: {self.eqs})
        )\n"""

    def alloc(self, lefty):
        for eq in self.eqs:
            self.reg = max(self.reg, eq.alloc(True))
        return self.reg

    def compile(self, y, prog, mem, vt):
        self.alloc(True)
        
        cache = Cache(prog)

        for eq in self.eqs:
            eq.compile(prog, mem, vt, cache)
        
        dst = Var(y.name).compile(prog, mem, vt, cache)
        if dst != 0:
            prog.fmov(0, dst)

        prog.append_epilogue()
        prog.append_text_section(mem.consts, vt.vt())
        # we need to prepend and not append prologue
        # because we don't know the stack size until
        # after compiling the body of the function
        prog.prepend_prologue()
