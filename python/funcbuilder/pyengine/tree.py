import sys
import numbers
from sympy import *

COUNT_TEMP_XMM = 3 if sys.platform == "win32" else 13


class Unary:
    def __init__(self, op, arg):
        self.op = op
        self.arg = arg

    def __repr__(self):
        return f"Unary[reg={self.reg}]('{self.op}', {self.arg})"
        
    def alloc(self, lefty):
        self.reg = self.arg.alloc(True)
        return self.reg

    def compile(self, prog, mem, vt):
        dst = self.arg.compile(prog, mem, vt)
        
        assert(dst != 1 and dst < prog.first_shadow + prog.count_shadows)

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
            
        return dst            


class Binary:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

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
        self.reg = l+1 if l == r else max(l, r)
        return self.reg        

    def compile(self, prog, mem, vt):        
        """
            compile performs the second pass of the Sethi–Ullman algorithm.
            
            In this pass, physical registers are assigned and the actual 
            machine code is generated.
            
            compile returns the physical register.
            
            We use the following conventinos:
            
            1. Registers 0 and 1, r(0) and r(1), i.e., (XMM1/XMM2 in AMD 
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
        t = self.idx
        
        # last accessible physical register
        last = prog.first_shadow + prog.count_shadows - 1   
        
        # the physical register holding the result of the current operation
        dst = prog.first_shadow + self.reg if self.reg < prog.count_shadows else 0
        
        if self.right.reg == self.left.reg:
            l = self.left.compile(prog, mem, vt)
            
            # we need to spill the result of the left limb if it is in the 
            # last physical reg to open space for the result of the right limb
            if l == 0 or l == last:
                prog.save_mem(l, t) 
                l = 0
            else:
                prog.fmov(l+1, l)   # to prevent collision with r
                l = l + 1
                
            r = self.right.compile(prog, mem, vt)
            
            if l == 0:
                prog.load_mem(1, t)
                l = 1                        
        elif self.right.reg > self.left.reg:
            r = self.right.compile(prog, mem, vt)
            
            # we don't need to check with last or move to a higher register
            # because r is either 0 or strictly greater than l
            if r == 0:
                prog.save_mem(r, t)
        
            l = self.left.compile(prog, mem, vt)
                
            if r == 0:
                prog.load_mem(1, t)
                r = 1   # 1 not 0 because l can be 0
        else:   # self.right.reg < self.left.reg:
            l = self.left.compile(prog, mem, vt)
            
            if l == 0:
                prog.save_mem(l, t) 
                
            r = self.right.compile(prog, mem, vt)
            
            if l == 0:
                prog.load_mem(1, t)
                l = 1                
        
        self.emit(dst, l, r, prog, mem, vt)        
            
        return dst            

        
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
            raise ValueError(f"binary op {self.op} not defined")


class Call:
    def __init__(self, fn, args):
        self.fn = fn
        self.args = args
        
    def __repr__(self):
        return f"Call[reg=?]({self.fn},...)"
        
    def alloc(self, lefty):
        pass
        
    def compile(self, prog, mem, vt):
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
            
            prog.call_binary(0, vt.find(self.fn))            
        else:
            raise ValueError("multi-variable call is not implemented yet")            
            
        return 0
            

class Const:
    def __init__(self, val):
        self.val = float(val)                

    def __repr__(self):
        return f"Const[reg={self.reg}]({self.val})"
        
    def alloc(self, lefty):
        self.reg = 0 if lefty else 1
        return self.reg        

    def compile(self, prog, mem, vt):
        dst = prog.first_shadow + self.reg
        prog.load_const(dst, mem.constant(self.val))
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

    def compile(self, prog, mem, vt):
        dst = prog.first_shadow + self.reg
        prog.load_mem(dst, mem.index(self.name))
        return dst


class Label:
    def __init__(self, label):
        self.label = label        

    def __repr__(self):
        return f"Label[reg=?]('{self.label}')"
        
    def alloc(self, lefty):
        pass

    def compile(self, prog, mem, vt):
        prog.set_label(self.label)
        return 0
        
        
class Branch:
    def __init__(self, cond, true_label, false_label=None):
        self.cond = cond
        self.true_label = true_label
        self.false_label = false_label

    def __repr__(self):
        return f"Branch[reg=?]({self.true_label}))"
        
    def alloc(self, lefty):
        pass

    def compile(self, prog, mem, vt):
        if self.cond == True:
            prog.branch(self.true_label)
            return

        dst = self.cond.compile(prog, mem, vt)
        
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
        
    def alloc(self, lefty):
        self.rhs.alloc(True)

    def compile(self, prog, mem, vt):
        dst = self.rhs.compile(prog, mem, vt)
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
            eq.alloc(True)
            eq.compile(prog, mem, vt)

        prog.append_epilogue(idx)
        prog.append_text_section(mem.consts, vt.vt())
        # we need to prepend and not append prologue
        # because we don't know the stack size until
        # after compiling the body of the function        
        prog.prepend_prologue()




