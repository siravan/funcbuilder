import sys
import numbers
from sympy import *

COUNT_TEMP_XMM = 3 if sys.platform == "win32" else 13


class Unary:
    def __init__(self, op, arg):
        self.op = op
        self.arg = arg
        self.is_pure = arg.is_pure and is_pure(op)

    def __repr__(self):
        return f"Unary[{self.is_pure}]('{self.op}', {self.arg})"

    def compile(self, dst, prog, mem, vt):
        self.arg.compile(0, prog, mem, vt)

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
        elif self.op == "exp":
            prog.call_unary(dst, vt.find("exp"))
        elif self.op == "ln":
            prog.call_unary(dst, vt.find("log"))
        elif self.op == "sin":
            prog.call_unary(dst, vt.find("sin"))
        elif self.op == "cos":
            prog.call_unary(dst, vt.find("cos"))
        elif self.op == "tan":
            prog.call_unary(dst, vt.find("tan"))
        elif self.op == "sinh":
            prog.call_unary(dst, vt.find("sinh"))
        elif self.op == "cosh":
            prog.call_unary(dst, vt.find("cosh"))
        elif self.op == "tanh":
            prog.call_unary(dst, vt.find("tanh"))
        elif self.op == "arcsin":
            prog.call_unary(dst, vt.find("asin"))
        elif self.op == "arccos":
            prog.call_unary(dst, vt.find("acos"))
        elif self.op == "arctan":
            prog.call_unary(dst, vt.find("atan"))
        elif self.op == "arcsinh":
            prog.call_unary(dst, vt.find("asinh"))
        elif self.op == "arccosh":
            prog.call_unary(dst, vt.find("acosh"))
        elif self.op == "arctanh":
            prog.call_unary(dst, vt.find("atanh"))
        else:
            raise ValueError(f"unary op {self.op} not defined")


class Binary:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right
        self.is_pure = left.is_pure and right.is_pure and is_pure(op)

    def __repr__(self):
        return f"Binary[{self.is_pure}]('{self.op}', {self.left}, {self.right})"

    def compile(self, dst, prog, mem, vt):
        sp = mem.push()
        use_reg = sp < prog.count_shadows and self.left.is_pure

        if use_reg:
            self.right.compile(prog.first_shadow + sp, prog, mem, vt)
        else:
            self.right.compile(0, prog, mem, vt)
            prog.save_stack(0, sp)

        self.left.compile(0, prog, mem, vt)
        sp = mem.pop()

        if use_reg:
            r = prog.first_shadow + sp
        else:
            prog.load_stack(1, sp)
            r = 1

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
        else:
            raise ValueError(f"binary op {self.op} not defined")


class Select:
    def __init__(self, cond, true_val, false_val):
        self.cond = cond
        self.true_val = true_val
        self.false_val = false_val
        self.is_pure = cond.is_pure and true_val.is_pure and false_val.is_pure

    def __repr__(self):
        return f"IfElse[{self.is_pure}]({self.cond}, {self.true_val}, {self.false_val})"

    def compile(self, dst, prog, mem, vt):
        sp = mem.push()
        use_reg_cond = (
            sp < prog.count_shadows and self.true_val.is_pure and self.false_val.is_pure
        )

        if use_reg_cond:
            self.cond.compile(prog.first_shadow + sp, prog, mem, vt)
        else:
            self.cond.compile(0, prog, mem, vt)
            prog.save_stack(0, sp)

        sp = mem.push()
        use_reg_false = sp < COUNT_TEMP_XMM and self.true_val.is_pure

        if use_reg_false:
            self.false_val.compile(prog.first_shadow + sp, prog, mem, vt)
        else:
            self.false_val.compile(0, prog, mem, vt)
            prog.save_stack(0, sp)

        self.true_val.compile(0, prog, mem, vt)

        sp = mem.pop()

        if use_reg_false:
            f = prog.first_shadow + sp
        else:
            prog.load_stack(1, sp)
            f = 1

        sp = mem.pop()

        if use_reg_cond:
            c = prog.first_shadow + sp
        else:
            prog.load_stack(2, sp)
            c = 2

        prog.select(dst, c, 0, f)


class Const:
    def __init__(self, val):
        self.val = float(val)
        self.is_pure = True

    def __repr__(self):
        return f"Const({self.val})"

    def compile(self, dst, prog, mem, vt):
        prog.load_const(dst, mem.constant(self.val))


class Var:
    def __init__(self, name):
        self.name = str(name)
        self.is_pure = True

    def __repr__(self):
        return f"Var('{self.name}')"

    def compile(self, dst, prog, mem, vt):
        prog.load_mem(dst, mem.index(self.name))


class Label:
    def __init__(self, label):
        self.label = label
        self.is_pure = True

    def __repr__(self):
        return f"Label('{self.label}')"

    def compile(self, dst, prog, mem, vt):
        prog.set_label(self.label)
        
        
class Branch:
    def __init__(self, cond, true_label, false_label=None):
        self.cond = cond
        self.true_label = true_label
        self.false_label = false_label
        self.is_pure = (cond == True) or cond.is_pure

    def __repr__(self):
        return f"Label('{self.label}')"

    def compile(self, dst, prog, mem, vt):
        if self.cond == True:
            prog.branch(self.true_label)
            return

        self.cond.compile(0, prog, mem, vt)                    
        if self.false_label is None:
            prog.branch_if(self.cond, self.true_label)
        else:
            prog.branch_if_else(self.cond, self.true_label, self.false_label)            
            

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


def is_pure(op):
    ops = [
        "plus",
        "minus",
        "times",
        "division",
        "square",
        "cube",
        "root",
        "recip",
        "abs",
        "neg",
        "lt",
        "leq",
        "gt",
        "geq",
        "eq",
        "neq",
        "and",
        "or",
        "xor",
    ]
    return op in ops



