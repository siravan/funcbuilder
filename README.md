# funcbuilder

FuncBuilder is an experimental just-in-time (JIT) compiler. It is a companion of [symjit](https://github.com/siravan/symjit), which is a jit compiler for sympy expressions. FuncBuilder provides a lower-level API that allows for step-by-step constructions of fast functions. While symjit uses two different backends (based on Rust and Python), FuncBuilder uses a pure Python code generator with no dependency on any packages outside the standard library. 

The FuncBuilder API is inspired by [llvmlite](https://github.com/numba/llvmlite), but is not identical. 

# Installation

As a pure Python package, FuncBuilder can be installed from PyPi as

```
python -m pip install FuncBuilder
```

# Tutorial

The workflow is as follows:

1. Create a `Builder` object. The function arguments are defined as this stage.
2. Add instructions step-by-step.
3. Compile to machine code. The output variable is defined at this stage.

A simple example,

```python
from funcbuilder import FuncBuilder

B, [x, y] = FuncBuilder('x', 'y')
a = B.fadd(x, y)
f = B.compile(a)

print(f(1.0, 2.0))  # prints 3.0
```

`FuncBuilder` accepts as arguments the names of the input variables (currently, the type of all variables is implicitely float64) and returns a tuple. The first item is a `Builder` object and the second a list of variables correspoding to the input variables.

Afterward, the program is built stepwise using the Builder API (discussed below). In the example above, `fadd` takes two variables `x` and `y` as inputs and returns the result of floating point addition as `a`.

Finally, `f = B.compile(a)` compiles the program and returns a function `f`, which has a type signature of `double f(double x, double y)`.

# Builder API

These are functions exported from the Builder object to add instructions.

## Standard Arithmatic Binary Operations

All these functions accept two double variables or constants as input and return a new temporary variable.

* `fadd(x, y)`: floating point addition.
* `fsub(x, y)`: floating point subtraction.
* `fmul(x, y)`: floating point addition.
* `fdiv(x, y)`: floating point division.
* `pow(x, y)`: floating point power. Special shortcut codes are generated when `y` is 1, 2, 3, -1, -2, 0.5, 1.5, and -0.5. Otherwise, `pow` standard function is called.

## Standard Unary Operations

These functions accept a single double variable or constant as input and return a new temporary variable.

* `square(x)`: returns `x**2`.
* `cube(x)`: returns `x**3`.
* `recip(x)`: returns `1/x`.
* `sqrt(x)`: returns the square root of `x`.

## Transcendental Functions

These functions also accept a double variable or constant as input and return a new temporary variable.

* `exp(x)`
* `log(x)`
* `sin(x)`
* `cos(x)`
* `tan(x)`
* `sinh(x)`
* `cosh(x)`
* `tanh(x)`
* `asin(x)`
* `acos(x)`
* `atan(x)`
* `asinh(x)`
* `acosh(x)`
* `atanh(x)`

## Comparison Functions

The following functions compare two floating point numbers. The result is encoded as a floating point number, with 0.0 corresponding to False and an all 1 mask (= NaN) to True.

* `lt(x, y)`: `x` is less than `y`.
* `leq(x, y)`: `x` is less than or equal to `y`.
* `gt(x, y)`: `x` is greater than `y`.
* `geq(x, y)`: `x` is greater than or equal to `y`.
* `eq(x, y)`: `x` is equal to `y`.
* `neq(x, y)`: `x` is not equal to `y`.

## Logical Operations

Boolean variables (encoded as float, discussed above) can be combined using,

* `and_(x, y)`: `x` and `y`.
* `or_(x, y)`: `x` or `y`.
* `xor(x, y)`: `x` xor `y`.
* `not_(x)`: not `x`.

Note that `and_`, `or_`, and `not_` have trailing underscores to distinguish them from Python's reserved words. 

## Branching Operations

Currently, FuncBuilder provides a simple API to implement conditional jumps and loops based on setting labels and branch instructions.

* `set_label(label)`: set a label at the current instruction position (labels are strings).
* `branch(label)`: unconditional jump to label.
* `branch(cond, true_label)`: conditional jump to `true_label` if `cond` (a variable) is True. If `cond` is False, the control flow continues. 
* `branch(cond, true_label, false_label)`: conditional jump to `true_label` if `cond` (a variable) is True and to `false_label` if it is False. 

## Phi Node

All the builder functions discussed up to this point return a new variable, as is expected from [static single-assignment (SSA)](https://en.wikipedia.org/wiki/Static_single-assignment_form) forms. However, to generate loops and accumulators, one needs the ability to reassign a value to the same variable (e.g., `i = i + 1`). Compilers, including LLVM, accommodate this by including Phi nodes. Therefore, FuncBuilder also provides a simple implementation of a Phi node as `phi` function to allow for constructing loops. Calling `phi` returns an uninitiated `Phi` object. During program construction, one assigns values to each node by calling the `add_incoming` function of each Phi node. Note that `add_incoming` is a function of the Phi object and not the builder object. 

The following example shows how to calculate factorial. Note that the two Phi nodes (`p` and `n`) are updated by calling `add_incoming`.

```python
from funcbuilder import FuncBuilder

B, [x] = FuncBuilder('x')

p = B.phi()
p.add_incoming(1.0)

n = B.phi()
n.add_incoming(x)

B.set_label('loop')
r1 = B.fmul(p, n)
p.add_incoming(r1)
r2 = B.fsub(n, 1.0)
n.add_incoming(r2)
r3 = B.geq(n, 1.0)
B.cbranch(r3, 'loop')

f = B.compile(p)

assert(f(5) == 120)
```


