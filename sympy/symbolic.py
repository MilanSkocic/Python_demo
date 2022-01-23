r"""
Symbolic maths.
"""


import sympy as sym
import numpy as np

# symbolic expression
a, n, t = sym.symbols("a n, t", real=True)
f = a*t**n
print(f"Symbolic expression: f={f}")

# lambdify symbolic expression
f_numeric = sym.lambdify((a, n, t), f, modules="numpy")
x = np.linspace(0, 10, 10)
y = f_numeric(1, 1, x)
print("Lambdify expression: f(a, n, t)")
print(y)

p  = sym.Symbol("p", real=True)
pk = list(sym.symbols("p[0:2]", real=True))

f = f.subs(dict(zip((a, n), pk)))
print(f"Lambdify expression in vectorial form for parameters: f(p, t)={f}")

f_numeric = sym.lambdify([p,t], f, modules="numpy")
y = f_numeric((1,1), x)
print(y)
