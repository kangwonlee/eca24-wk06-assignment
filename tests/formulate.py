import sympy as sym


m, c, k, t = sym.symbols('m c k t', real=True)

x = sym.Function('x')

print(x(t))

lhs = m * sym.Derivative(x(t), t, t) + c * sym.Derivative(x(t), t) + k * x(t)

print(lhs)

s = sym.dsolve(lhs, x(t), )

print(s)

s1 = s.rhs

print('x(t) =', s1)

x0 = s1.subs({t:0})

print(x0)

v = sym.Derivative(s1, t)
print('v =', v)

v0 = sym.expand(v.subs({t:0}))
print('v0 =', v0)
