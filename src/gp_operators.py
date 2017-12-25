import math
import operator
import numpy as np


### OPERATORS ###

def adjust_vector(v):
    return [float(x) for x in np.nan_to_num(v)]

# Operator add1
def add1(v1, v2):
    return list(adjust_vector(map(operator.add, v1, v2)))

# Operator add2
def add2(v, s):
    return list(adjust_vector([x+s for x in v]))

# Operator add3
def add3(s1, s2):
    return float(np.nan_to_num(s1 + s2))

# Operator sub1
def sub1(v1, v2):
    return list(adjust_vector(map(operator.sub, v1, v2)))

# Operator sub2
def sub2(v, s):
    return list(adjust_vector([x-s for x in v]))

# Operator sub3
def sub3(s1, s2):
    return float(np.nan_to_num(s1 - s2))

# Operator mul1
def mul1(v1, v2):
    return list(adjust_vector(map(operator.mul, v1, v2)))

# Operator mul2
def mul2(v1, v2):
    return float(np.nan_to_num(sum(mul1(v1, v2))))

# Operator mul3
def mul3(v, s):
    return list(adjust_vector([x*s for x in v]))

# Operator mul4
def mul4(s1, s2):
    return float(np.nan_to_num(s1 * s2))

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1.0

# Operator div1
def div1(v1, v2):
    return list(adjust_vector(map(protectedDiv, v1, v2)))

# Operator div2
def div2(v, s):
    return list(adjust_vector([protectedDiv(x, s) for x in v]))

# Operator div3
def div3(s1, s2):
    return float(np.nan_to_num(protectedDiv(s1, s2)))

# Vector normalization (logistic)
def norm_logistic(v):
    v_exp = adjust_vector(np.exp(map(operator.neg, v)))
    return [(0.0001 + 1 / (1 + s)) for s in v_exp]

def norm_logistic_scalar(s):
    return (0.0001 + 1 / (1 + float(np.nan_to_num(np.exp(s)))))

# Vector normalization (unit)
def norm_unit(v):
    magnitude = 0.0
    for s in v:
        magnitude += s*s
    magnitude = math.sqrt(magnitude)
    return list(adjust_vector([s/magnitude for s in v]))

# Operator pow1
def pow1(v1, v2):
#    v1 = norm_unit(map(operator.abs, v1))
#    v2 = norm_unit(v2)
    v1 = norm_logistic(v1)
    v2 = norm_logistic(v2)
    return list(adjust_vector(map(operator.pow, v1, v2)))

# Operator pow2
def pow2(v, s):
#    v = norm_unit(map(operator.abs, v))
    v = norm_logistic(v)
    s = norm_logistic_scalar(s)
    return list(adjust_vector([operator.pow(x,s) for x in v]))

# Operator pow3
def pow3(s1, s2):
    return float(np.nan_to_num(math.pow(abs(s1), s2)))

# Operator neg
def neg(v):
    return map(operator.neg, v)

# Operator ident
def ident(*ss):
    return tuple(ss)

# Operator ident with vector lenght normalization (unit)
def ident_unit(*ss):
    return tuple(norm_unit(ss))

# Operator ident with probabilistic output
def ident_softmax(*ss):
    v_exp = list(adjust_vector(np.exp(ss)))
    return tuple(v_exp / np.sum(v_exp, axis=0))
