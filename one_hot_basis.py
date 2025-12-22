import numpy as np
from itertools import product

def product_rightmost_outer(*iterables):
    return [
        tuple(reversed(t))
        for t in product(*reversed(iterables))
    ]

def reflected_qary_gray_digits(n, q):
    """Reflected q-ary Gray code on n digits (MSD changes slowest)."""
    if n < 0 or q <= 0:
        raise ValueError("Need n>=0, q>0.")
    if n == 0:
        return [()]
    prev = reflected_qary_gray_digits(n - 1, q)
    out = []
    for a in range(q):
        block = prev if (a % 2 == 0) else prev[::-1]
        out.extend((a,) + t for t in block)
    return out

def one_hot_bounce_bitstrings(n, q, rev=True):
    """
    The *first* register is the fastest-changing one.
    """
    digits_list = reflected_qary_gray_digits(n, q)

    one_hot_bitstrings = []
    for digits in digits_list:
        # ---- KEY FIX: reverse digit order when mapping to registers ----
        if rev:
            digits = digits[::-1]
        chunks = []
        for d in digits:
            reg = ['0'] * q
            reg[d] = '1'
            chunks.append(''.join(reg))
        one_hot_bitstrings.append(''.join(chunks))
    return one_hot_bitstrings

# D-dimensional
def one_hot_bounce_bitstrings_product(D, n, q, output_mode="str_tuple"):
    """
    Now: the *first* register is the fastest-changing one.
    """
    one_hot_bitstrings = one_hot_bounce_bitstrings(n, q)

    if output_mode == "str":
        one_hot_D_copies = [one_hot_bitstrings]*D
        one_hot_D_bitstrings = []
        for s in product_rightmost_outer(*one_hot_D_copies):
            one_hot_D_bitstrings.append("".join(s))
        return one_hot_D_bitstrings
    elif output_mode == "str_tuple":
        one_hot_D_copies = [one_hot_bitstrings]*D
        return list(product_rightmost_outer(*one_hot_D_copies))


    """
    one_hot_basis = np.zeros((len(digits_list), n * q), dtype=dtype)
    for row, digits in enumerate(digits_list):
        digits = digits[::-1]  # ---- KEY FIX ----
        for r, d in enumerate(digits):
            one_hot_basis[row, r * q + d] = 1
    
    one_hot_D_copies = [one_hot_basis]*D
    out = []
    for s in product_rightmost_outer(*one_hot_D_copies):
        out.append("".join(s))
    return out
    """