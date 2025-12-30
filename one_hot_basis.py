import numpy as np
from itertools import product
from typing import Iterator, Literal, Union, List

'''
def iter_binary_strings(n, reverse=False):
    for bits in product(['0', '1'], repeat=n):
        if reverse:
            yield ''.join(bits)[::-1]
        else:
            yield ''.join(bits)

def iter_gray_code(
    n: int,
    reverse: bool = False,
    start: int = 0,
    count: int | None = None,
    out: Literal["int", "str", "bits"] = "str",
) -> Iterator[Union[int, str, List[int]]]:
    """
    Iterate n-bit Gray code: g(i) = i ^ (i >> 1).

    Args:
        n: number of bits.
        reverse_bits: if True, reverse the bit order of the output (mirror).
        start: starting index i (mod 2**n).
        count: how many codes to emit; default emits full cycle 2**n.
        out: "int" | "str" | "bits".

    Yields:
        Gray code in the chosen format.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    N = 1 << n
    if count is None:
        count = N
    if count < 0:
        raise ValueError("count must be >= 0")

    start %= N

    def reverse_nbits(x: int) -> int:
        y = 0
        for _ in range(n):
            y = (y << 1) | (x & 1)
            x >>= 1
        return y

    for k in range(count):
        i = (start + k) % N
        g = i ^ (i >> 1)

        if reverse and n > 1:
            g = reverse_nbits(g)

        if out == "int":
            yield g
        elif out == "str":
            yield format(g, f"0{n}b")
        elif out == "bits":
            s = format(g, f"0{n}b")
            yield [int(c) for c in s]
        else:
            raise ValueError("out must be one of: 'int', 'str', 'bits'")

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
'''

def ith_lex_binary(n: int, i: int, reverse=True) -> str:
    """
    Return the n-bit binary representation of i (zero-padded).
    Example: binary_bits(5, 4) -> '0101'
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if i < 0 or i >= (1 << n):
        raise ValueError(f"i must satisfy 0 <= i < 2**n = {1 << n}.")
    bitstring = format(i, f"0{n}b")
    if reverse:
        return bitstring[::-1]
    else:
        return bitstring


def ith_lex_onehot(n: int, q: int, i: int, reverse=True) -> str:
    """
    Return the i-th bitstring (0-based) in the lexicographic (non-Gray)
    one-hot product basis with n blocks of size q.

    Layout: [block 0 | block 1 | ... | block n-1]
    """
    if n <= 0 or q <= 0:
        raise ValueError("n and q must be positive.")
    if i < 0 or i >= q**n:
        raise ValueError(f"i must satisfy 0 <= i < q**n = {q**n}.")

    bits = []
    x = i
    for _ in range(n):
        d = x % q          # base-q digit
        x //= q
        block = ['0'] * q
        block[d] = '1'
        bits.append(''.join(block))

    bitstring = ''.join(bits)
    if reverse:
        return bitstring
    else:
        return bitstring[::-1]


def ith_gray_onehot(n: int, q: int, i: int, reverse=True) -> str:
    """
    Return the i-th bitstring (0-based) in the reflected-base-q Gray/snake
    ordering of the q^n one-hot product basis.

    Output layout: [block 0 | block 1 | ... | block n-1], each block length q.
    Within each block, index 0 is the leftmost bit.
    """
    if n <= 0 or q <= 0:
        raise ValueError("n and q must be positive.")
    if i < 0 or i >= q**n:
        raise ValueError(f"i must satisfy 0 <= i < q**n = {q**n}.")

    # base-q digits d_k, k=0..n-1 (least significant first)
    d = []
    x = i
    for _ in range(n):
        d.append(x % q)
        x //= q

    # compute coordinates a_k using parity of more-significant digits
    a = [0] * n
    parity = 0  # parity of sum of digits above current k (mod 2), built MSB->LSB
    for k in range(n - 1, -1, -1):
        if parity == 0:
            a[k] = d[k]
        else:
            a[k] = (q - 1) - d[k]
        parity ^= (d[k] & 1)  # update parity with current digit's parity

    # build one-hot blocks
    blocks = []
    for k in range(n):
        bits = ['0'] * q
        bits[a[k]] = '1'
        blocks.append(''.join(bits))

    bitstring = ''.join(blocks)
    if reverse:
        return bitstring
    else:
        return bitstring[::-1]
    

def ith_gray_binary(n: int, i: int, reverse=True) -> str:
    def gray_int(i: int) -> int:
        """Return the integer value of the i-th binary-reflected Gray code."""
        if i < 0:
            raise ValueError("i must be >= 0.")
        return i ^ (i >> 1)
    """
    Return the i-th Gray code as an m-bit string (zero-padded).
    Example: gray_bits(3, 4) -> '0010'
    """
    if n <= 0:
        raise ValueError("m must be positive.")
    g = gray_int(i)

    bitstring = format(g, f"0{n}b")
    if reverse:
        return bitstring[::-1]
    else:
        return bitstring

 
if __name__ == "__main__":
    for i in range(16):
        print(ith_lex_binary(4, i, True), end=" ")
        print(ith_gray_binary(4, i, True), end=" ")
        print(ith_lex_onehot(2, 4, i, True), end=" ")
        print(ith_gray_onehot(2, 4, i, True), end=" ")
        print()