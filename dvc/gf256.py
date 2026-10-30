"""Vectorized arithmetic in GF(2^8) using the AES polynomial x^8 + x^4 + x^3 + x + 1.

Addition is XOR. Multiplication uses a precomputed 256 x 256 table so that
``MUL[c][array]`` multiplies a whole image plane by the constant ``c`` with one
NumPy gather. Decoding therefore needs only table lookups and XOR.
"""

from __future__ import annotations

import numpy as np

_POLY = 0x11B


def _build_tables():
    exp = np.zeros(512, dtype=np.uint8)
    log = np.zeros(256, dtype=np.int32)
    value = 1
    for power in range(255):
        exp[power] = value
        log[value] = power
        # Multiply by the generator 0x03: value * 2 xor value.
        doubled = value << 1
        if doubled & 0x100:
            doubled ^= _POLY
        value = doubled ^ value
    exp[255:510] = exp[0:255]
    mul = np.zeros((256, 256), dtype=np.uint8)
    nonzero = np.arange(1, 256)
    for a in range(1, 256):
        mul[a, 1:] = exp[(log[a] + log[nonzero]) % 255]
    return exp, log, mul


EXP, LOG, MUL = _build_tables()


def gf_mul(a: int, b: int) -> int:
    return int(MUL[a, b])


def gf_inv(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("0 has no inverse in GF(256)")
    return int(EXP[(255 - LOG[a]) % 255])


def gf_pow(a: int, exponent: int) -> int:
    result = 1
    for _ in range(exponent):
        result = gf_mul(result, a)
    return result


def lagrange_at_zero(xs: list[int]) -> list[int]:
    """Coefficients c_j such that f(0) = XOR_j c_j * f(x_j) for any polynomial of degree < len(xs)."""
    if len(set(xs)) != len(xs):
        raise ValueError("Share x-coordinates must be distinct")
    if any(x == 0 for x in xs):
        raise ValueError("x = 0 is reserved for the secret")
    coeffs = []
    for j, xj in enumerate(xs):
        numerator = 1
        denominator = 1
        for m, xm in enumerate(xs):
            if m == j:
                continue
            numerator = gf_mul(numerator, xm)  # (0 - xm) = xm in characteristic 2
            denominator = gf_mul(denominator, xm ^ xj)
        coeffs.append(gf_mul(numerator, gf_inv(denominator)))
    return coeffs


def scale(plane: np.ndarray, constant: int) -> np.ndarray:
    """Multiply every element of a uint8 array by a GF(256) constant."""
    return MUL[constant][plane]
