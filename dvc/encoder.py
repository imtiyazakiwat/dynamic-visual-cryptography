"""Phase 3: threshold encoding with no pixel expansion.

Recovery modes and the share scheme each one uses:

* ``xor`` with k == n -> scheme ``xor-nn``. Shares 1..n-1 are uniform keystream
  bytes; share n is the secret XOR all others. XOR of all n shares is the
  secret, bit-exact.
* ``xor`` with k < n -> scheme ``gf256``. Each pixel byte is the constant term of
  a random degree-(k-1) polynomial over GF(2^8); share i holds f(i). Any k shares
  recover the secret bit-exact by Lagrange interpolation (XOR plus table
  lookups); any k-1 shares are uniformly distributed and independent of it.
* ``or`` -> scheme ``rg-or``. Each channel is halftoned to ink/no-ink bits
  (Floyd-Steinberg), then encoded with the Chen-Tsao (k, n) random-grid rule:
  per pixel, k bits are drawn with XOR equal to the secret bit, the remaining
  n-k shares receive copies of randomly chosen bits among those k, and the n
  bits are randomly permuted across shares. Stacking (OR) any k shares shows the
  secret as a contrast difference; any k-1 shares are uniform noise.

Every share has the same H x W x C shape as the secret.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import gf256
from .entropy import SessionKey
from .errors import INVALID_PARAMS, DVCError

RECOVERY_OR = "or"
RECOVERY_XOR = "xor"
SCHEME_XOR_NN = "xor-nn"
SCHEME_GF256 = "gf256"
SCHEME_RG_OR = "rg-or"

MAX_SHARES = 16
BAND_ROWS = 256

INK = 0      # pixel value for an "ink" (black / opaque) bit in an OR share
CLEAR = 255  # pixel value for a "clear" (white / transparent) bit


@dataclass
class Share:
    index: int          # 1-based share number; also the x-coordinate for gf256
    array: np.ndarray   # H x W x C uint8
    scheme: str

    def canonical_bytes(self) -> bytes:
        return share_canonical_bytes(self.array)


def share_canonical_bytes(array: np.ndarray) -> bytes:
    header = f"{array.shape[0]}x{array.shape[1]}x{array.shape[2]}|uint8|".encode()
    return header + np.ascontiguousarray(array, dtype=np.uint8).tobytes()


def share_digest(array: np.ndarray) -> str:
    return hashlib.sha256(share_canonical_bytes(array)).hexdigest()


def validate_params(k: int, n: int, recovery: str) -> None:
    if recovery not in (RECOVERY_OR, RECOVERY_XOR):
        raise DVCError(INVALID_PARAMS, f"Recovery mode must be 'or' or 'xor', got '{recovery}'.")
    if not isinstance(k, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise DVCError(INVALID_PARAMS, "k and n must be integers.")
    if k < 2:
        raise DVCError(INVALID_PARAMS, f"k must be at least 2, got {k}.")
    if n < k:
        raise DVCError(INVALID_PARAMS, f"n must be at least k, got k={k}, n={n}.")
    if n > MAX_SHARES:
        raise DVCError(INVALID_PARAMS, f"n must be at most {MAX_SHARES}, got {n}.")


def scheme_for(k: int, n: int, recovery: str) -> str:
    if recovery == RECOVERY_OR:
        return SCHEME_RG_OR
    return SCHEME_XOR_NN if k == n else SCHEME_GF256


def halftone_plane(plane: np.ndarray) -> np.ndarray:
    """Floyd-Steinberg halftone of one uint8 plane. Returns ink bits (1 = dark)."""
    bilevel = np.asarray(Image.fromarray(plane, mode="L").convert("1", dither=Image.Dither.FLOYDSTEINBERG))
    return (~bilevel).astype(np.uint8)


def ink_bits(array: np.ndarray, image_mode: str) -> np.ndarray:
    """H x W x C ink bits for the secret: thresholded for binary, halftoned otherwise."""
    if image_mode == "binary":
        return (array < 128).astype(np.uint8)
    return np.stack([halftone_plane(array[:, :, c]) for c in range(array.shape[2])], axis=2)


def _bits_from_bytes(stream: np.ndarray, count: int) -> np.ndarray:
    return np.unpackbits(stream)[:count]


def _encode_xor_nn(secret: np.ndarray, key: SessionKey, n: int) -> list[np.ndarray]:
    height, width, channels = secret.shape
    plane_size = height * width
    shares = [np.empty_like(secret) for _ in range(n)]
    for c in range(channels):
        accumulator = secret[:, :, c].copy()
        for i in range(n - 1):
            random_plane = key.keystream(f"xor-nn|ch{c}|share{i + 1}", plane_size).reshape(height, width)
            shares[i][:, :, c] = random_plane
            accumulator ^= random_plane
        shares[n - 1][:, :, c] = accumulator
    return shares


def _encode_gf256(secret: np.ndarray, key: SessionKey, k: int, n: int) -> list[np.ndarray]:
    height, width, channels = secret.shape
    plane_size = height * width
    shares = [np.empty_like(secret) for _ in range(n)]
    xs = list(range(1, n + 1))
    for c in range(channels):
        coefficients = [secret[:, :, c]]
        for j in range(1, k):
            coefficients.append(key.keystream(f"gf256|ch{c}|coef{j}", plane_size).reshape(height, width))
        for share_number, x in enumerate(xs):
            # Horner's rule: f(x) = (((a_{k-1} x + a_{k-2}) x + ...) x + a_0)
            value = coefficients[-1].copy()
            for coefficient in reversed(coefficients[:-1]):
                value = gf256.scale(value, x) ^ coefficient
            shares[share_number][:, :, c] = value
    return shares


def _encode_rg_or(secret_ink: np.ndarray, key: SessionKey, k: int, n: int) -> list[np.ndarray]:
    height, width, channels = secret_ink.shape
    share_bits = np.empty((n, height, width, channels), dtype=np.uint8)
    for c in range(channels):
        for band_start in range(0, height, BAND_ROWS):
            band_end = min(band_start + BAND_ROWS, height)
            band_height = band_end - band_start
            pixels = band_height * width
            label = f"rg-or|ch{c}|band{band_start}"
            secret_bits = secret_ink[band_start:band_end, :, c].reshape(-1)

            # k bits per pixel whose XOR equals the secret bit.
            random_bits = _bits_from_bytes(key.keystream(label + "|bits", -(-(k - 1) * pixels // 8)),
                                           (k - 1) * pixels).reshape(k - 1, pixels)
            last = secret_bits ^ np.bitwise_xor.reduce(random_bits, axis=0)
            base = np.concatenate([random_bits, last[None, :]], axis=0)

            if n > k:
                picks = key.keystream(label + "|copy", (n - k) * pixels * 4).view(np.uint32)
                picks = (picks % k).reshape(n - k, pixels)
                copies = np.take_along_axis(base, picks, axis=0)
                stacked = np.concatenate([base, copies], axis=0)
            else:
                stacked = base

            sort_keys = key.keystream(label + "|perm", n * pixels * 4).view(np.uint32).reshape(n, pixels)
            order = np.argsort(sort_keys, axis=0, kind="stable")
            permuted = np.take_along_axis(stacked, order, axis=0)
            share_bits[:, band_start:band_end, :, c] = permuted.reshape(n, band_height, width)
    return [np.where(share_bits[i] == 1, INK, CLEAR).astype(np.uint8) for i in range(n)]


def encode(secret: np.ndarray, image_mode: str, key: SessionKey, k: int, n: int,
           recovery: str) -> list[Share]:
    validate_params(k, n, recovery)
    if secret.dtype != np.uint8 or secret.ndim != 3:
        raise DVCError(INVALID_PARAMS, "Secret must be an H x W x C uint8 array.", "encode")
    scheme = scheme_for(k, n, recovery)
    if scheme == SCHEME_XOR_NN:
        arrays = _encode_xor_nn(secret, key, n)
    elif scheme == SCHEME_GF256:
        arrays = _encode_gf256(secret, key, k, n)
    else:
        arrays = _encode_rg_or(ink_bits(secret, image_mode), key, k, n)
    return [Share(index=i + 1, array=arrays[i], scheme=scheme) for i in range(n)]
