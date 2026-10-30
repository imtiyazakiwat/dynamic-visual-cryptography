"""Phase 6: reconstruction from an admitted quorum, plus post-processing."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from . import gf256
from .encoder import CLEAR, INK, SCHEME_GF256, SCHEME_RG_OR, SCHEME_XOR_NN
from .errors import MISSING, PARAM_MISMATCH, DVCError
from .preprocess import image_digest

OR_BLUR_RADIUS = 2.5


@dataclass
class Reconstruction:
    raw: np.ndarray          # direct combine output (bit-exact for XOR paths)
    display: np.ndarray      # post-processed for viewing (equals raw for XOR paths)
    used_indices: list[int]
    decode_ms: float
    raw_digest: str


def stack_or(arrays: list[np.ndarray]) -> np.ndarray:
    """Physically stack transparencies: a pixel is ink if it is ink on any share."""
    ink = np.zeros(arrays[0].shape, dtype=bool)
    for array in arrays:
        ink |= array == INK
    return np.where(ink, INK, CLEAR).astype(np.uint8)


def combine_xor(arrays: list[np.ndarray]) -> np.ndarray:
    result = arrays[0].copy()
    for array in arrays[1:]:
        result ^= array
    return result


def combine_gf256(arrays: list[np.ndarray], indices: list[int]) -> np.ndarray:
    coefficients = gf256.lagrange_at_zero(indices)
    result = np.zeros_like(arrays[0])
    for coefficient, array in zip(coefficients, arrays):
        result ^= gf256.scale(array, coefficient)
    return result


def enhance_or(stacked: np.ndarray, radius: float = OR_BLUR_RADIUS, sharpen: bool = False) -> np.ndarray:
    """Make an OR-stacked image easier to read on screen.

    Stacked random-grid output encodes the secret as a density difference between
    fully-inked and half-inked regions. A Gaussian low-pass turns that density into
    gray levels; contrast stretching maps the range back to 0..255; an optional
    unsharp mask restores edges.
    """
    planes = []
    for c in range(stacked.shape[2]):
        plane = Image.fromarray(stacked[:, :, c], mode="L").filter(ImageFilter.GaussianBlur(radius))
        values = np.asarray(plane, dtype=np.float64)
        low, high = np.percentile(values, 1), np.percentile(values, 99)
        if high - low < 1e-6:
            stretched = values
        else:
            stretched = np.clip((values - low) / (high - low) * 255.0, 0, 255)
        plane = Image.fromarray(stretched.astype(np.uint8), mode="L")
        if sharpen:
            plane = plane.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))
        planes.append(np.asarray(plane, dtype=np.uint8))
    return np.stack(planes, axis=2)


def reconstruct(shares: dict[int, np.ndarray], scheme: str, k: int, n: int,
                sharpen: bool = False) -> Reconstruction:
    if len(shares) < k:
        raise DVCError(MISSING, f"Reconstruction needs {k} shares, got {len(shares)}.", "reconstruct")
    shapes = {array.shape for array in shares.values()}
    if len(shapes) != 1:
        raise DVCError(PARAM_MISMATCH, f"Admitted shares have different shapes: {sorted(shapes)}.", "reconstruct")

    start = time.perf_counter()
    if scheme == SCHEME_XOR_NN:
        if len(shares) != n:
            raise DVCError(MISSING, f"The (n, n) XOR scheme needs all {n} shares.", "reconstruct")
        used = sorted(shares)
        raw = combine_xor([shares[i] for i in used])
        display = raw
    elif scheme == SCHEME_GF256:
        used = sorted(shares)[:k]
        raw = combine_gf256([shares[i] for i in used], used)
        display = raw
    elif scheme == SCHEME_RG_OR:
        # Stacking more than k shares only darkens the white regions further,
        # so the quorum uses exactly k shares, as the random-grid rule intends.
        used = sorted(shares)[:k]
        raw = stack_or([shares[i] for i in used])
        display = enhance_or(raw, sharpen=sharpen)
    else:
        raise DVCError(PARAM_MISMATCH, f"Unknown scheme '{scheme}'.", "reconstruct")
    decode_ms = (time.perf_counter() - start) * 1000.0
    return Reconstruction(raw=raw, display=display, used_indices=used, decode_ms=decode_ms,
                          raw_digest=image_digest(raw))
