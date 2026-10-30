"""Phase 7: fidelity and secrecy metrics, implemented with NumPy only."""

from __future__ import annotations

import math

import numpy as np

MAX_PIXEL = 255.0


def mse(reference: np.ndarray, test: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if reference.shape != test.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {test.shape}")
    return float(np.mean((reference - test) ** 2))


def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    """PSNR = 10 log10(MAX^2 / MSE). Returns +inf for identical images."""
    error = mse(reference, test)
    if error == 0:
        return math.inf
    return 10.0 * math.log10(MAX_PIXEL**2 / error)


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    offsets = np.arange(size) - (size - 1) / 2.0
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _filter_valid(plane: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Separable 'valid' convolution built from shifted slices (fast, no SciPy)."""
    size = kernel.size
    height, width = plane.shape
    rows = np.zeros((height - size + 1, width), dtype=np.float64)
    for i, weight in enumerate(kernel):
        rows += weight * plane[i:i + height - size + 1, :]
    out = np.zeros((height - size + 1, width - size + 1), dtype=np.float64)
    for i, weight in enumerate(kernel):
        out += weight * rows[:, i:i + width - size + 1]
    return out


def _ssim_plane(x: np.ndarray, y: np.ndarray) -> float:
    c1 = (0.01 * MAX_PIXEL) ** 2
    c2 = (0.03 * MAX_PIXEL) ** 2
    size = 11 if min(x.shape) >= 11 else (min(x.shape) // 2) * 2 + 1
    kernel = _gaussian_kernel(size, 1.5)
    mu_x = _filter_valid(x, kernel)
    mu_y = _filter_valid(y, kernel)
    sigma_x = _filter_valid(x * x, kernel) - mu_x**2
    sigma_y = _filter_valid(y * y, kernel) - mu_y**2
    sigma_xy = _filter_valid(x * y, kernel) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return float(np.mean(numerator / denominator))


def ssim(reference: np.ndarray, test: np.ndarray) -> float:
    """Mean SSIM (Wang et al. 2004, 11x11 Gaussian window, sigma 1.5), averaged over channels."""
    reference = np.asarray(reference, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if reference.shape != test.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {test.shape}")
    if reference.ndim == 2:
        reference = reference[:, :, None]
        test = test[:, :, None]
    return float(np.mean([_ssim_plane(reference[:, :, c], test[:, :, c]) for c in range(reference.shape[2])]))


def shannon_entropy(array: np.ndarray) -> float:
    """Bits per symbol over the byte histogram (max 8.0 for uint8)."""
    counts = np.bincount(np.asarray(array, dtype=np.uint8).reshape(-1), minlength=256).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; returns 0.0 when either input is constant (no linear relationship)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError("Inputs must have the same number of elements")
    a = a - a.mean()
    b = b - b.mean()
    denominator = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if denominator == 0:
        return 0.0
    return float((a * b).sum() / denominator)


def adjacent_correlation(array: np.ndarray) -> dict:
    """Correlation between horizontally, vertically, and diagonally adjacent pixels."""
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 2:
        array = array[:, :, None]
    return {
        "horizontal": pearson(array[:, :-1, :], array[:, 1:, :]),
        "vertical": pearson(array[:-1, :, :], array[1:, :, :]),
        "diagonal": pearson(array[:-1, :-1, :], array[1:, 1:, :]),
    }


def npcr(a: np.ndarray, b: np.ndarray) -> float:
    """Number of Pixels Change Rate, in percent."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean(a != b) * 100.0)


def uaci(a: np.ndarray, b: np.ndarray) -> float:
    """Unified Average Changing Intensity, in percent."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b) / MAX_PIXEL) * 100.0)


def luminance(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 2:
        return array
    if array.shape[2] == 1:
        return array[:, :, 0]
    return 0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]


def michelson_regions(image: np.ndarray, light_mask: np.ndarray) -> float:
    """Michelson contrast between the mean luminance of secret-light and secret-dark regions."""
    lum = luminance(image)
    if light_mask.all() or not light_mask.any():
        return 0.0
    light = float(lum[light_mask].mean())
    dark = float(lum[~light_mask].mean())
    if light + dark == 0:
        return 0.0
    return (light - dark) / (light + dark)


def contrast_retention(secret: np.ndarray, output: np.ndarray) -> dict:
    """Cout / Cin using the secret's own light/dark split (median luminance threshold)."""
    lum = luminance(secret)
    threshold = np.median(lum)
    light_mask = lum > threshold
    if not light_mask.any() or light_mask.all():
        light_mask = lum >= threshold
    c_in = michelson_regions(secret, light_mask)
    c_out = michelson_regions(output, light_mask)
    retention = c_out / c_in if c_in > 0 else 0.0
    return {"c_in": c_in, "c_out": c_out, "retention": retention}


def vc_relative_contrast(secret_ink: np.ndarray, stacked: np.ndarray) -> float:
    """Visual-cryptography contrast alpha = P(clear | secret clear) - P(clear | secret ink)."""
    clear = stacked == 255
    ink = secret_ink.astype(bool)
    if ink.all() or not ink.any():
        return 0.0
    return float(clear[~ink].mean() - clear[ink].mean())
