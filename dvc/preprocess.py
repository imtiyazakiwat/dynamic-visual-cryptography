"""Phase 1: acquisition and normalization of the secret image."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import DIMENSION_LIMIT, UNSUPPORTED_MODE, DVCError

MAX_SIDE = 2048
MIN_SIDE = 8

MODE_BINARY = "binary"
MODE_GRAY = "gray"
MODE_COLOR = "color"


@dataclass
class CanonicalImage:
    """H x W x C uint8 array plus what was learned while normalizing it."""

    array: np.ndarray
    mode: str
    source_mode: str
    orientation_applied: bool
    alpha_flattened: bool

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def channels(self) -> int:
        return int(self.array.shape[2])

    def digest(self) -> str:
        return image_digest(self.array)

    def to_pil(self) -> Image.Image:
        return array_to_pil(self.array)


def image_digest(array: np.ndarray) -> str:
    header = f"{array.shape}|{array.dtype}".encode()
    return hashlib.sha256(header + np.ascontiguousarray(array).tobytes()).hexdigest()


def array_to_pil(array: np.ndarray) -> Image.Image:
    if array.ndim == 3 and array.shape[2] == 1:
        return Image.fromarray(array[:, :, 0], mode="L")
    if array.ndim == 2:
        return Image.fromarray(array, mode="L")
    return Image.fromarray(array, mode="RGB")


def _is_effectively_binary(gray: np.ndarray) -> bool:
    values = np.unique(gray)
    return values.size <= 2 and set(values.tolist()) <= {0, 255}


def normalize(image: Image.Image, force_mode: str | None = None) -> CanonicalImage:
    """Convert any Pillow image into the canonical H x W x C uint8 layout.

    force_mode lets the operator request binary/gray/color explicitly; otherwise
    the mode is detected from the source.
    """
    source_mode = image.mode

    exif_orientation = image.getexif().get(0x0112, 1)
    orientation_applied = exif_orientation not in (1, None)
    if orientation_applied:
        image = ImageOps.exif_transpose(image)

    alpha_flattened = False
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")
        alpha_flattened = True

    width, height = image.size
    if width > MAX_SIDE or height > MAX_SIDE:
        raise DVCError(
            DIMENSION_LIMIT,
            f"Image is {width}x{height}; the maximum is {MAX_SIDE}x{MAX_SIDE}.",
            phase="preprocess",
        )
    if width < MIN_SIDE or height < MIN_SIDE:
        raise DVCError(
            DIMENSION_LIMIT,
            f"Image is {width}x{height}; the minimum is {MIN_SIDE}x{MIN_SIDE}.",
            phase="preprocess",
        )

    if force_mode is None:
        if image.mode == "1":
            mode = MODE_BINARY
        elif image.mode in ("L", "I", "I;16", "F"):
            gray = np.asarray(image.convert("L"))
            mode = MODE_BINARY if _is_effectively_binary(gray) else MODE_GRAY
        else:
            rgb = np.asarray(image.convert("RGB"))
            if np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 1], rgb[:, :, 2]):
                gray = rgb[:, :, 0]
                mode = MODE_BINARY if _is_effectively_binary(gray) else MODE_GRAY
            else:
                mode = MODE_COLOR
    else:
        if force_mode not in (MODE_BINARY, MODE_GRAY, MODE_COLOR):
            raise DVCError(UNSUPPORTED_MODE, f"Unknown mode '{force_mode}'.", phase="preprocess")
        mode = force_mode

    if mode == MODE_COLOR:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    elif mode == MODE_GRAY:
        array = np.asarray(image.convert("L"), dtype=np.uint8)[:, :, None].copy()
    else:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        array = np.where(gray >= 128, 255, 0).astype(np.uint8)[:, :, None]

    return CanonicalImage(
        array=array,
        mode=mode,
        source_mode=source_mode,
        orientation_applied=bool(orientation_applied),
        alpha_flattened=alpha_flattened,
    )


def load_image(source, force_mode: str | None = None) -> CanonicalImage:
    """Load from a path, bytes, file-like object, PIL image, or uint8 array and normalize."""
    if isinstance(source, Image.Image):
        return normalize(source, force_mode=force_mode)
    if isinstance(source, np.ndarray):
        if source.dtype != np.uint8 or source.ndim not in (2, 3) or (source.ndim == 3 and source.shape[2] not in (1, 3, 4)):
            raise DVCError(UNSUPPORTED_MODE, "Arrays must be uint8 with shape HxW, HxWx1, HxWx3, or HxWx4.",
                           phase="preprocess")
        if source.ndim == 3 and source.shape[2] == 1:
            source = source[:, :, 0]
        return normalize(Image.fromarray(source), force_mode=force_mode)
    try:
        if isinstance(source, (bytes, bytearray)):
            image = Image.open(io.BytesIO(source))
        else:
            image = Image.open(source)
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DVCError(UNSUPPORTED_MODE, f"Cannot decode image: {exc}", phase="preprocess") from exc
    return normalize(image, force_mode=force_mode)


def null_path_psnr(canonical: CanonicalImage) -> float:
    """Encode the canonical array as PNG, decode it, and return PSNR vs. the original.

    This certifies that intake/serialization is lossless before any secrecy transform.
    """
    from .metrics import psnr

    buffer = io.BytesIO()
    canonical.to_pil().save(buffer, format="PNG")
    buffer.seek(0)
    reloaded = np.asarray(Image.open(buffer))
    if reloaded.ndim == 2:
        reloaded = reloaded[:, :, None]
    return psnr(canonical.array, reloaded)
