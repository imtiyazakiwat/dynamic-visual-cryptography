import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def color_image():
    yy, xx = np.mgrid[0:96, 0:128]
    rgb = np.stack([xx * 2, yy * 2, (xx + yy)], axis=2).astype(np.uint8)
    return Image.fromarray(rgb)


@pytest.fixture
def gray_image():
    yy, xx = np.mgrid[0:64, 0:80]
    return Image.fromarray(((xx * 3 + yy * 2) % 256).astype(np.uint8))


@pytest.fixture
def binary_image():
    array = np.full((64, 64), 255, dtype=np.uint8)
    array[16:48, 16:48] = 0
    return Image.fromarray(array)


@pytest.fixture
def workdir(tmp_path):
    return tmp_path / "runs"
