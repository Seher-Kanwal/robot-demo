"""
Shared test fixtures. Nothing here needs a GPU, the detector, or a model download:
FakeEmbedder reads an object id and a viewpoint id straight out of a synthetic crop's
pixel values, so tests control exactly which crops "look alike".

    channel 0 (B) = object id   -> shared direction for every view of that object
    channel 1 (G) = viewpoint   -> smaller per-view component
    channel 2 (R) = random texture, so crops aren't flat (sharpness checks need edges)

Two views of one object score ~0.8, the identical view 1.0, different objects <= 0.2.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

TEACH_DIR = Path(__file__).resolve().parents[1]
if str(TEACH_DIR) not in sys.path:
    sys.path.insert(0, str(TEACH_DIR))

DIM = 32
BACKGROUND = (200, 200, 200)


class FakeEmbedder:
    def __init__(self, model_id="fake-v1"):
        self.model_id = model_id
        self.dim = DIM
        self.calls = 0

    def embed(self, crops):
        self.calls += 1
        out = np.zeros((len(crops), DIM), np.float32)
        for k, c in enumerate(crops):
            oid = int(np.median(c[..., 0])) % 16
            view = int(np.median(c[..., 1])) % 16
            out[k, oid] = 1.0
            out[k, 16 + view] = 0.5
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def make_crop(oid, view=0, size=(80, 120), seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((size[0], size[1], 3), np.uint8)
    img[..., 0] = oid
    img[..., 1] = view
    img[..., 2] = rng.integers(0, 255, size, dtype=np.uint8)
    return img


def make_frame(objects, shape=(720, 1280), seed=0):
    """objects: list of (oid, view, box). Returns a BGR frame with each object painted in."""
    frame = np.zeros((shape[0], shape[1], 3), np.uint8)
    frame[:] = BACKGROUND
    for k, (oid, view, (x1, y1, x2, y2)) in enumerate(objects):
        frame[y1:y2, x1:x2] = make_crop(oid, view, (y2 - y1, x2 - x1), seed=seed + k)
    return frame


@pytest.fixture
def embedder():
    return FakeEmbedder()
