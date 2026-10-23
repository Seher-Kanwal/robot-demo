"""Small image/box helpers shared by the teach-mode modules (no model code here)."""
import base64

import cv2
import numpy as np


def clip_box(shape, box, pad=0.0):
    """box: (x1, y1, x2, y2) floats -> int box grown by `pad` (fraction of box size per
    side) and clipped to an image of `shape` (H, W[, C])."""
    h0, w0 = shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, x2 = int(round(max(0.0, min(w0, x1 - px)))), int(round(max(0.0, min(w0, x2 + px))))
    y1, y2 = int(round(max(0.0, min(h0, y1 - py)))), int(round(max(0.0, min(h0, y2 + py))))
    return x1, y1, x2, y2


def crop(im0, box, pad=0.05):
    x1, y1, x2, y2 = clip_box(im0.shape, box, pad)
    return im0[y1:y2, x1:x2].copy()


def short_side(box):
    return min(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))


def box_center(box):
    return (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def sharpness(img_bgr):
    """Variance of the Laplacian, measured at a fixed scale so a big crop doesn't score
    sharper than a small one just for having more pixels. Low = blurry."""
    if img_bgr is None or img_bgr.size == 0:
        return 0.0
    h, w = img_bgr.shape[:2]
    scale = 128.0 / max(1, min(h, w))
    small = cv2.resize(img_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def resize_max(img_bgr, max_dim):
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return img_bgr


def encode_jpg(img_bgr, max_dim=1280, quality=85):
    ok, buf = cv2.imencode(".jpg", resize_max(img_bgr, max_dim), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("failed to encode image")
    return buf.tobytes()


def to_data_uri(img_bgr, max_dim=220, quality=85):
    if img_bgr is None or img_bgr.size == 0:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(encode_jpg(img_bgr, max_dim, quality)).decode("ascii")


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
