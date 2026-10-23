"""
Appearance embeddings for object crops, used to recognize the *same physical object* again
(instance-level), not just the same category.

Default is DINOv2 ViT-S/14 (self-supervised, much better at telling two different objects
of the same kind apart than CLIP, which is trained to match captions and so maps every
remote control close to every other one). CLIP ViT-B/32 stays available for comparison
via calibrate.py.

Crops are padded to a square before resizing instead of center-cropped: a standard
resize-then-center-crop would cut both ends off an elongated object like a remote.
"""
import cv2
import numpy as np
import torch

EMBEDDERS = {
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
    "clip-b32": "clip:ViT-B/32",
}
DEFAULT_EMBEDDER = "dinov2-small"

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_PAD_BGR = tuple(int(round(v * 255)) for v in _IMAGENET_MEAN[::-1])  # pads to ~0 after normalization


def square_pad(img_bgr, fill=_PAD_BGR):
    h, w = img_bgr.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(img_bgr, top, side - h - top, left, side - w - left,
                              cv2.BORDER_CONSTANT, value=fill)


class Embedder:
    """embed(list of BGR uint8 crops) -> float32 array (M, dim), rows L2-normalized.
    `model_id` is stored alongside saved embeddings, so memory can tell when it was
    built with a different model and needs re-embedding."""

    def __init__(self, name=DEFAULT_EMBEDDER, device=None, input_size=224):
        if name not in EMBEDDERS:
            raise ValueError(f"unknown embedder {name!r}; choose from {sorted(EMBEDDERS)}")
        self.name = name
        self.model_id = EMBEDDERS[name]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        print(f"[embedder] loading {self.model_id} on {self.device} ...")
        if self.model_id.startswith("clip:"):
            import clip
            self._model, self._clip_preprocess = clip.load(self.model_id.split(":", 1)[1], device=self.device)
            self.dim = int(self._model.visual.output_dim)
        else:
            from transformers import AutoModel
            self._model = AutoModel.from_pretrained(self.model_id).to(self.device)
            self.dim = int(self._model.config.hidden_size)
        self._model.eval()

    def _to_tensor(self, crop_bgr):
        img = cv2.resize(square_pad(crop_bgr), (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        return torch.from_numpy(rgb.transpose(2, 0, 1))

    @torch.no_grad()
    def embed(self, crops):
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.model_id.startswith("clip:"):
            from PIL import Image
            batch = torch.stack([
                self._clip_preprocess(Image.fromarray(cv2.cvtColor(square_pad(c), cv2.COLOR_BGR2RGB)))
                for c in crops
            ]).to(self.device)
            feats = self._model.encode_image(batch).float()
        else:
            batch = torch.stack([self._to_tensor(c) for c in crops]).to(self.device)
            feats = self._model(pixel_values=batch).pooler_output.float()  # layer-normed CLS token
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.cpu().numpy().astype(np.float32)
