"""
Persistent, appearance-based memory of human-given names for OOD ("unknown") objects.

Unlike the spatial `cache`/`CACHE`/`CACHE_LIVE` OrderedDicts in robot_implementation.py /
robot_web_app.py (keyed by a quantized on-screen bounding-box position, so they only
recognize an object while it stays near the same spot within one process's lifetime),
this compares CLIP ViT-B/32 image embeddings via cosine similarity, so the same physical
object is recognized regardless of where it appears on screen, in a later frame, or in a
brand-new process -- it's persisted to a JSON file on disk.

CLIP is loaded lazily, only once a lookup()/remember() call actually needs it (and lookup()
skips loading it at all while the memory is still empty) -- same lazy-loading philosophy as
qwen3_generator's stage-2 verifier model.
"""
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

DEFAULT_SIM_THRESHOLD = 0.85  # see OODMemory docstring for rationale
_EMBED_DIM = 512  # CLIP ViT-B/32 image embedding dimensionality


class OODMemory:
    """
    sim_threshold: cosine-similarity cutoff (embeddings are L2-normalized, so cosine
    similarity == dot product) above which a stored name is considered a match for a new
    crop. An earlier measurement against static background crops (a cabinet edge, a blank
    wall corner) found different-object similarity around ~0.84-0.90 and same-object
    repeats around ~0.95-0.97, suggesting ~0.92 as a precision-favoring cutoff. In practice
    that was too strict for a held, actively-manipulated object (a lotion bottle being
    turned/rotated in hand): the detector's box jitters frame to frame even when the scene
    looks still, so the spatial cache misses often, and the resulting embeddings for the
    same physical object vary enough (angle, partial occlusion by fingers, motion blur)
    that some frames legitimately scored below 0.92 against the one stored view -- falling
    through to a fresh, independently-unstable VLM guess each time. 0.85 favors recall
    (fewer repeat VLM/human queries for the same object) at the cost of an occasional
    false-accept (reusing a name for a different-but-similar object). Tune via
    --memory-sim-thres based on what you observe; remembering the same object from a few
    different angles (just relabel it again when it's wrong) also helps independently of
    this threshold, since remember() appends a new embedding rather than replacing the old
    one.
    """

    def __init__(self, path, sim_threshold=DEFAULT_SIM_THRESHOLD, device=None):
        self.path = Path(path)
        self.sim_threshold = sim_threshold
        self.device = device  # None => auto-pick at first lazy load
        self._model = None
        self._preprocess = None
        self._lock = threading.Lock()  # guards _entries/_matrix and file I/O
        self._entries = []             # list[{"name","embedding","added_at"}]
        self._matrix = None            # cached np.ndarray (N, 512), rebuilt on remember()
        self._load()

    # -- persistence -----------------------------------------------------------
    def _load(self):
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[ood_memory] warning: could not read {self.path} ({e}); starting empty")
            return
        if not isinstance(raw, list):
            print(f"[ood_memory] warning: {self.path} is not a JSON list; starting empty")
            return
        entries = []
        for item in raw:
            if (isinstance(item, dict) and "name" in item and "embedding" in item
                    and len(item.get("embedding") or []) == _EMBED_DIM):
                entries.append(item)
            else:
                print(f"[ood_memory] warning: skipping malformed entry in {self.path}: {item!r}")
        self._entries = entries
        self._rebuild_matrix()
        print(f"[ood_memory] loaded {len(self._entries)} entries from {self.path}")

    def _save(self):
        # Atomic write (tmp file + os.replace) so a crash/kill mid-write never truncates or
        # corrupts the existing file -- this gets rewritten in full on every remember() call
        # over a long-running session.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + f".tmp{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, indent=4)
        os.replace(tmp, self.path)

    def _rebuild_matrix(self):
        self._matrix = (
            np.stack([np.asarray(e["embedding"], dtype=np.float32) for e in self._entries])
            if self._entries else None
        )

    # -- CLIP (lazy) -------------------------------------------------------------
    def _ensure_clip_loaded(self):
        if self._model is not None:
            return
        import clip  # already installed in the sam_oed env; see SAM-Dataset/tools.py for prior use
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ood_memory] loading CLIP ViT-B/32 on {device} ...")
        self._model, self._preprocess = clip.load("ViT-B/32", device=device)
        self._model.eval()
        self.device = device

    def _embed(self, crop) -> np.ndarray:
        """crop: PIL.Image (RGB) or BGR np.ndarray. Not lock-protected itself -- only ever
        called from within lookup()/remember(), which already hold self._lock."""
        self._ensure_clip_loaded()
        if isinstance(crop, np.ndarray):
            crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            tensor = self._preprocess(crop).unsqueeze(0).to(self.device)
            feat = self._model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).float().cpu().numpy()

    # -- public API ----------------------------------------------------------
    def lookup(self, crop) -> Optional[str]:
        with self._lock:
            if not self._entries:
                return None  # skip loading CLIP entirely until something's been remembered
            query = self._embed(crop)
            sims = self._matrix @ query
            best = int(np.argmax(sims))
            best_sim = float(sims[best])
            best_name = self._entries[best]["name"]
        if best_sim >= self.sim_threshold:
            print(f"[ood_memory] matched '{best_name}' (sim={best_sim:.3f} >= {self.sim_threshold})")
            return best_name
        return None

    def remember(self, crop, name) -> None:
        name = (name or "").strip().lower()
        if not name:
            return
        with self._lock:
            embedding = self._embed(crop)
            self._entries.append({
                "name": name,
                "embedding": embedding.tolist(),
                "added_at": datetime.now(timezone.utc).isoformat(),
            })
            self._rebuild_matrix()
            self._save()
        print(f"[ood_memory] remembered '{name}' ({len(self._entries)} total entries)")

    def list_entries(self):
        """Returns [{"index", "name", "added_at"}, ...] -- no embeddings, for display/CLI use."""
        with self._lock:
            return [
                {"index": i, "name": e["name"], "added_at": e["added_at"]}
                for i, e in enumerate(self._entries)
            ]

    def forget(self, indices) -> int:
        """Removes entries at the given 0-based indices (as returned by list_entries()).
        Returns how many were actually removed."""
        with self._lock:
            drop = set(indices)
            kept = [e for i, e in enumerate(self._entries) if i not in drop]
            removed = len(self._entries) - len(kept)
            if removed:
                self._entries = kept
                self._rebuild_matrix()
                self._save()
        return removed
