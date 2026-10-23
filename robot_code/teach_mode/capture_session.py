"""
Step 4, "collect multiple views": the user shows the object near the center mark (+) and
presses Capture once per viewpoint (front, side, back, ...). This module decides *which*
box in the frame is the object being taught, and turns a button press into a stored view.

Every frame, select_target() decides which box is the object, by similarity to the views
collected so far (the seed view from naming, plus earlier captures) and by continuity:
  * acquire   a box anywhere needs similarity >= acquire_sim. Measured on real taught
              objects (dinov2-small): at 0.55 ~2% of different objects get through, at
              0.4 ~14% -- a monitor behind the hand, a phone next to the cup.
  * follow    once acquired, a box that overlaps the last one well (IoU >= follow_iou,
              seen within follow_s) only needs min_sim -- turning the object shows a new
              side that may look less like the views so far. A big background box the
              object merely passes in front of doesn't overlap it well enough.
  * prefer    the followed box; otherwise the one nearest the center mark -- this is how
              the user points at one of two look-alikes.
The track starts where the object was when it was named (seed_box), so an object still
sitting there is followed from the first frame.
Only that box is drawn, so the user always sees what a press will grab. Every capture
widens the reference, so turning the object a bit at a time keeps it found.

If nothing in view looks like the object, a press saves nothing (no blind crop of whatever
is in the middle). Other quality problems (blurry, too small, duplicate of an earlier view)
come back as badges and the user decides whether to delete the shot.
"""
import math
import time

import numpy as np

from imgutils import box_center, crop, iou, normalize_rows, sharpness, short_side, to_data_uri

GUIDE_STEPS = ["front", "side", "back", "other side", "top", "angled"]
CENTER_TOL_FRAC = 0.04  # a box this close to the center mark (fraction of the short side) counts as on it


class CaptureSession:
    def __init__(self, object_id, name, seed_embs, seed_files=(), seed_box=None, now=0.0, min_sim=0.4,
                 acquire_sim=0.55, follow_iou=0.3, follow_s=1.0, match_sim=0.62, dup_thres=0.95,
                 blur_thres=35.0, min_side=40, pad=0.05):
        self.object_id = object_id
        self.name = name
        self.seed_embs = np.atleast_2d(normalize_rows(seed_embs))
        self.seed_files = list(seed_files)
        self.min_sim = min_sim          # to keep following the tracked box; below: not the object
        self.acquire_sim = acquire_sim  # to pick up a box that isn't the tracked one
        self.follow_iou = follow_iou
        self.follow_s = follow_s
        self.match_sim = match_sim      # above: memory would already recognize this view
        self.dup_thres = dup_thres  # above: same viewpoint as a view already collected
        self.blur_thres = blur_thres
        self.min_side = min_side
        self.pad = pad
        self.started_at = time.time()
        self.views = []  # [{idx, crop, emb, origin, sim, badges, thumb}]
        self._next_idx = 1
        self.target_box = None   # this frame's pick (None if not found)
        self.target_score = None
        self.target_in_frame = False
        self.track_box = None if seed_box is None else tuple(float(v) for v in seed_box)  # last place seen
        self.track_at = now
        self.last_center = None if seed_box is None else box_center(self.track_box)  # (used after Done)

    def reference_embs(self):
        if not self.views:
            return self.seed_embs
        return np.concatenate([self.seed_embs, np.stack([v["emb"] for v in self.views])])

    def similarity(self, emb):
        """Best cosine similarity of one embedding to the object's views so far."""
        ref = self.reference_embs()
        return float((ref @ normalize_rows(emb)).max()) if len(ref) else None

    def level(self, sim):
        """What a similarity means for the user: dup / match / new (angle) / different."""
        if sim is None:
            return None
        if sim >= self.dup_thres:
            return "dup"
        if sim >= self.match_sim:
            return "match"
        if sim >= self.min_sim:
            return "new"
        return "different"

    # -- per frame ---------------------------------------------------------------
    @staticmethod
    def center_mark(frame_shape):
        h, w = frame_shape[:2]
        return w / 2.0, h / 2.0

    def select_target(self, boxes, embs, frame_shape, now):
        """boxes: candidate boxes (non-person detections); embs: their embeddings.
        Returns the index of the box being taught, or None if nothing qualifies."""
        self.target_in_frame = False
        self.target_box = self.target_score = None
        ref = self.reference_embs()
        if not len(boxes) or not len(ref):
            return None
        sims = (normalize_rows(embs) @ ref.T).max(axis=1)
        following = self.track_box is not None and now - self.track_at <= self.follow_s
        cx, cy = self.center_mark(frame_shape)
        tol = CENTER_TOL_FRAC * min(frame_shape[:2])
        best, best_key = None, None
        for i, box in enumerate(boxes):
            sim = float(sims[i])
            followed = following and iou(box, self.track_box) >= self.follow_iou
            if sim < (self.min_sim if followed else self.acquire_sim):
                continue
            # distance from the mark to the box (0 if the box covers the mark, within the slack)
            x1, y1, x2, y2 = box
            d = max(0.0, math.hypot(max(x1 - cx, 0.0, cx - x2), max(y1 - cy, 0.0, cy - y2)) - tol)
            key = (not followed, d, -sim)
            if best is None or key < best_key:
                best, best_key = i, key
        if best is not None:
            self.target_box = self.track_box = tuple(float(v) for v in boxes[best])
            self.track_at = now
            self.target_score = float(sims[best])
            self.target_in_frame = True
            self.last_center = box_center(self.target_box)
        return best

    # -- button presses ------------------------------------------------------------
    def shoot(self, im0, embed_fn):
        if not self.target_in_frame:
            raise ValueError(f"no {self.name} in view -- nothing saved. Show it to the camera.")
        box, origin = self.target_box, "capture"
        shot = crop(im0, box, self.pad)
        emb = normalize_rows(embed_fn([shot]))[0]
        sim = self.similarity(emb)  # against the views *before* this one
        badges = []
        if short_side(box) < self.min_side:
            badges.append("small")
        if sharpness(shot) < self.blur_thres:
            badges.append("blurry")
        level = self.level(sim)
        if level == "dup":
            badges.append("duplicate")
        elif level == "different":
            badges.append("different?")
        view = {"idx": self._next_idx, "crop": shot, "emb": emb, "origin": origin, "sim": sim, "badges": badges,
                "thumb": to_data_uri(shot, max_dim=160)}
        self._next_idx += 1
        self.views.append(view)
        return self.public_view(view)

    def delete(self, idx):
        before = len(self.views)
        self.views = [v for v in self.views if v["idx"] != idx]
        return len(self.views) < before

    # -- UI --------------------------------------------------------------------------
    def guide_text(self):
        n = len(self.views)
        return f"View {n + 1}: show the {GUIDE_STEPS[n % len(GUIDE_STEPS)]}"

    @staticmethod
    def public_view(view):
        return {"idx": view["idx"], "thumb": view["thumb"], "badges": view["badges"], "origin": view["origin"],
                "sim": None if view["sim"] is None else round(view["sim"], 3)}

    def public(self):
        return {
            "object_id": self.object_id, "name": self.name,
            "target_found": self.target_in_frame,
            "target_score": None if self.target_score is None else round(self.target_score, 3),
            "target_level": self.level(self.target_score),
            "guide_text": self.guide_text(),
            "seed_views": ["/memory/" + f for f in self.seed_files],
            "views": [self.public_view(v) for v in self.views],
        }
