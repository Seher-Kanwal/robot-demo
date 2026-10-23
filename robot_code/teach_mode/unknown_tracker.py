"""
Turns per-frame OOD detections into stable "unknown object" identities, so the robot asks
about each physical object once instead of once per frame.

The detector has no notion of identity and its boxes jitter, so a detection is linked to an
existing item if it overlaps the item's last box (IoU) *or* looks like it (embedding
similarity) -- the second rule keeps the identity when the object is picked up and moved.

An item only becomes a question after it has been seen in several frames within a short
window (filters one-frame detector flicker), and questions are asked one at a time, FIFO.
Skipping an item puts its appearance on an ignore list, so it isn't asked about again
this session.
"""
from collections import OrderedDict, deque
from itertools import count

import numpy as np

from imgutils import iou, normalize_rows, sharpness, short_side

TRACKING, QUEUED, ASKING = "tracking", "queued", "asking"


class UnknownTracker:
    def __init__(self, same_thres=0.72, iou_thres=0.3, min_hits=3, promote_window_s=2.0, expire_s=10.0,
                 ask_visible_s=1.5, ignore_thres=0.75, keep_crops=3):
        self.same_thres = same_thres
        self.iou_thres = iou_thres
        self.min_hits = min_hits
        self.promote_window_s = promote_window_s
        self.expire_s = expire_s
        self.ask_visible_s = ask_visible_s  # only start asking about something seen this recently
        self.ignore_thres = ignore_thres
        self.keep_crops = keep_crops
        self.paused = False  # no new questions while the capture window is open
        self.items = OrderedDict()  # id -> item dict, creation order == FIFO order
        self._ids = count(1)
        self._ignored = []  # embeddings of skipped objects
        self._new_question = None

    # -- per frame ---------------------------------------------------------------
    def update(self, boxes, embs, crops, now):
        """boxes: list of (x1,y1,x2,y2); embs: (N, dim); crops: list of BGR arrays.
        Returns one (item_id, state) per detection; state is tracking/queued/asking, or
        (None, "ignored") for a skipped object."""
        embs = normalize_rows(embs) if len(boxes) else np.zeros((0, 1), np.float32)
        out = [None] * len(boxes)

        live = []
        ignored = np.stack(self._ignored) if self._ignored else None
        for d in range(len(boxes)):
            if ignored is not None and float((ignored @ embs[d]).max()) >= self.ignore_thres:
                out[d] = (None, "ignored")
            else:
                live.append(d)

        # Greedy one-to-one association, strongest (similarity + overlap) pairs first.
        pairs = []
        for d in live:
            for item_id, item in self.items.items():
                sim = float(item["emb"] @ embs[d])
                ov = iou(boxes[d], item["box"])
                if ov >= self.iou_thres or sim >= self.same_thres:
                    pairs.append((sim + ov, d, item_id))
        pairs.sort(key=lambda p: -p[0])
        used_d, used_i = set(), set()
        for _, d, item_id in pairs:
            if d in used_d or item_id in used_i:
                continue
            used_d.add(d)
            used_i.add(item_id)
            self._observe(self.items[item_id], boxes[d], embs[d], crops[d], now)
            out[d] = (item_id, None)
        for d in live:
            if d not in used_d:
                item_id = next(self._ids)
                self.items[item_id] = {
                    "id": item_id, "box": tuple(boxes[d]), "emb": embs[d], "best": [], "hits": deque(),
                    "first_seen": now, "last_seen": now, "state": TRACKING,
                }
                self._observe(self.items[item_id], boxes[d], embs[d], crops[d], now)
                out[d] = (item_id, None)

        for item in list(self.items.values()):
            while item["hits"] and now - item["hits"][0] > self.promote_window_s:
                item["hits"].popleft()
            if item["state"] == TRACKING and len(item["hits"]) >= self.min_hits:
                item["state"] = QUEUED
            if item["state"] != ASKING and now - item["last_seen"] > self.expire_s:
                del self.items[item["id"]]

        if not self.paused and self.asking() is None:
            for item in self.items.values():
                if item["state"] == QUEUED and now - item["last_seen"] <= self.ask_visible_s:
                    item["state"] = ASKING
                    self._new_question = item["id"]
                    break

        return [(o[0], self.items[o[0]]["state"]) if o[0] is not None else o for o in out]

    def _observe(self, item, box, emb, crop, now):
        item["box"] = tuple(box)
        item["emb"] = normalize_rows(0.7 * item["emb"] + 0.3 * emb)
        item["last_seen"] = now
        item["hits"].append(now)
        quality = min(sharpness(crop), 400.0) * min(1.0, short_side(box) / 80.0)
        item["best"].append((quality, crop, emb))
        item["best"].sort(key=lambda b: -b[0])
        del item["best"][self.keep_crops:]

    # -- questions -----------------------------------------------------------------
    def asking(self):
        for item in self.items.values():
            if item["state"] == ASKING:
                return item
        return None

    def pop_new_question(self):
        """The item that just became the current question this frame (once), else None."""
        item_id, self._new_question = self._new_question, None
        return item_id

    def queue_len(self):
        return sum(1 for i in self.items.values() if i["state"] == QUEUED)

    @staticmethod
    def seed(item):
        """(crop, embedding) of the item's best-quality sighting -- becomes view 0."""
        _, crop, emb = item["best"][0]
        return crop, emb

    def take(self, item_id):
        """Removes and returns an item (it was answered), or None if it no longer exists."""
        if self._new_question == item_id:
            self._new_question = None
        return self.items.pop(item_id, None)

    def skip(self, item_id):
        item = self.take(item_id)
        if item is not None:
            self._ignored.append(item["emb"])
            self._ignored.extend(b[2] for b in item["best"])
        return item

    def purge_similar(self, emb, thres=None):
        """Drops other items that look like `emb` -- after one is answered, a duplicate
        identity of the same object must not get asked about too."""
        thres = self.same_thres if thres is None else thres
        emb = normalize_rows(emb)
        for item_id in [i for i, it in self.items.items() if float(it["emb"] @ emb) >= thres]:
            del self.items[item_id]

    def ignore(self, emb):
        """Adds an appearance to the skip list (the user skipped this object)."""
        self._ignored.append(normalize_rows(emb))

    def ignored_mask(self, embs):
        """One bool per row of `embs`: does it look like something the user skipped?"""
        embs = normalize_rows(embs)
        if not self._ignored or not len(embs):
            return [False] * len(embs)
        return list((embs @ np.stack(self._ignored).T).max(axis=1) >= self.ignore_thres)

    def reset_ignored(self):
        self._ignored.clear()

    def clear(self):
        """Forgets every pending/queued question (the skip list stays)."""
        self.items.clear()
        self._new_question = None
