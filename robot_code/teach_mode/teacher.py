"""
Per-frame logic of Module 1 ("Ask & Remember"), independent of Flask and of the detector:
teach_app.py runs the detector and hands each frame's boxes here.

    detections -> embed the OOD crops once
               -> capture window open?  find the object being taught, take requested shots   [step 4]
               -> memory match (every frame, before anything else -- no positional cache)     [step 7]
               -> what's left is unknown                                                      [step 3]
               -> draw the "OOD detector" and "Robot view" images, publish voice events

Two modes, switchable at runtime (set_mode):
  teach    Once an unknown object has been seen steadily (UnknownTracker decides that), the
           robot *stops on that image*: every unknown object in it becomes a numbered item,
           and the robot asks about them one at a time, staying on that image until every
           item is named (or skipped). Only then does it open the capture window (live) for
           each named object in turn, and after the last one it goes live again and looks
           for the next unknown object.
  explore  after teaching, look around, live, with no interaction: the robot view shows known
           VOC boxes plus only those OOD boxes the detector is very sure of (score >=
           explore_ood_conf) *and* memory can name. Every other OOD box is dropped.

Thread model: process() runs on the camera thread and is the only place models run.
The Flask handlers call answer()/skip()/request_shot()/... from request threads; they
only change state under self.lock. A Capture press is queued and fulfilled by the next
process() call, which owns the current frame.
"""
import math
import threading
import time
from collections import deque
from itertools import count
from pathlib import Path

import cv2
import numpy as np

from capture_session import CENTER_TOL_FRAC, CaptureSession
from imgutils import box_center, crop, encode_jpg, short_side, to_data_uri
from naming import normalize_name

# BGR colors. Known/unknown match robot_implementation.KNOWN_COLOR / OOD_COLOR.
KNOWN_COLOR = (255, 0, 0)
OOD_COLOR = (0, 0, 255)
MEMORY_COLOR = (60, 190, 60)
ASK_COLOR = (0, 210, 255)
CAPTURE_COLOR = (220, 80, 170)
IGNORED_COLOR = (140, 140, 140)
GUIDE_COLOR = (255, 255, 255)

STEP_CAMERA, STEP_OOD, STEP_ASK, STEP_VIEWS, STEP_MEMORY, STEP_MOVE, STEP_RECOGNIZE = range(1, 8)
MODES = ("teach", "explore")


class _ShotRequest:
    def __init__(self):
        self.event = threading.Event()
        self.result = None
        self.error = None


class Teacher:
    def __init__(self, embedder, memory, tracker, voc_names, ood_class_id, person_class_id=0,
                 negatives_dir=None, min_side=24, crop_pad=0.05, match_known_boxes=False, known_extra_thres=0.05,
                 capture_min_sim=0.4, capture_acquire_sim=0.55, recog_streak=2, reappear_s=3.0, move_frac=0.2, still_s=0.6,
                 still_frac=0.03, announce_cooldown_s=4.0, step_hold_s=3.5, show_scores=False,
                 mode="teach", explore_ood_conf=0.95, freeze_cooldown_s=1.5):
        self.embedder = embedder
        self.memory = memory
        self.tracker = tracker
        self.voc_names = voc_names
        self.ood_class_id = ood_class_id
        self.person_class_id = person_class_id
        self.negatives_dir = Path(negatives_dir) if negatives_dir else None
        self.min_side = min_side
        self.crop_pad = crop_pad
        self.match_known_boxes = match_known_boxes
        self.known_extra_thres = known_extra_thres
        self.capture_min_sim = capture_min_sim
        self.capture_acquire_sim = capture_acquire_sim
        self.recog_streak = recog_streak
        self.reappear_s = reappear_s
        self.move_frac = move_frac
        self.still_s = still_s
        self.still_frac = still_frac
        self.announce_cooldown_s = announce_cooldown_s
        self.step_hold_s = step_hold_s
        self.show_scores = show_scores
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.explore_ood_conf = explore_ood_conf
        self.freeze_cooldown_s = freeze_cooldown_s  # after going live again, before stopping on a new image

        self.lock = threading.RLock()
        self.capture = None
        self.frozen = None       # the image the robot stopped on (teach mode), see _freeze()
        self._freeze_cooldown_until = 0.0
        self._item_ids = count(1)
        self.events = deque(maxlen=100)
        self._seq = 0
        self._shots = []
        self._rec = {}           # object_id -> recognition/announcement bookkeeping
        self._step_hold = None   # (step, until) -- briefly show step 5 / 7 on the tracker
        self._awaiting_move = None  # object just memorized; step 6 until it's recognized again
        self._ask_thumb = (None, None)
        self.frame_count = 0
        self.fps = 0.0
        self._frame = {"known_labels": [], "recognized": [], "width": None}

    # ===================================================================== per frame
    def process(self, im0, boxes, labels, scores, now=None):
        """boxes (N,4) float xyxy in im0 pixels, labels (N,) int, scores (N,) float.
        Returns (robot_view_bgr, ood_view_bgr) -- while stopped on an image, those are that
        image's views, not this frame's (unless the capture window needs the live picture)."""
        now = time.monotonic() if now is None else now
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(labels).astype(int).reshape(-1)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        h, w = im0.shape[:2]
        with self.lock:
            capture = self.capture
            explore = self.mode == "explore"
            big = [i for i in range(len(boxes)) if short_side(boxes[i]) >= self.min_side]
            ood_rows = [i for i in big if labels[i] == self.ood_class_id]
            if explore:  # only OOD boxes the detector is very sure of are even considered
                ood_rows = [i for i in ood_rows if scores[i] >= self.explore_ood_conf]
            known_rows = [i for i in big if labels[i] not in (self.ood_class_id, self.person_class_id)]
            extra_rows = known_rows if (capture is not None or self.match_known_boxes) else []
            cand_rows = ood_rows + extra_rows
            crops = {r: crop(im0, boxes[r], self.crop_pad) for r in cand_rows}
            embs_list = self.embedder.embed([crops[r] for r in cand_rows])
            embs = {r: embs_list[k] for k, r in enumerate(cand_rows)}
            states = {}  # row -> (kind, text)

            # -- step 4: the object being taught (tracked, or a strong look-alike nearest the mark) + Capture presses
            target_row = None
            if capture is not None:
                k = capture.select_target([boxes[r] for r in cand_rows], embs_list, im0.shape, now)
                if k is not None:
                    target_row = cand_rows[k]
                for req in self._shots:
                    try:
                        req.result = capture.shoot(im0, self.embedder.embed)
                        n = len(capture.views)
                        self._emit("capture_view", f"Got view {n}.", n=n)
                    except Exception as e:  # never let one bad shot kill the camera loop
                        req.error = str(e)
                    req.event.set()
                self._shots.clear()

            # -- step 7: memory first, for every OOD box (and known boxes if enabled)
            match_rows = [r for r in ood_rows if r != target_row]
            if self.match_known_boxes:
                match_rows += [r for r in known_rows if r != target_row]
            matches = self.memory.match(np.stack([embs[r] for r in match_rows])) if match_rows else []
            matched = {}  # object_id -> (name, score, box)
            unknown_rows, closest = [], {}
            for r, m in zip(match_rows, matches):
                is_known = labels[r] != self.ood_class_id
                ok = m is not None and m["accepted"] and (
                    not is_known or m["score"] >= self.memory.accept_thres + self.known_extra_thres)
                if ok:
                    states[r] = ("memory", f"memory: {m['name']} {m['score']:.2f}")
                    if m["object_id"] not in matched or m["score"] > matched[m["object_id"]][1]:
                        matched[m["object_id"]] = (m["name"], m["score"], tuple(boxes[r]))
                elif not is_known and not explore:  # explore: unnamed OOD boxes are simply dropped
                    unknown_rows.append(r)
                    if m is not None:
                        closest[r] = m
            # No "I recognize ..." while stopped on an image: the user is busy naming things.
            self._update_recognitions(matched, now, w, announce=self.frozen is None)

            # -- step 3: unknown objects. The tracker only decides *when* an unknown object is
            # really there (seen steadily, not one-frame detector flicker); then the robot stops
            # on this image and asks about every unknown object in it.
            ignored = self.tracker.ignored_mask(np.stack([embs[r] for r in unknown_rows])) if unknown_rows else []
            new_rows = []
            for r, is_ignored in zip(unknown_rows, ignored):
                if is_ignored:
                    states[r] = ("ignored", "ignored")
                    continue
                new_rows.append(r)
                hint = ""
                if self.show_scores and r in closest:
                    hint = f" ({closest[r]['name']} {closest[r]['score']:.2f})"
                states[r] = ("unknown", "unknown" + hint)
            if (not explore and self.frozen is None and capture is None
                    and now >= self._freeze_cooldown_until):
                self.tracker.update([tuple(boxes[r]) for r in new_rows],
                                    np.stack([embs[r] for r in new_rows]) if new_rows else [],
                                    [crops[r] for r in new_rows], now)
                if self.tracker.pop_new_question() is not None and new_rows:
                    self._freeze(im0, boxes, labels, states, new_rows, crops, embs, now)

            self.frame_count += 1
            self._frame = {
                "width": w,
                "known_labels": sorted({self.voc_names.get(int(labels[i]), "") for i in known_rows} - {""}),
                "recognized": [{"object_id": oid, "name": n, "score": round(s, 3)}
                               for oid, (n, s, _b) in sorted(matched.items(), key=lambda kv: -kv[1][1])],
            }
            if self.frozen is not None and capture is None:
                return self._draw_frozen(), self.frozen["ood_img"]
            if capture is not None:
                robot = self._draw_capture(im0, capture)
            else:
                robot = self._draw_robot(im0, boxes, labels, states, hide_unnamed_ood=explore)
            return robot, self._draw_ood(im0, boxes, labels)

    def _update_recognitions(self, matched, now, width, announce=True):
        """Announces "I recognize this object from memory" once per *appearance*: the first
        time an object is recognized, when it comes back after being out of sight, or when
        it has been moved and has come to rest somewhere else. Never every frame. With
        announce=False the bookkeeping still runs, so nothing is "new" again afterwards."""
        for oid, (name, _score, box) in matched.items():
            st = self._rec.setdefault(oid, {"streak": 0, "last_seen": None, "absent_for": math.inf,
                                            "announced_at": None, "announced_center": None, "track": deque()})
            gap = math.inf if st["last_seen"] is None else now - st["last_seen"]
            if gap > 1.0:
                st["streak"], st["absent_for"] = 1, gap
                st["track"].clear()
            else:
                st["streak"] += 1
            st["last_seen"] = now
            c = box_center(box)
            st["track"].append((now, c))
            while st["track"] and now - st["track"][0][0] > self.still_s:
                st["track"].popleft()
            if not announce or st["streak"] < self.recog_streak:
                continue
            span = now - st["track"][0][0]
            still = span >= self.still_s * 0.75 and all(
                math.hypot(p[0] - c[0], p[1] - c[1]) <= self.still_frac * width for _, p in st["track"])
            if not still:
                continue  # wait until it's put down, not while it's being carried
            if st["announced_at"] is None:
                reason = True
            elif st["absent_for"] >= self.reappear_s:
                reason = True
            else:
                ac = st["announced_center"]
                reason = ac is None or math.hypot(c[0] - ac[0], c[1] - ac[1]) > self.move_frac * width
            if reason and (st["announced_at"] is None or now - st["announced_at"] >= self.announce_cooldown_s):
                self._emit("recognized", f"I recognize this object from memory. It is the {name}.",
                           object_id=oid, name=name)
                st["announced_at"], st["announced_center"], st["absent_for"] = now, c, 0.0
                self._step_hold = (STEP_RECOGNIZE, now + self.step_hold_s)
                if self._awaiting_move == oid:
                    self._awaiting_move = None

    # ===================================================================== stop-on-image (teach)
    def _freeze(self, im0, boxes, labels, states, rows, crops, embs, now):
        """Stops on this image: every unknown object in it becomes a numbered item, left to right."""
        rows = sorted(rows, key=lambda r: box_center(boxes[r])[0])
        items = [{"id": next(self._item_ids), "row": r, "box": tuple(float(v) for v in boxes[r]),
                  "crop": crops[r], "emb": np.array(embs[r], dtype=np.float32), "state": "pending", "name": None}
                 for r in rows]
        self.frozen = {
            "frame": im0.copy(), "boxes": boxes.copy(), "labels": labels.copy(), "items": items, "at": now,
            "base_states": {r: s for r, s in states.items() if r not in rows},  # known/memory/ignored boxes
            "ood_img": self._draw_ood(im0, boxes, labels),
        }
        self.tracker.clear()
        n = len(items)
        self._emit("ask", "I found an unfamiliar object. What is this called?" if n == 1 else
                   f"I found {n} unfamiliar objects. Let's go one by one. What is this one called?",
                   index=1, total=n)

    def _current_item(self):
        """The item being asked about right now (None while the capture window is open)."""
        if self.frozen is None or self.capture is not None:
            return None
        return next((it for it in self.frozen["items"] if it["state"] == "pending"), None)

    def _frozen_item(self, item_id):
        if self.frozen is None:
            return None
        return next((it for it in self.frozen["items"] if it["id"] == item_id), None)

    def _advance(self, now, speak_resume=True, thanks=False):
        """After an item is named or skipped, or a capture window closes: ask about the next
        item on the stopped image; once all are named/skipped, open the capture window for
        the next named object that has no views yet; when none are left, go live again."""
        fz = self.frozen
        if fz is None or self.capture is not None:
            return
        items = fz["items"]
        pending = [it for it in items if it["state"] == "pending"]
        if pending:
            self._emit("ask", "What is this one called?", index=items.index(pending[0]) + 1, total=len(items))
            return
        uncaptured = next((it for it in items if it["state"] == "named" and not it["captured"]), None)
        if uncaptured is not None:
            uncaptured["captured"] = True  # done even if the window is cancelled
            self._open_capture(uncaptured["object_id"], seed_box=uncaptured["box"], now=now)
            self._emit("thanks" if thanks else "capture_start",
                       ("Thanks! " if thanks else "Now ") + f"Please show me the {uncaptured['name']} near the "
                       f"center mark, turn it to different sides, and press Capture for each view.",
                       object_id=uncaptured["object_id"], name=uncaptured["name"])
            return
        self.frozen = None
        self.tracker.clear()
        self._freeze_cooldown_until = now + self.freeze_cooldown_s
        if speak_resume:
            self._emit("resume", "Okay. I will keep looking around.")

    def _draw_frozen(self):
        fz = self.frozen
        states = dict(fz["base_states"])
        current = self._current_item()
        for k, it in enumerate(fz["items"], 1):
            if it["state"] == "named":
                states[it["row"]] = ("memory", f"{k}: {it['name']}")
            elif it["state"] == "skipped":
                states[it["row"]] = ("ignored", f"{k}: skipped")
            elif it is current:
                states[it["row"]] = ("asking", f"{k}: ? what is this")
            else:
                states[it["row"]] = ("unknown", f"{k}: ?")
        img = self._draw_robot(fz["frame"], fz["boxes"], fz["labels"], states)
        done = sum(1 for it in fz["items"] if it["state"] != "pending")
        _banner(img, f"PAUSED - name each object ({done}/{len(fz['items'])} done)")
        return img

    # ===================================================================== drawing
    def _draw_ood(self, im0, boxes, labels):
        img = np.ascontiguousarray(im0.copy())
        for i in range(len(boxes)):
            cls = int(labels[i])
            if cls == self.ood_class_id:
                draw_box(img, boxes[i], "Unknown (OOD)", OOD_COLOR)
            else:
                draw_box(img, boxes[i], f"{self.voc_names.get(cls, cls)} (VOC)", KNOWN_COLOR)
        return img

    def _draw_robot(self, im0, boxes, labels, states, hide_unnamed_ood=False):
        img = np.ascontiguousarray(im0.copy())
        colors = {"memory": MEMORY_COLOR, "asking": ASK_COLOR, "ignored": IGNORED_COLOR, "unknown": OOD_COLOR}
        for i in range(len(boxes)):
            kind, text = states.get(i, (None, None))
            if kind is None:
                cls = int(labels[i])
                if cls == self.ood_class_id:
                    if hide_unnamed_ood:
                        continue
                    kind, text = "unknown", "unknown"
                else:
                    draw_box(img, boxes[i], str(self.voc_names.get(cls, cls)), KNOWN_COLOR)
                    continue
            thick = 5 if kind == "asking" else None
            draw_box(img, boxes[i], text, colors[kind], thickness=thick)
        return img

    def _draw_capture(self, im0, capture):
        """Capture window: the center mark, and only the box being taught, labeled with its
        similarity to the object being saved."""
        img = np.ascontiguousarray(im0.copy())
        if capture.target_in_frame:
            draw_box(img, capture.target_box, f"{capture.name} {capture.target_score:.2f}", CAPTURE_COLOR, thickness=5)
        else:
            _banner(img, f"no {capture.name} in view - show it near the +")
        _draw_center_mark(img, capture.center_mark(im0.shape))
        return img

    # ===================================================================== events/state
    def _emit(self, kind, text, **extra):
        self._seq += 1
        self.events.append({"seq": self._seq, "type": kind, "text": text, "at": time.time(), **extra})

    def current_step(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.frame_count == 0:
                return STEP_CAMERA
            if self.mode == "explore":
                return STEP_RECOGNIZE if self._frame["recognized"] else STEP_OOD
            if self.capture is not None:
                return STEP_VIEWS
            hold = self._step_hold
            if hold and now < hold[1] and hold[0] == STEP_MEMORY:
                return STEP_MEMORY
            if self.frozen is not None:
                return STEP_ASK
            if hold and now < hold[1]:
                return hold[0]
            if self._awaiting_move is not None:
                return STEP_MOVE
            return STEP_OOD

    def snapshot(self, since=0, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            item = self._current_item()
            ask = None
            frozen = None
            if self.frozen is not None:
                items = self.frozen["items"]
                frozen = {"total": len(items),
                          **{s: sum(1 for it in items if it["state"] == s) for s in ("pending", "named", "skipped")}}
            if item is not None:
                if self._ask_thumb[0] is not item["crop"]:
                    self._ask_thumb = (item["crop"], to_data_uri(item["crop"], max_dim=360, quality=90))
                ask = {"item_id": item["id"], "crop": self._ask_thumb[1], "visible": True,
                       "index": self.frozen["items"].index(item) + 1, "total": len(self.frozen["items"])}
            return {
                "mode": self.mode,
                "explore_ood_conf": self.explore_ood_conf,
                "frame_count": self.frame_count,
                "fps": round(self.fps, 1),
                "current_step": self.current_step(now),
                "known_labels": self._frame["known_labels"],
                "recognized": self._frame["recognized"],
                "asking": ask,
                "frozen": frozen,
                "queue_len": max(0, (frozen["pending"] - 1) if frozen and item is not None else 0),
                "capture": None if self.capture is None else self.capture.public(),
                "memory_version": self.memory.version,
                "events": [e for e in self.events if e["seq"] > since],
                "last_seq": self._seq,
            }

    # ===================================================================== commands (request threads)
    def set_mode(self, mode):
        """teach <-> explore. A stopped image and pending questions are dropped either way, and
        entering explore forgets which objects were already greeted, so exploring announces
        each remembered object again the first time it comes into view."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self.lock:
            if mode == self.mode:
                return mode
            if self.capture is not None:
                raise ValueError("finish or cancel the capture window first")
            self.mode = mode
            self.frozen = None
            self.tracker.clear()
            self._step_hold = None
            if mode == "explore":
                self._awaiting_move = None
                self._rec.clear()
                self._emit("mode", "Explore mode. I will tell you which objects I remember.", mode=mode)
            else:
                self._emit("mode", "Teach mode. I will ask about objects I do not know.", mode=mode)
            return mode

    def answer(self, item_id, raw_name, now=None):
        """Step 3 -> 5: store the object's crop from the stopped image as its first view, then
        ask about the next object on the image. The capture window (step 4) opens only once
        every object on it is named or skipped. Raises ValueError (bad name / capture already
        open) or KeyError (no such pending object)."""
        now = time.monotonic() if now is None else now
        name = normalize_name(raw_name)
        if not name:
            raise ValueError("didn't catch a name -- please say or type it again")
        with self.lock:
            if self.capture is not None:
                raise ValueError("finish the current capture first")
            item = self._frozen_item(item_id)
            if item is None or item["state"] != "pending":
                raise KeyError(item_id)
            object_id = self.memory.add_object(name, [item["crop"]], item["emb"][None], origin="ask")
            item.update(state="named", name=name, object_id=object_id, captured=False)
            if any(it["state"] == "pending" for it in self.frozen["items"]):
                self._emit("named", f"Thanks, the {name}.", object_id=object_id, name=name)
            self._advance(now, thanks=True)
            return {"object_id": object_id, "name": name}

    def skip(self, item_id, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            item = self._frozen_item(item_id)
            if item is None or item["state"] != "pending":
                return False
            self._skip_item(item)
            self._advance(now)
            return True

    def skip_rest(self, now=None):
        """Skips every object still waiting on the stopped image and goes live again."""
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.frozen is None or self.capture is not None:
                return 0
            pending = [it for it in self.frozen["items"] if it["state"] == "pending"]
            for it in pending:
                self._skip_item(it)
            self._advance(now)
            return len(pending)

    def _skip_item(self, item):
        item["state"] = "skipped"
        self.tracker.ignore(item["emb"])
        if self.negatives_dir is not None:
            # Skipped objects are exactly the "please don't match me" crops calibrate.py wants.
            self.negatives_dir.mkdir(parents=True, exist_ok=True)
            path = self.negatives_dir / f"neg_{int(time.time() * 1000)}_{item['id']}.jpg"
            path.write_bytes(encode_jpg(item["crop"], max_dim=256, quality=90))

    def _open_capture(self, object_id, seed_box=None, now=0.0):
        """seed_box: where the object was on the stopped image -- tracking starts there."""
        obj = self.memory.get(object_id)
        self.capture = CaptureSession(object_id, obj["name"], self.memory.embeddings_for(object_id),
                                      seed_files=[v["file"] for v in obj["views"]], seed_box=seed_box, now=now,
                                      min_sim=self.capture_min_sim, acquire_sim=self.capture_acquire_sim,
                                      match_sim=self.memory.accept_thres)

    def start_capture(self, object_id):
        with self.lock:
            if self.mode != "teach":
                raise ValueError("switch to Teach mode to add views")
            if self.capture is not None:
                raise ValueError("a capture window is already open")
            if self.memory.get(object_id) is None:
                raise KeyError(object_id)
            self._open_capture(object_id)
            self._emit("capture_start", f"Show me the {self.capture.name} near the center mark, and press Capture.",
                       object_id=object_id)

    def request_shot(self, timeout=2.0):
        """Queues one Capture press for the camera thread and waits for it."""
        req = _ShotRequest()
        with self.lock:
            if self.capture is None:
                raise ValueError("no capture window is open")
            self._shots.append(req)
        if not req.event.wait(timeout):
            with self.lock:
                if req in self._shots:
                    self._shots.remove(req)
            raise TimeoutError("no camera frame arrived -- is the camera running?")
        if req.error:
            raise RuntimeError(req.error)
        return req.result

    def delete_shot(self, idx):
        with self.lock:
            return self.capture is not None and self.capture.delete(int(idx))

    def _close_capture(self):
        self.capture = None
        for req in self._shots:
            req.error = "capture window closed"
            req.event.set()
        self._shots.clear()

    def finish_capture(self, now=None):
        """Step 5: commit the captured views to memory, then on to the next object on the
        stopped image (or back to live if that was the last one)."""
        now = time.monotonic() if now is None else now
        with self.lock:
            cap = self.capture
            if cap is None:
                raise ValueError("no capture window is open")
            if cap.views:
                self.memory.add_views(cap.object_id, [v["crop"] for v in cap.views],
                                      np.stack([v["emb"] for v in cap.views]), [v["origin"] for v in cap.views])
            obj = self.memory.get(cap.object_id)
            total = obj["n_views"] if obj else len(cap.views)
            self._close_capture()
            # It's still in the user's hand right now -- don't greet it as "recognized" until
            # it has actually been put somewhere (moved away from here) or has been out of sight.
            self._rec[cap.object_id] = {"streak": 0, "last_seen": now, "absent_for": 0.0, "announced_at": now,
                                        "announced_center": cap.last_center, "track": deque()}
            self._awaiting_move = cap.object_id
            self._step_hold = (STEP_MEMORY, now + self.step_hold_s)
            more = self.frozen is not None and any(
                it["state"] == "pending" or (it["state"] == "named" and not it["captured"])
                for it in self.frozen["items"])
            self._emit("memorized", f"Got it. I've memorized the {cap.name} from {total} views."
                       + ("" if more else " Move it somewhere else, and I will find it."),
                       object_id=cap.object_id, name=cap.name, n_views=total)
            self._advance(now, speak_resume=False)
            return {"object_id": cap.object_id, "n_views": total}

    def cancel_capture(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.capture is None:
                return False
            self._close_capture()
            self._advance(now, speak_resume=False)
            return True

    def rename_object(self, object_id, name):
        name = normalize_name(name) or (name or "").strip()
        with self.lock:
            ok = self.memory.rename(object_id, name)
            if ok and self.capture is not None and self.capture.object_id == object_id:
                self.capture.name = name
            return ok

    def delete_object(self, object_id):
        with self.lock:
            if self.capture is not None and self.capture.object_id == object_id:
                self.cancel_capture()
            self._rec.pop(object_id, None)
            if self._awaiting_move == object_id:
                self._awaiting_move = None
            return self.memory.delete(object_id)

    def check_can_swap_memory(self):
        """Raises ValueError while a capture window is open (it would save into the swapped memory)."""
        with self.lock:
            if self.capture is not None:
                raise ValueError("finish or cancel the capture window first")

    def memory_swapped(self):
        """After memory.reload() put a different memory in place: forget greetings and the
        object waiting to be moved, both of which name objects of the old memory."""
        with self.lock:
            self._rec.clear()
            self._awaiting_move = None
            self._step_hold = None


def _banner(img, text):
    """A dark strip across the top of the image with white text."""
    h, w = img.shape[:2]
    scale = max(0.6, w / 1280 * 0.9)
    (tw, th), _ = cv2.getTextSize(text, 0, scale, 2)
    cv2.rectangle(img, (0, 0), (min(w, tw + 24), th + 20), (30, 30, 30), -1)
    cv2.putText(img, text, (12, th + 10), 0, scale, (255, 255, 255), 2, cv2.LINE_AA)


def draw_box(img, box, label, color, thickness=None, txt_color=(255, 255, 255), dashed=False):
    """cv2 box + filled label, like utils.plots.Annotator.box_label's cv2 path. Labels are
    forced to ASCII: cv2 can't render other scripts (the web page shows the real name)."""
    lw = thickness or max(round(sum(img.shape[:2]) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    if dashed:
        _dashed_rect(img, p1, p2, color, lw)
    else:
        cv2.rectangle(img, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    label = (label or "").encode("ascii", "replace").decode("ascii")
    if label:
        tf = max(lw - 1, 1) if thickness is None else 2
        scale = max(sum(img.shape[:2]) / 2 * 0.003, 2) / 3
        tw, th = cv2.getTextSize(label, 0, fontScale=scale, thickness=tf)[0]
        outside = p1[1] - th >= 3
        q2 = (p1[0] + tw, p1[1] - th - 3) if outside else (p1[0] + tw, p1[1] + th + 3)
        cv2.rectangle(img, p1, q2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (p1[0], p1[1] - 2 if outside else p1[1] + th + 2), 0, scale, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)


def _draw_center_mark(img, center):
    """A '+' with a dark outline, readable on any background. Its arms are as long as the
    slack a box gets for counting as on the mark."""
    x, y = int(center[0]), int(center[1])
    arm = max(12, int(min(img.shape[:2]) * CENTER_TOL_FRAC))
    for color, lw in (((0, 0, 0), 6), (GUIDE_COLOR, 2)):
        cv2.line(img, (x - arm, y), (x + arm, y), color, lw, cv2.LINE_AA)
        cv2.line(img, (x, y - arm), (x, y + arm), color, lw, cv2.LINE_AA)


def _dashed_rect(img, p1, p2, color, lw, dash=18):
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, lw)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, lw)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, lw)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, lw)
