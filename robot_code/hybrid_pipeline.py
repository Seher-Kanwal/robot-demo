"""
Part 3 of the demo, "OOD + memory + OED": the whole method in one pass per camera frame.

  1. OOD detection   the detector + energy score split every box into known (VOC) and OOD
  2. memory          every OOD box is looked up in Module 1's memory (the objects taught on the
                     teach page: multi-view DINOv2 embeddings in teach_mode/teach_memory/); a
                     match names it, and no VLM is asked
  3. OED (VLM)       only the OOD boxes the memory can't name go to Qwen3-VL (stage 1, stage 2
                     if stage 1 is unsure), through the same robot_implementation code as Module 2

It runs inside the Module 2 server (robot_web_app.py, page /hybrid), because Qwen is loaded
there. Module 1's memory is only read here, never written, and is read again whenever Module 1
changes it on disk (a newly taught object, a fresh-demo reset or restore), so objects taught in
Part 1 show up here without a restart.

Every OOD box gets a number (#1, #2, ...) that is the same on all the stage images and in the
per-object lists, so the page can show where each box went.
"""
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

FILE = Path(__file__).resolve()
TEACH_DIR = FILE.parent / "teach_mode"
if str(TEACH_DIR) not in sys.path:
    sys.path.append(str(TEACH_DIR))

import robot_implementation as ri  # noqa: E402
import imgutils  # noqa: E402 (teach_mode)
from embedder import DEFAULT_EMBEDDER, EMBEDDERS, Embedder  # noqa: E402
from multiview_memory import DEFAULT_ACCEPT_THRES, DEFAULT_MARGIN, MultiViewMemory  # noqa: E402

DEFAULT_MEMORY_DIR = TEACH_DIR / "teach_memory"  # Module 1's default --memory-dir
CROP_PAD = 0.05  # same crop as Module 1 (teacher.Teacher crop_pad), so the scores match its own
MIN_SIDE = 24    # same as Module 1: a box this small can't be embedded reliably, it goes to the VLM

# BGR, the colors robot_implementation.annotate() uses, so all pages read the same way
KNOWN_COLOR = ri.KNOWN_COLOR
OOD_COLOR = ri.OOD_COLOR
MEMORY_COLOR = ri.MEMORY_COLOR
VLM_COLOR = ri.VLM_COLOR
FADED_COLOR = (110, 110, 110)


def _ascii(text):
    # cv2.putText can't draw anything else; the page shows the real names.
    return (text or "").encode("ascii", "replace").decode("ascii")


class TaughtMemory:
    """Module 1's memory, read-only. refresh() loads the embedder on first use and reads the
    memory again whenever memory.json changes (Module 1 writes it atomically)."""

    def __init__(self, memory_dir=DEFAULT_MEMORY_DIR, device=None, accept_thres=DEFAULT_ACCEPT_THRES,
                 margin=DEFAULT_MARGIN):
        self.dir = Path(memory_dir)
        self.device = device
        self.accept_thres = accept_thres
        self.margin = margin
        self.embedder = None
        self.memory = None
        self.version = 0  # bumped on every reload, so the page knows when to refetch the gallery
        self._stamp = None
        self._lock = threading.Lock()

    def _file_stamp(self):
        try:
            st = (self.dir / "memory.json").stat()
            return st.st_mtime_ns, st.st_size
        except OSError:
            return None

    def _embedder_name(self):
        """The embedder the memory was built with, so MultiViewMemory never has to re-embed
        (and save) it: this process must not write Module 1's files."""
        try:
            model_id = json.loads((self.dir / "memory.json").read_text(encoding="utf-8")).get("embedder")
        except (OSError, ValueError):
            model_id = None
        return next((k for k, v in EMBEDDERS.items() if v == model_id), DEFAULT_EMBEDDER)

    def refresh(self):
        stamp = self._file_stamp()
        with self._lock:
            if self.memory is not None and stamp == self._stamp:
                return False
            name = self._embedder_name()
            if self.embedder is None or self.embedder.name != name:
                self.embedder = Embedder(name, device=self.device)
                self.embedder.embed([np.zeros((64, 64, 3), np.uint8)])  # warm-up
            self.memory = MultiViewMemory(self.dir, self.embedder, accept_thres=self.accept_thres, margin=self.margin)
            self._stamp = stamp
            self.version += 1
            return True

    def objects(self):
        if self.memory is None:
            return []
        return [{"id": o["id"], "name": o["name"], "n_views": o["n_views"],
                 "thumb": o["views"][0]["file"] if o["views"] else None}
                for o in self.memory.list_objects()]

    def first_view(self, object_id):
        obj = self.memory.get(object_id) if self.memory is not None else None
        return obj["views"][0]["file"] if obj and obj["views"] else None


def _draw(img, box, text, color, thickness=3, dashed=False, big=False):
    """big: the step images, which the page shows at about a quarter of their size."""
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    if big:
        thickness += 2
    if dashed:
        _dashed_rect(img, (x1, y1), (x2, y2), color, thickness)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if text:
        text = _ascii(text)
        scale, tw = (1.3, 3) if big else (0.7, 2)
        (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, tw)
        ty = y1 - 4 if y1 - h - base - 6 >= 0 else y1 + h + base + 4
        tx = max(0, min(x1, img.shape[1] - w - 8))  # keep the label inside the image
        cv2.rectangle(img, (tx, ty - h - base - 2), (tx + w + 8, ty + 2), color, -1)
        cv2.putText(img, text, (tx + 4, ty - base + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), tw, cv2.LINE_AA)


def _dashed_rect(img, p1, p2, color, lw, dash=16):
    (x1, y1), (x2, y2) = p1, p2
    for a, b, fixed, horizontal in ((x1, x2, y1, True), (x1, x2, y2, True), (y1, y2, x1, False), (y1, y2, x2, False)):
        for s in range(a, b, dash * 2):
            e = min(s + dash, b)
            if horizontal:
                cv2.line(img, (s, fixed), (e, fixed), color, lw, cv2.LINE_AA)
            else:
                cv2.line(img, (fixed, s), (fixed, e), color, lw, cv2.LINE_AA)


class HybridPipeline:
    """run() takes one frame's detections and returns the full result for the /hybrid page."""

    def __init__(self, memory, verify_model_id):
        self.memory = memory
        self.verify_model_id = verify_model_id

    def run(self, im0, boxes, labels, scores, cache, encode):
        """boxes: (N, 4) tensor xyxy in im0 pixels, labels / scores: (N,) tensors (as
        robot_web_app has them after ri.scale_boxes). cache: the spatial VLM cache for this
        path. encode(img_bgr, max_dim) -> data URI. Returns (result dict, ood_labels_by_row);
        the latter in the {row: (label, source)} shape robot_web_app/annotate use."""
        reloaded = self.memory.refresh()
        mem = self.memory.memory
        n = int(len(boxes))
        boxes_np = boxes.cpu().numpy().astype(np.float32).reshape(-1, 4) if n else np.zeros((0, 4), np.float32)
        labels_np = labels.cpu().numpy().astype(int).reshape(-1) if n else np.zeros((0,), int)
        scores_np = scores.cpu().numpy().astype(np.float32).reshape(-1) if n else np.zeros((0,), np.float32)
        ood_rows = [i for i in range(n) if labels_np[i] == ri.OOD_CLASS_ID]
        known_rows = [i for i in range(n) if labels_np[i] != ri.OOD_CLASS_ID]

        # one entry per OOD box, numbered left to right so the numbers are easy to follow on screen
        ood_rows.sort(key=lambda r: (boxes_np[r][0], boxes_np[r][1]))
        objs = []
        for k, r in enumerate(ood_rows, start=1):
            c = imgutils.crop(im0, boxes_np[r], CROP_PAD)
            objs.append({"n": k, "row": int(r), "box": [int(round(float(v))) for v in boxes_np[r]],
                         "ood_score": round(float(scores_np[r]), 3), "crop_img": c,
                         "route": None, "name": None, "source": None, "memory": None})

        # -- step 2: memory
        t0 = time.perf_counter()
        embeddable = [o for o in objs if o["crop_img"].size and imgutils.short_side(o["box"]) >= MIN_SIDE]
        if embeddable and mem is not None:
            matches = mem.match(self.memory.embedder.embed([o["crop_img"] for o in embeddable]))
            for o, m in zip(embeddable, matches):
                if m is None:
                    continue
                o["memory"] = {"name": m["name"], "object_id": m["object_id"], "score": round(m["score"], 3),
                               "second": round(m["second"], 3), "accepted": bool(m["accepted"]),
                               "view": self.memory.first_view(m["object_id"])}
                if m["accepted"]:
                    o.update(route="memory", name=m["name"], source="memory")
        for o in objs:
            if o["route"] is None and imgutils.short_side(o["box"]) < MIN_SIDE:
                o["too_small"] = True
        memory_ms = (time.perf_counter() - t0) * 1000

        # -- step 3: the VLM, only for what memory couldn't name
        rest = [o for o in objs if o["route"] is None]
        vlm_ms, new_calls = 0.0, 0
        if rest:
            idx = [o["row"] for o in rest]
            cached = [ri.ood_cache_key(boxes[r]) in cache for r in idx]
            new_calls = sum(1 for c in cached if not c)
            t1 = time.perf_counter()
            results = ri.label_ood_detections(im0, boxes[idx], cache, self.verify_model_id,
                                              human_qa_enabled=False, memory=None)
            vlm_ms = (time.perf_counter() - t1) * 1000
            for o, (label, src), was_cached in zip(rest, results, cached):
                unknown = ri._is_unknown(label)
                o.update(route="vlm", name=None if unknown else label, source=src, cached=bool(was_cached))

        ood_labels_by_row = {}
        for o in objs:
            ood_labels_by_row[o["row"]] = (o["name"] or "unknown", o["source"] or "vlm1")

        images = self._draw_all(im0, boxes_np, labels_np, scores_np, known_rows, objs, encode)
        known = [{"label": ri.VOC_NAME_MAPPING[int(labels_np[r])], "score": round(float(scores_np[r]), 3)}
                 for r in known_rows]
        out_objs = []
        for o in objs:
            item = {k: v for k, v in o.items() if k != "crop_img"}
            item["crop"] = encode(o["crop_img"], 160) if o["crop_img"].size else None
            out_objs.append(item)
        n_mem = sum(1 for o in objs if o["route"] == "memory")
        result = {
            "images": images, "objects": out_objs, "known": known,
            "counts": {"total": n, "known": len(known_rows), "ood": len(objs), "memory": n_mem,
                       "vlm": len(rest), "vlm_new": new_calls, "vlm_cached": len(rest) - new_calls,
                       "vlm_named": sum(1 for o in rest if o["name"]),
                       "unknown": sum(1 for o in rest if not o["name"])},
            "timing": {"memory_ms": round(memory_ms, 1), "vlm_ms": round(vlm_ms, 1)},
            "memory": {"version": self.memory.version, "reloaded": reloaded,
                       "objects": len(mem.list_objects()) if mem is not None else 0,
                       "accept_thres": self.memory.accept_thres, "margin": self.memory.margin,
                       "embedder": self.memory.embedder.model_id if self.memory.embedder else None},
        }
        return result, ood_labels_by_row

    @staticmethod
    def _draw_all(im0, boxes, labels, scores, known_rows, objs, encode):
        def base(with_known):
            img = np.ascontiguousarray(im0.copy())
            if with_known:
                for r in known_rows:
                    _draw(img, boxes[r], f"{ri.VOC_NAME_MAPPING[int(labels[r])]} {scores[r]:.2f}", KNOWN_COLOR, 2)
            return img

        # top left: the OOD detector on its own -- known classes + every OOD box as "unknown"
        ood_only = base(True)
        for o in objs:
            _draw(ood_only, o["box"], f"#{o['n']} unknown {o['ood_score']:.2f}", OOD_COLOR)
        # top right: the final answer of the whole pipeline
        final = base(True)
        for o in objs:
            if o["route"] == "memory":
                _draw(final, o["box"], f"#{o['n']} {o['name']} [memory]", MEMORY_COLOR)
            elif o["name"]:
                _draw(final, o["box"], f"#{o['n']} {o['name']} [vlm]", VLM_COLOR)
            else:
                _draw(final, o["box"], f"#{o['n']} unknown", OOD_COLOR)
        # bottom: one image per step, only that step's boxes
        step1 = base(False)
        for o in objs:
            _draw(step1, o["box"], f"#{o['n']} OOD {o['ood_score']:.2f}", OOD_COLOR, big=True)
        step2 = base(False)
        for o in objs:
            if o["route"] == "memory":
                _draw(step2, o["box"], f"#{o['n']} {o['name']} {o['memory']['score']:.2f}", MEMORY_COLOR, big=True)
            else:
                _draw(step2, o["box"], f"#{o['n']} not in memory", FADED_COLOR, 2, dashed=True, big=True)
        step3 = base(False)
        for o in objs:
            if o["route"] == "memory":
                _draw(step3, o["box"], f"#{o['n']} (memory)", FADED_COLOR, 2, dashed=True, big=True)
            elif o["name"]:
                _draw(step3, o["box"], f"#{o['n']} {o['name']}", VLM_COLOR, big=True)
            else:
                _draw(step3, o["box"], f"#{o['n']} unknown", OOD_COLOR, big=True)
        return {"ood": encode(ood_only, 1280), "final": encode(final, 1280),
                "step1": encode(step1, 800), "step2": encode(step2, 800), "step3": encode(step3, 800)}
