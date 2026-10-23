"""
"Where did I last see it?" memory for the /assistant page (robot_assistant.py).

Every processed frame's named detections (known VOC classes + OOD objects the VLM, CLIP
memory or a human already named) are turned into *sightings*: the object's box, a short
spatial description relative to the other objects in the same frame ("on the sofa, next to
the teddy bear"), and -- when the RealSense depth came along with the frame -- how far away
and in which direction it is. Only the latest sighting per label is kept, plus a small copy
of the frame it came from, so the robot can answer from memory after it has driven away
and show the snapshot it saw the object in.

Everything here is plain numpy/cv2 on pixel boxes: no model calls, cheap enough to run on
every frame. It is deliberately in-memory only -- objects move, so last session's locations
would only mislead.
"""
import math
import threading

import cv2
import numpy as np

# VOC class names (robot_implementation.VOC_NAME_MAPPING) -> how a person would say them.
DISPLAY_NAMES = {
    "couch": "sofa", "tv": "TV", "tvmonitor": "TV", "dining table": "table", "diningtable": "table",
    "potted plant": "plant", "pottedplant": "plant", "motorcycle": "motorbike",
}
# Labels that never become sightings: still-unresolved OOD boxes, operator dismissals, the
# VLM's "this crop is just background" answer, and --skip-vlm's placeholder.
SKIP_LABELS = {"", "unknown", "dismissed", "background", "ood"}
# Things other objects can be "on". Compared against normalized labels *and* their display
# names, so both the detector's "couch"/"dining table" and a VLM's "sofa"/"desk" count.
SURFACES = {
    "table", "dining table", "coffee table", "side table", "desk", "sofa", "couch", "chair", "armchair",
    "bed", "shelf", "bookshelf", "counter", "countertop", "cabinet", "bench", "stool", "dresser",
    "nightstand", "tv stand", "drawer", "ottoman", "windowsill", "sideboard",
}

SNAPSHOT_MAX_DIM = 640
MAX_DEPTH_MM = 8000                # D435 readings past ~8 m are mostly noise
SUPPORT_DEPTH_TOLERANCE_M = 0.6    # "on the sofa" is rejected if it's this much nearer/farther than the sofa
TARGET_COLOR = (60, 60, 235)       # BGR red, like the infographic's highlight box
CONTEXT_COLOR = (200, 200, 200)


def norm_label(label):
    return " ".join((label or "").replace("_", " ").replace("-", " ").lower().split())


def display_name(label):
    key = norm_label(label)
    return DISPLAY_NAMES.get(key, key)


def _is_surface(label):
    key = norm_label(label)
    return key in SURFACES or display_name(key) in SURFACES


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _h_overlap(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))


def _box_gap(a, b):
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def _median_depth_m(depth_info, im_shape, box, shrink=0.25):
    """Median of the valid depth readings in the box (shrunk by `shrink` on every side, so
    the background around a loosely-fitting box doesn't drag the median), in metres, or None
    if too few pixels have a reading. The depth image may be smaller than the color frame
    (robot_client.py sends it at half resolution); boxes are rescaled to match."""
    if depth_info is None:
        return None
    depth = depth_info["depth_mm"]
    dh, dw = depth.shape[:2]
    h, w = im_shape[:2]
    sx, sy = dw / w, dh / h
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 + bw * shrink, x2 - bw * shrink
    y1, y2 = y1 + bh * shrink, y2 - bh * shrink
    c1, c2 = max(0, int(x1 * sx)), min(dw, int(math.ceil(x2 * sx)))
    r1, r2 = max(0, int(y1 * sy)), min(dh, int(math.ceil(y2 * sy)))
    if c2 <= c1 or r2 <= r1:
        return None
    patch = depth[r1:r2, c1:c2]
    valid = patch[(patch > 0) & (patch < MAX_DEPTH_MM)]
    if valid.size < max(5, min(30, patch.size // 3)):
        return None
    return float(np.median(valid)) / 1000.0


def estimate_position(box, depth_info, im_shape):
    """Distance + bearing of the box's centre from the camera, from the aligned depth and
    the color intrinsics (pinhole model). Returns None without usable depth. bearing_deg is
    positive to the robot's right."""
    z = _median_depth_m(depth_info, im_shape, box)
    if z is None:
        return None
    u = (box[0] + box[2]) / 2.0
    x = (u - depth_info["cx"]) * z / depth_info["fx"]
    distance = math.hypot(x, z)
    bearing = math.degrees(math.atan2(x, z))
    side = "right" if bearing > 0 else "left"
    if abs(bearing) < 10:
        where = "straight ahead"
    elif abs(bearing) < 30:
        where = f"ahead, slightly to my {side}"
    else:
        where = f"to my {side}"
    return {"distance_m": round(distance, 2), "bearing_deg": round(bearing, 1),
            "text": f"about {distance:.1f} m {where}"}


def _region_text(box, im_shape):
    cx = (box[0] + box[2]) / 2.0 / im_shape[1]
    if cx < 0.33:
        return "on the left side of my view"
    if cx > 0.67:
        return "on the right side of my view"
    return "in the middle of my view"


def describe_relation(target, others, im_shape, depth_info=None):
    """2D spatial relation of `target` ({"label","box"}) to the other objects in the same
    frame. "on X": X is a surface-like object clearly bigger than the target, under most of
    the target's width, with the target's bottom edge inside X's vertical span (and, with
    depth, at about the same distance). "next to Y": the closest other object, preferring
    ones resting at a similar height. Falls back to the region of the view."""
    tb = target["box"]
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    t_key = norm_label(target["label"])
    t_depth = _median_depth_m(depth_info, im_shape, tb) if depth_info is not None else None

    support, best_area = None, None
    for o in others:
        ob = o["box"]
        if norm_label(o["label"]) == t_key or not _is_surface(o["label"]):
            continue
        sh = ob[3] - ob[1]
        if _area(ob) < 1.5 * _area(tb) or tw <= 0 or _h_overlap(tb, ob) / tw < 0.6:
            continue
        if not (ob[1] - 0.05 * sh <= tb[3] <= ob[3] + 0.05 * sh):
            continue
        if t_depth is not None:
            # The strip of the support right under the target: if the target is really
            # resting on it, both are about the same distance from the camera.
            strip = (tb[0], tb[3], tb[2], min(ob[3], tb[3] + max(10.0, 0.3 * th)))
            s_depth = _median_depth_m(depth_info, im_shape, strip, shrink=0.0)
            if s_depth is not None and abs(s_depth - t_depth) > SUPPORT_DEPTH_TOLERANCE_M:
                continue
        if best_area is None or _area(ob) < best_area:
            support, best_area = o, _area(ob)

    neighbor, best_score = None, None
    size = max(tw, th, 1.0)
    for o in others:
        ob = o["box"]
        if o is support or norm_label(o["label"]) == t_key:
            continue
        cx, cy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0
        if ob[0] <= cx <= ob[2] and ob[1] <= cy <= ob[3]:
            continue  # it encloses the target (a surface we're not "on", a background region)
        gap = _box_gap(tb, ob) / size
        if gap >= 1.0:
            continue
        score = gap + 0.5 * abs(tb[3] - ob[3]) / max(th, 1.0)
        if norm_label(o["label"]) == "person":
            score += 10.0  # only "next to the person" when nothing else is close
        if best_score is None or score < best_score:
            neighbor, best_score = o, score

    parts = []
    if support is not None:
        parts.append(f"on the {display_name(support['label'])}")
    if neighbor is not None:
        parts.append(f"next to the {display_name(neighbor['label'])}")
    if not parts:
        parts.append(_region_text(tb, im_shape))
    return {
        "support": display_name(support["label"]) if support is not None else None,
        "neighbor": display_name(neighbor["label"]) if neighbor is not None else None,
        "support_box": list(support["box"]) if support is not None else None,
        "neighbor_box": list(neighbor["box"]) if neighbor is not None else None,
        "text": ", ".join(parts),
    }


def compute_sightings(objects, im_shape, depth_info=None):
    """objects: [{"label","source","box":[x1,y1,x2,y2] in frame px,"score"}] for one frame.
    Returns {normalized label: sighting} -- one per label (its highest-scoring instance),
    each with its relation to everything else in the frame and, with depth, its position."""
    best = {}
    for o in objects:
        key = norm_label(o["label"])
        if key in SKIP_LABELS:
            continue
        if key not in best or o["score"] > best[key]["score"]:
            best[key] = o
    out = {}
    for key, o in best.items():
        others = [x for x in objects if x is not o and norm_label(x["label"]) not in SKIP_LABELS]
        out[key] = {
            "key": key, "label": o["label"], "display": display_name(key), "source": o["source"],
            "box": [float(v) for v in o["box"]], "score": float(o["score"]),
            "relation": describe_relation(o, others, im_shape, depth_info),
            "position": estimate_position(o["box"], depth_info, im_shape) if depth_info is not None else None,
            "im_shape": (int(im_shape[0]), int(im_shape[1])),
        }
    return out


def _draw_box(img, box, scale, color, thickness, text):
    x1, y1, x2, y2 = (int(round(v * scale)) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, tht), base = cv2.getTextSize(text, font, fs, ft)
    ty = y1 - 4 if y1 - tht - 8 > 0 else y1 + tht + 6
    cv2.rectangle(img, (x1, ty - tht - 4), (x1 + tw + 6, ty + base - 2), color, -1)
    cv2.putText(img, text, (x1 + 3, ty - 2), font, fs, (20, 20, 20) if color == CONTEXT_COLOR else (255, 255, 255),
                ft, cv2.LINE_AA)


class SceneMemory:
    """Latest sighting per label + the (downscaled) frames those sightings came from. Only
    frames some sighting still points at are kept, so memory stays bounded by the number of
    distinct labels ever seen, not by how long the robot has been driving around."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sightings = {}  # key -> sighting (+ "frame_id", "ts", "prev_support")
        self._frames = {}     # frame_id -> (small BGR frame, scale from full-res px)

    def record(self, frame_id, ts, im0, sightings):
        if not sightings:
            return
        h, w = im0.shape[:2]
        scale = min(1.0, SNAPSHOT_MAX_DIM / max(h, w))
        small = cv2.resize(im0, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA) \
            if scale < 1.0 else im0.copy()
        with self._lock:
            self._frames[frame_id] = (small, scale)
            for key, s in sightings.items():
                prev = self._sightings.get(key)
                prev_support = prev["relation"]["support"] if prev else None
                self._sightings[key] = {
                    **s, "frame_id": frame_id, "ts": ts,
                    "prev_support": prev_support if prev_support != s["relation"]["support"] else
                    (prev or {}).get("prev_support"),
                }
            referenced = {s["frame_id"] for s in self._sightings.values()}
            for fid in [f for f in self._frames if f not in referenced]:
                del self._frames[fid]

    def get(self, key):
        with self._lock:
            s = self._sightings.get(norm_label(key))
            return dict(s) if s else None

    def all(self):
        with self._lock:
            return [dict(s) for s in self._sightings.values()]

    def entries(self, now):
        """Compact rows for the page's "Objects I know" strip, most recently seen first."""
        with self._lock:
            rows = [{
                "key": s["key"], "label": s["display"], "source": s["source"],
                "relation": s["relation"]["text"], "age_s": round(now - s["ts"], 1), "v": s["frame_id"],
            } for s in self._sightings.values()]
        rows.sort(key=lambda r: r["age_s"])
        return rows

    def snapshot_jpg(self, key, quality=85):
        """JPEG bytes of the frame `key` was last seen in: the object in a thick red box,
        what it's on / next to in thin grey ones (the infographic's step-4 inset)."""
        with self._lock:
            s = self._sightings.get(norm_label(key))
            frame = self._frames.get(s["frame_id"]) if s else None
            if frame is None:
                return None
            small, scale = frame
            img = small.copy()
        rel = s["relation"]
        if rel.get("support_box"):
            _draw_box(img, rel["support_box"], scale, CONTEXT_COLOR, 1, rel["support"])
        if rel.get("neighbor_box"):
            _draw_box(img, rel["neighbor_box"], scale, CONTEXT_COLOR, 1, rel["neighbor"])
        _draw_box(img, s["box"], scale, TARGET_COLOR, 3, s["display"])
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None
