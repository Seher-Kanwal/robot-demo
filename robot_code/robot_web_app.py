"""
Local interactive web demo for the OOD pipeline in robot_implementation.py.

Runs the same detector + energy-OOD split + VLM stage-1/stage-2 logic, but
instead of an OpenCV window and a blocking stdin prompt for stage-3 QA, it
serves robot_code/live_feed_demo.html and exposes two small JSON endpoints so
the operator can pick/upload an image and type names for anything still
"unknown" after both VLM passes, from the browser.

Usage:
    python robot_code/robot_web_app.py [--weights ...] [--host 127.0.0.1] [--port 7860]

Then open http://<host>:<port>/ in a browser.

Stage 1 (the primary VLM) by default talks to a persistent vLLM server instead of
loading the checkpoint in-process -- start it first, in a separate terminal:

    robot_code/start_vllm_server.sh
    python robot_code/robot_web_app.py

Pass --no-vllm to load stage 1 locally instead (no server needed).

The page can also run a "Qwen3-VL direct" baseline next to the pipeline: the same VLM
gets the *whole* image and has to find + name every object itself (boxes and all), with no
detector in front of it -- /api/qwen-direct/run for one image, /api/qwen-direct/live to
toggle it on the --realsense/--ros-topic feed. The two take turns on the GPU (GPU_LOCK), so
the ms/frame the page shows for each is that path's own compute time. Run on its own
("Module 3", http://<host>:<port>/qwen-direct), it pauses the detector pipeline instead.

The live view stops on a frame that still has objects nobody could name, until each one is
named or dismissed on the page (--no-freeze to keep streaming).

Part 3 of the demo, "OOD + memory + OED" (http://<host>:<port>/hybrid, hybrid_pipeline.py),
also runs here: OOD boxes are looked up in Module 1's taught memory first, and only the rest go
to the VLM. It replaces the Module 2 pipeline on the live feed while that page is open.
"""
import argparse
import base64
import csv
import io
import json
import os
import statistics
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, request, send_from_directory
from PIL import Image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(FILE.parent) not in sys.path:
    sys.path.append(str(FILE.parent))

import robot_implementation as ri  # noqa: E402 (sets QWEN_MODEL_ID / sys.path on import)
import ood_memory  # noqa: E402
import robot_assistant  # noqa: E402

PAGE_PATH = FILE.parent / "live_feed_demo.html"
ASSISTANT_PAGE_PATH = FILE.parent / "robot_assistant.html"
MODULE3_PAGE_PATH = FILE.parent / "qwen_direct.html"
HYBRID_PAGE_PATH = FILE.parent / "hybrid_pipeline.html"

# Curated set of repo-root images that make decent demo material; anything
# missing is silently skipped so this doesn't break if the repo changes.
SAMPLE_IMAGE_CANDIDATES = [
    "table setup.jpg", "inDoor.jpg", "ID.jpg", "000000000009.jpg", "10000_1.jpg", "output.jpg",
]
# Same idea for videos: a specific frame is pulled out on demand (see api_run),
# so the operator can step through a clip looking for a good OOD case instead
# of only ever testing a single still image.
SAMPLE_VIDEO_CANDIDATES = [
    "Videos/15_seconds.mp4", "Videos/Indoor_scene_3.mp4", "Videos/walking_scene.mp4", "Videos/testing.mp4",
]


def _video_frame_count(path):
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


SAMPLES_BY_NAME = OrderedDict()
for _name in SAMPLE_IMAGE_CANDIDATES:
    _path = ROOT / _name
    if _path.is_file():
        SAMPLES_BY_NAME[_name] = {"path": _path, "kind": "image"}
for _name in SAMPLE_VIDEO_CANDIDATES:
    _path = ROOT / _name
    if _path.is_file():
        SAMPLES_BY_NAME[_name] = {"path": _path, "kind": "video", "frames": _video_frame_count(_path)}
# Also offer anything dropped into the repo-root Samples/ folder, so new demo material
# shows up in the picker without editing the candidate lists above.
_SAMPLES_DIR = ROOT / "Samples"
if _SAMPLES_DIR.is_dir():
    for _path in sorted(_SAMPLES_DIR.iterdir()):
        _name = f"Samples/{_path.name}"
        _ext = _path.suffix.lower().lstrip(".")
        if not _path.is_file() or _name in SAMPLES_BY_NAME:
            continue
        if _ext in ("jpg", "jpeg", "png", "bmp", "webp"):
            SAMPLES_BY_NAME[_name] = {"path": _path, "kind": "image"}
        elif _ext in ("mp4", "avi", "mov", "mkv", "webm"):
            SAMPLES_BY_NAME[_name] = {"path": _path, "kind": "video", "frames": _video_frame_count(_path)}

app = Flask(__name__)

# job_id -> the frame + detections that produced it, so /api/submit can finish
# annotating the exact same frame once the operator types names in. Bounded like CACHE
# below -- a long-running live push session (robot_client.py) creates one job per frame,
# and nothing else ever removes an entry, so this would otherwise grow forever.
MAX_STATE_ENTRIES = 200
STATE = OrderedDict()
CACHE = OrderedDict()  # same spatial LRU role as `cache` in robot_implementation.main()
PREDICTOR = None
ARGS = None
MEMORY = None  # set in main(), once ARGS (--memory-path/--memory-sim-thres) is parsed
LIVE_SOURCE = None  # human-readable label of the --ros-topic/--realsense feed, None if neither
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Live (pushed frames / robot_client.py) mode -------------------------------
# Frames arriving at /api/run with live=1 (robot_client.py always sends this) also get
# stashed here, so the page can poll /api/pushed/latest and show whatever a remote
# client is streaming in -- no rclpy needed on this side at all (contrast the --ros-topic
# / live_loop path above, which needs rclpy in this same process). Each pushed frame
# is an ordinary job_id in STATE, so stage 3 for it reuses the normal /api/submit/<job_id>
# endpoint below -- the only addition there is writing resolved names back into the
# shared CACHE so a name typed once actually resolves later frames near the same spot,
# instead of asking again every frame.
PUSHED_LOCK = threading.Lock()
PUSHED_STATE = {
    "ready": False, "job_id": None, "source": None, "image": None, "image_ood": None, "resolution": None,
    "stats": None, "labels": [], "known_labels": [], "pending": [], "energy_thres": None, "frame_count": 0,
}
# Every still-unknown object ever seen on the pushed-frame stream, keyed by its spatial
# cache key (same idea as LIVE_UNKNOWN below, for the --ros-topic path). Each pushed
# frame gets a brand-new job_id, so a per-job "pending" list would lose every unresolved
# object the instant the next frame arrives -- this dict is what actually survives across
# frames. An entry is only removed once the operator types a real name via
# /api/pushed/label; it never disappears just because a later frame doesn't detect it.
PUSHED_UNKNOWN = OrderedDict()

# --- Live (--ros-topic / --realsense) mode -------------------------------------
# A background thread (live_loop) continuously pulls frames from a ROS 2
# topic, runs the same detector + energy-OOD + VLM stage-1/2 pipeline as
# api_run(), and publishes the latest result into LIVE_STATE for the page to
# poll via /api/live/latest. Stage 3 (still-unknown objects) never blocks this
# loop -- the browser submits names asynchronously via /api/live/label, which
# writes straight into CACHE_LIVE so the *next* frame near that same spot
# picks up the human-provided name instead of re-querying the VLM.
CACHE_LIVE = OrderedDict()
LIVE_LOCK = threading.Lock()
LIVE_STATE = {
    "ready": False, "error": None, "image": None, "image_ood": None, "resolution": None,
    "stats": None, "labels": [], "known_labels": [], "pending": [], "frame_count": 0, "timing": None,
}
# Every still-unknown object ever seen on the ROS live loop, keyed by its spatial cache
# key -- unlike LIVE_STATE["pending"] (which used to just be "whatever is unknown in the
# *current* frame"), an entry here is never dropped just because the object isn't in the
# latest frame (camera panned away, object occluded, detector missed it on one pass). It's
# only removed once the operator actually types a name for it in api_live_label, so the
# QA list on the page keeps accumulating instead of losing entries the operator hasn't
# had a chance to name yet.
LIVE_UNKNOWN = OrderedDict()
# The --realsense/--ros-topic frame iterator, shared by live_loop and qwen_live_loop through
# grab_frame(). Both sources drain whatever queued up and hand out the newest frame on every
# next(), so each GPU pass grabs its frame only once the GPU is its own (see GPU_LOCK): a
# pass always works on what the camera sees *now*, never on a frame that sat waiting while
# the other path (or the previous several-second Qwen pass) had the GPU. A generator can't
# be entered from two threads at once, hence FRAME_LOCK. FRAME_SEQ counts frames grabbed.
# The iterator is opened on first use and closed again while the camera is switched off
# (POST /api/camera, see release_camera), so the hub can hand the RealSense to Module 1
# without this server reloading its models. LIVE_FRAMES_FACTORY opens a fresh one.
LIVE_FRAMES = None
LIVE_FRAMES_FACTORY = None
FRAME_LOCK = threading.Lock()
FRAME_SEQ = 0
CAMERA_ON = threading.Event()
CAMERA = {"state": "off"}  # "off" (released), "opening" or "on"


class CameraPaused(RuntimeError):
    """grab_frame() while the camera is switched off; the live loops wait for it to come back."""
# Module 3 on its own: how often live_loop refreshes the plain camera picture the page shows
# (it polls once a second). Grabbing + JPEG-encoding every camera frame instead only steals
# CPU/GIL time from the Qwen pass's token-by-token generate() loop.
SOLO_PREVIEW_INTERVAL = 0.25
# Stop on the image (live loop): when a live frame still has objects both VLM stages called
# "unknown", the live view stops on that frame until the operator has named or dismissed every
# one of them, then goes live again (--no-freeze keeps streaming instead). Holds that frame,
# its detections and the still-unnamed objects ("pending", keyed like LIVE_UNKNOWN), or None.
# Guarded by LIVE_LOCK.
LIVE_FROZEN = None

# --- GPU turn-taking ------------------------------------------------------------
# Held around each path's heavy compute (detector + VLM stages in api_run/live_loop, the
# whole-image grounding call in run_qwen_direct/qwen_live_loop), so the pipeline and the
# Qwen-direct baseline alternate on the GPU instead of slowing each other down mid-run. Every
# timer that the page compares starts *after* acquiring it, so waiting for the other path's
# turn never counts against either one. Lock order is always GPU_LOCK -> FRAME_LOCK and
# GPU_LOCK -> qwen3_generator's own _GENERATE_LOCK, never the reverse.
GPU_LOCK = threading.Lock()

# --- Qwen-direct baseline (whole image -> VLM boxes + labels, no detector) -------
QWEN_DIRECT_COLOR = (142, 207, 62)  # BGR green; matches --qwen in live_feed_demo.html
QWEN_LIVE_ENABLED = threading.Event()  # set/cleared from the page via /api/qwen-direct/live
# "Module 3" on its own: while set, live_loop skips the detector + VLM pipeline entirely (it only
# refreshes the plain camera picture), so the Qwen-direct baseline gets the GPU to itself.
# Set by /api/qwen-direct/live with {"enabled": true, "solo": true}; any other toggle clears it.
QWEN_SOLO = threading.Event()
QWEN_LIVE_LOCK = threading.Lock()
QWEN_LIVE_STATE = {
    "ready": False, "error": None, "image": None, "frame_count": None, "resolution": None,
    "labels": [], "count": 0, "timing": None, "raw_text": "", "runs": 0, "model_id": None,
    "session": 0,
}

# --- Part 3: OOD + memory + VLM (/hybrid page, hybrid_pipeline.py) ---------------
# While the /hybrid page is open, live_loop runs hybrid_step() instead of the Module 2 pipeline:
# every OOD box is looked up in Module 1's memory first, and only what memory can't name goes
# to the VLM. The page switches it on when it opens and off when it closes; a page that vanished
# without saying so (a crashed tab) switches it off by not polling for HYBRID_TIMEOUT_S.
HYBRID_TIMEOUT_S = 8.0
HYBRID_LOCK = threading.Lock()
HYBRID = {"enabled": False, "last_poll": 0.0, "fps": None, "last_t": None}
HYBRID_STATE = {"ready": False, "error": None, "warming": None, "seq": 0, "frame_count": 0, "totals": None}
HYBRID_PIPELINE = None  # hybrid_pipeline.HybridPipeline, built on first use (loads the DINOv2 embedder)
CACHE_HYBRID = OrderedDict()  # the VLM's spatial cache for this path, like CACHE_LIVE


def hybrid_totals():
    return {"frames": 0, "ood": 0, "memory": 0, "vlm": 0, "vlm_new": 0, "unknown": 0,
            "detect_ms": 0.0, "memory_ms": 0.0, "vlm_ms": 0.0}


def hybrid_active():
    with HYBRID_LOCK:
        return HYBRID["enabled"] and time.time() - HYBRID["last_poll"] < HYBRID_TIMEOUT_S


# --- OOD + VLM vs VLM-only results (the page's "Comparison results" panel) --------
# One row per finished frame of either method: the live pipeline (live_loop), the live
# Qwen-direct baseline (qwen_live_loop, side by side or solo), and the single-image runs
# (/api/run, /api/qwen-direct/run). Served as a per-method summary and as a CSV for the report.
COMPARE_METHODS = {"pipeline": "OOD + VLM (detector + energy OOD + Qwen)", "qwen": "VLM only (Qwen3-VL direct)"}
COMPARE_LOCK = threading.Lock()
COMPARE_LOG = []
COMPARE_LOG_MAX = 20000


def compare_log(mode, method, source, frame, ms, n_objects, labels):
    """mode: "live" or "image"; method: a COMPARE_METHODS key; labels: the names it gave."""
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": mode, "method": method, "source": source,
           "frame": frame, "ms": round(float(ms), 1), "objects": int(n_objects), "labels": list(labels)}
    with COMPARE_LOCK:
        COMPARE_LOG.append(row)
        del COMPARE_LOG[:-COMPARE_LOG_MAX]


def pipeline_label_names(known_labels, resolved_labels):
    return list(known_labels) + [r["label"] for r in resolved_labels]


def compare_summary():
    """{mode: {method: {frames, mean_ms, median_ms, fps, mean_objects, labels: [[name, n], ...]}}}.
    fps is 1000 / mean ms: the rate the method manages on its own (the live loops take turns
    on the GPU while compared, so their wall-clock fps there is lower for both)."""
    with COMPARE_LOCK:
        rows = list(COMPARE_LOG)
    out = {}
    for mode in ("live", "image"):
        for method in COMPARE_METHODS:
            sel = [r for r in rows if r["mode"] == mode and r["method"] == method]
            if not sel:
                continue
            ms = [r["ms"] for r in sel]
            counts = {}
            for r in sel:
                for name in r["labels"]:
                    counts[name] = counts.get(name, 0) + 1
            mean_ms = statistics.fmean(ms)
            out.setdefault(mode, {})[method] = {
                "frames": len(sel), "mean_ms": round(mean_ms, 1), "median_ms": round(statistics.median(ms), 1),
                "fps": round(1000.0 / mean_ms, 2) if mean_ms > 0 else None,
                "mean_objects": round(statistics.fmean(r["objects"] for r in sel), 2),
                "labels": sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])),
            }
    return out


# --- Robot assistant (/assistant page, robot_assistant.py) ----------------------
# Fed every processed frame from both live paths (live_loop and robot_client.py pushes)
# via assistant_observe(); answers "where are my X?" from them. Set in main().
ASSISTANT = None
# When the /assistant page last polled. While it's open the live view must keep moving
# (the user turns the robot to new views to search), so the live loop doesn't stop on
# unnamed objects then -- nobody on that page is there to name them. See assistant_active().
ASSISTANT_TIMEOUT_S = 8.0
ASSISTANT_POLL = {"last": 0.0}
# Last downscaled grayscale frame per live path, for flush_cache_on_motion().
MOTION_PREV = {}


def assistant_active():
    return time.time() - ASSISTANT_POLL["last"] < ASSISTANT_TIMEOUT_S


def flush_cache_on_motion(name, im0, cache):
    """CACHE/CACHE_LIVE are keyed by on-screen box position, which is only valid while the
    camera stands still: once the robot drives or turns, a *different* object landing on the
    same pixels would inherit the old object's name (and the assistant would cheerfully
    report the wrong thing as found). So when consecutive frames of one path differ by more
    than --motion-flush-thres (mean absolute difference of 160x90 grayscale thumbnails,
    0-255), that path's spatial cache is cleared. Names a human gave survive anyway: CLIP
    MEMORY re-recognizes those by appearance."""
    if not ARGS.motion_flush_thres:
        return False
    small = cv2.cvtColor(cv2.resize(im0, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    prev = MOTION_PREV.get(name)
    MOTION_PREV[name] = small
    if prev is None:
        return False
    if float(cv2.absdiff(small, prev).mean()) > ARGS.motion_flush_thres:
        cache.clear()
        return True
    return False


def parse_depth_upload():
    """robot_client.py --depth sends a 16-bit PNG (`depth`, millimetres, aligned to the color
    frame but possibly smaller) plus the color intrinsics (`intrinsics` JSON). Returns the
    depth_info dict scene_memory expects, or None if absent/malformed -- depth only adds
    distances to the assistant's answers, so a bad upload must never fail the frame."""
    f = request.files.get("depth")
    if f is None:
        return None
    try:
        depth_mm = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        intr = json.loads(request.form.get("intrinsics") or "{}")
        if depth_mm is None or depth_mm.dtype != np.uint16 or depth_mm.ndim != 2:
            return None
        return {"depth_mm": depth_mm, **{k: float(intr[k]) for k in ("fx", "fy", "cx", "cy")}}
    except (ValueError, KeyError, TypeError, cv2.error) as e:
        print(f"[robot_web_app] ignoring malformed depth upload: {e}")
        return None


def assistant_observe(im0, boxes, labels, scores, ood_labels_by_row, preview_b64, depth_info, source):
    """Hands one processed frame to the assistant. Never raises: a bug in the assistant
    must not take the live detection pipeline down with it."""
    if ASSISTANT is None:
        return
    try:
        ASSISTANT.observe(im0, boxes, labels, scores, ood_labels_by_row, preview_b64, depth_info, source)
    except Exception as e:
        print(f"[robot_web_app] assistant observe failed: {e}")


def cache_key_to_str(key):
    return "_".join(str(v) for v in key)


def cache_key_from_str(key_str):
    return tuple(int(v) for v in key_str.split("_"))


def get_predictor():
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = ri.build_model(ARGS.cfg, ARGS.weights, ARGS.conf_thres, DEVICE)
    return PREDICTOR


def encode_jpg(img_bgr, max_dim=1400, quality=90):
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))))
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("failed to encode image")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def decode_jpg_b64(data_uri):
    """Reverses encode_jpg's "data:image/jpeg;base64,..." output back into a BGR ndarray,
    or None if data_uri is falsy. Used to recover a stored crop thumbnail (see
    PUSHED_UNKNOWN/LIVE_UNKNOWN) for OODMemory.remember() once the operator names it."""
    if not data_uri:
        return None
    _, _, b64 = data_uri.partition(",")
    raw = base64.b64decode(b64)
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def clip_box(im0, box):
    h0, w0 = im0.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box.tolist())
    x1, x2 = max(0, min(w0, x1)), max(0, min(w0, x2))
    y1, y2 = max(0, min(h0, y1)), max(0, min(h0, y2))
    return x1, y1, x2, y2


def stats_and_labels(labels_tensor, ood_labels_by_row, boxes):
    n_total = int(len(labels_tensor))
    n_known = int((labels_tensor != ri.OOD_CLASS_ID).sum().item()) if n_total else 0
    n_ood = len(ood_labels_by_row)
    # Tally by *how* each OOD box was resolved (source), not just whether it's
    # still "unknown" -- once a human types a name it's no longer pending, but
    # it was never a VLM label either, so it needs its own bucket.
    n_pending = n_human = n_vlm = n_memory = 0
    resolved = []
    for row, (label, src) in ood_labels_by_row.items():
        if ri._is_unknown(label):
            n_pending += 1
        elif src == "dismissed":
            pass  # operator dismissed it without naming it -- not "resolved", don't list/count it
        else:
            if src == "human":
                n_human += 1
            elif src == "memory":
                n_memory += 1
            else:
                n_vlm += 1
            # One pill per resolved box, not deduped by name -- two different objects
            # sharing a name (e.g. two "cup"s from different sources) are still two
            # distinct sightings the operator may want to tell apart by provenance.
            # `key` is the same spatial cache key `pending` items carry -- lets the
            # frontend voice assistant announce a resolved object once instead of on
            # every poll tick it's still in frame.
            resolved.append({
                "label": label, "source": src,
                "key": cache_key_to_str(ri.ood_cache_key(boxes[row])),
            })
    resolved.sort(key=lambda item: (item["label"], item["source"]))
    # Known (in-distribution) class names present in this frame -- unlike `resolved`,
    # not deduped/keyed by object identity (the detector doesn't track identity across
    # frames); the frontend voice assistant dedupes these by label text instead.
    known_labels = [
        ri.VOC_NAME_MAPPING[int(labels_tensor[row])]
        for row in range(n_total) if int(labels_tensor[row]) != ri.OOD_CLASS_ID
    ]
    stats = {
        "total": n_total, "known": n_known, "ood": n_ood,
        "ood_vlm": n_vlm, "ood_human": n_human, "ood_memory": n_memory, "pending": n_pending,
    }
    return stats, resolved, known_labels


def label_ood_detections_timed(im0, ood_boxes_xyxy, cache, verify_model_id, memory=None):
    """Same resolution logic as ri.label_ood_detections(..., human_qa_enabled=False),
    but issues one VLM call per object instead of batching the whole frame together,
    and times each call. That's deliberately less efficient than the production path
    (see robot_implementation.label_ood_detections, which batches for throughput) —
    it exists so a single "one query, one object" latency is actually measurable,
    which a batched call can't give you.

    Returns (results, timings), both aligned 1:1 with ood_boxes_xyxy: results is the
    usual list[(label, source)]; timings is a list of per-row dicts with cached /
    stage1_ms / stage2_ms / total_ms.
    """
    from OOD_object_name_generation.qwen3_generator import describe_images

    h0, w0 = im0.shape[:2]
    results = []
    timings = []

    for box in ood_boxes_xyxy:
        key = ri.ood_cache_key(box)
        if key in cache:
            cache.move_to_end(key)
            results.append(cache[key])
            timings.append({"cached": True, "stage1_ms": 0.0, "stage2_ms": 0.0, "total_ms": 0.0})
            continue

        x1, y1, x2, y2 = clip_box(im0, box)
        crop = im0[y1:y2, x1:x2]
        if crop.size == 0:
            results.append(("unknown", "vlm1"))
            cache[key] = results[-1]
            timings.append({"cached": False, "stage1_ms": 0.0, "stage2_ms": 0.0, "total_ms": 0.0})
            continue
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        remembered = memory.lookup(crop_pil) if memory is not None else None
        if remembered:
            results.append((remembered, "memory"))
            cache[key] = results[-1]
            timings.append({"cached": False, "stage1_ms": 0.0, "stage2_ms": 0.0, "total_ms": 0.0})
            continue

        t0 = time.perf_counter()
        stage1_label = describe_images([crop_pil], max_new_tokens=6, batch_size=1)[0]
        stage1_ms = (time.perf_counter() - t0) * 1000

        stage2_ms = 0.0
        label, src = stage1_label, "vlm1"
        if ri._is_unknown(stage1_label):
            t1 = time.perf_counter()
            stage2_label = describe_images(
                [crop_pil], prompt=ri.VERIFY_PROMPT, max_new_tokens=6, batch_size=1, model_id=verify_model_id,
            )[0]
            stage2_ms = (time.perf_counter() - t1) * 1000
            label, src = stage2_label, "vlm2"

        results.append((label, src))
        cache[key] = results[-1]
        timings.append({
            "cached": False, "stage1_ms": stage1_ms, "stage2_ms": stage2_ms,
            "total_ms": stage1_ms + stage2_ms,
        })

    return results, timings


def update_fps(prev_fps, last_t, now, alpha=0.3):
    """Exponential moving average of the wall-clock rate between successive results --
    i.e. *observed* throughput, including any time spent waiting for the other path's GPU
    turn (contrast the per-frame compute ms, which excludes it). None until two results."""
    if last_t is None:
        return prev_fps
    inst = 1.0 / max(1e-6, now - last_t)
    return inst if prev_fps is None else (1 - alpha) * prev_fps + alpha * inst


def qwen_direct_model_id():
    return ARGS.qwen_direct_model_id or os.environ.get("QWEN_MODEL_ID")


def grab_frame():
    """The live source's newest frame, as (im0, depth_info, seq). Thread-safe; see LIVE_FRAMES.
    Opens the camera if needed; raises CameraPaused while it's switched off."""
    global FRAME_SEQ, LIVE_FRAMES
    with FRAME_LOCK:
        if not CAMERA_ON.is_set():
            raise CameraPaused("the camera is switched off")
        if LIVE_FRAMES is None:
            CAMERA["state"] = "opening"
            LIVE_FRAMES = LIVE_FRAMES_FACTORY()
        try:
            item = next(LIVE_FRAMES, None)
        except Exception:
            LIVE_FRAMES = None  # a generator that raised is finished; the next grab reopens
            CAMERA["state"] = "off"
            raise
        if item is None:
            LIVE_FRAMES = None
            raise RuntimeError("the camera stream ended")
        CAMERA["state"] = "on"
        FRAME_SEQ += 1
        seq = FRAME_SEQ
    # (im0, is_last), or (im0, is_last, depth_info) from realsense_iter_frames(with_depth=True)
    return item[0], (item[2] if len(item) > 2 else None), seq


def release_camera():
    """Switches the camera off and closes the RealSense (once no grab is in progress)."""
    global LIVE_FRAMES
    CAMERA_ON.clear()
    with FRAME_LOCK:
        if LIVE_FRAMES is not None:
            LIVE_FRAMES.close()  # runs the generator's finally: pipeline.stop()
            LIVE_FRAMES = None
        CAMERA["state"] = "off"


def qwen_direct_infer(im0):
    """The baseline: one whole-image grounding call, no detector/OOD split/crops. The caller
    holds GPU_LOCK. Returns (detections, raw_text, infer_ms)."""
    from OOD_object_name_generation.qwen3_generator import detect_objects

    img = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
    t0 = time.perf_counter()
    detections, raw_text = detect_objects(
        img, max_new_tokens=ARGS.qwen_direct_max_tokens, model_id=ARGS.qwen_direct_model_id,
    )
    return detections, raw_text, (time.perf_counter() - t0) * 1000


def run_qwen_direct(im0):
    """qwen_direct_infer() in its own GPU turn; infer_ms is timed inside GPU_LOCK (see its comment)."""
    with GPU_LOCK:
        return qwen_direct_infer(im0)


def annotate_qwen_direct(im0, detections):
    annotator = ri.Annotator(np.ascontiguousarray(im0), line_width=3, example=str(ri.VOC_NAME_MAPPING))
    for det in detections:
        # The VLM's labels are free text: the cv2 drawing path Annotator picks for an ASCII
        # `example` has no fallback for anything else, so keep the drawn text ASCII.
        text = det["label"].encode("ascii", "replace").decode("ascii")
        annotator.box_label(det["box"], text, color=QWEN_DIRECT_COLOR)
    return annotator.result()


def qwen_direct_labels(detections):
    # Same {label, source} shape as stats_and_labels()' resolved list, so the page's
    # shared pill renderer handles it unchanged.
    return sorted(({"label": det["label"], "source": "qwen"} for det in detections), key=lambda d: d["label"])


def qwen_live_loop():
    """Background worker for the live Qwen-direct comparison. Idles until the page sets
    QWEN_LIVE_ENABLED, then runs the baseline back to back, each pass on the camera frame
    current at the moment it gets the GPU (see LIVE_FRAMES) -- it's several seconds per pass,
    so it naturally skips the frames in between. Side by side with Module 2 (not solo), it
    waits for the pipeline to finish one frame between passes so the two take turns.
    Like live_loop, never raises: an error is recorded for the page and the comparison
    switches itself off."""
    last_pipeline_frame = None
    last_done_t = None
    fps = None
    session = None
    while True:
        QWEN_LIVE_ENABLED.wait()
        with QWEN_LIVE_LOCK:
            current_session = QWEN_LIVE_STATE["session"]
        if current_session != session:  # toggled off and on again: start a new fps average
            session, last_done_t, fps = current_session, None, None
        if not QWEN_SOLO.is_set():
            with LIVE_LOCK:
                pipeline_frame = LIVE_STATE["frame_count"]
                # Stopped on a frame, or crashed: the pipeline won't take a turn, don't wait for it.
                pipeline_running = LIVE_FROZEN is None and not LIVE_STATE["error"]
            if pipeline_running and pipeline_frame == last_pipeline_frame:
                time.sleep(0.02)
                continue
            last_pipeline_frame = pipeline_frame
        try:
            with GPU_LOCK:
                im0, _depth, frame_count = grab_frame()
                detections, raw_text, infer_ms = qwen_direct_infer(im0)
            image = encode_jpg(annotate_qwen_direct(im0.copy(), detections))
        except CameraPaused:
            time.sleep(0.2)
            continue
        except Exception as e:
            print(f"[robot_web_app] qwen-direct live loop error: {e}")
            QWEN_LIVE_ENABLED.clear()
            with QWEN_LIVE_LOCK:
                QWEN_LIVE_STATE["error"] = str(e)
            continue
        now = time.perf_counter()
        fps = update_fps(fps, last_done_t, now)
        last_done_t = now
        with QWEN_LIVE_LOCK:
            QWEN_LIVE_STATE.update({
                "ready": True, "error": None, "image": image, "frame_count": frame_count,
                "resolution": f"{im0.shape[1]}x{im0.shape[0]}",
                "labels": qwen_direct_labels(detections), "count": len(detections),
                "timing": {"infer_ms": round(infer_ms, 1), "fps": round(fps, 2) if fps else None},
                "raw_text": raw_text, "runs": QWEN_LIVE_STATE["runs"] + 1,
                "model_id": qwen_direct_model_id(),
            })
        compare_log("live", "qwen", LIVE_SOURCE, frame_count, infer_ms, len(detections),
                    [det["label"] for det in detections])


def live_loop():
    """Runs forever: _run_live_loop() while the camera is switched on (CAMERA_ON). A camera
    error switches it off (and frees the device) instead of ending the thread for good."""
    while True:
        CAMERA_ON.wait()
        _run_live_loop()


def _run_live_loop():
    """Runs on the live loop's thread until the camera is switched off: grabs frames from the
    live source (a ROS 2 topic via ri.ros_iter_frames, or a USB RealSense via
    ri.realsense_iter_frames; see grab_frame), runs the real pipeline (batched, unlike
    label_ood_detections_timed above) with stage 3 disabled, and publishes each result into
    LIVE_STATE. Never raises into the caller -- on error it records the message into
    LIVE_STATE['error'] and switches the camera off, so a camera hiccup surfaces on the page
    instead of silently killing the server.

    Stops on a frame that still has unnamed objects (see LIVE_FROZEN), and runs no pipeline at
    all while Module 3 has the camera to itself (QWEN_SOLO) -- then it only refreshes the
    plain camera picture now and then (SOLO_PREVIEW_INTERVAL)."""
    global LIVE_FROZEN
    predictor = get_predictor()
    last_done_t = None
    fps = None
    try:
        while True:
            if QWEN_SOLO.is_set():  # pipeline paused: the page shows the plain camera picture
                im0, _depth, _seq = grab_frame()
                raw_preview = encode_jpg(im0, max_dim=960, quality=80)
                with LIVE_LOCK:
                    LIVE_STATE.update({"ready": True, "error": None, "image": raw_preview,
                                       "image_ood": raw_preview, "solo": True,
                                       "resolution": f"{im0.shape[1]}x{im0.shape[0]}"})
                time.sleep(SOLO_PREVIEW_INTERVAL)
                continue
            if hybrid_active():  # Part 3 instead of Module 2 (a stopped frame waits meanwhile)
                hybrid_step(predictor)
                continue
            with LIVE_LOCK:
                stopped = LIVE_FROZEN is not None
                if stopped and assistant_active():
                    # Go live again for the /assistant page; the unnamed objects stay on the
                    # operator view's list (LIVE_UNKNOWN) to be named later.
                    for key, item in LIVE_FROZEN["pending"].items():
                        LIVE_UNKNOWN[key] = item
                        LIVE_UNKNOWN.move_to_end(key)
                    LIVE_FROZEN = None
                    LIVE_STATE.update({"frozen": False, "pending": list(LIVE_UNKNOWN.values())})
                    stopped = False
            if stopped:
                time.sleep(0.05)
                continue
            # Detector + VLM stages hold the GPU for one turn (see GPU_LOCK). The frame is
            # grabbed only once the turn is ours, so after waiting out a Qwen-direct pass we
            # work on the current frame, not one from several seconds ago; the timers start
            # after the grab, so neither the wait nor the grab is billed here.
            with GPU_LOCK:
                im0, depth_info, seq = grab_frame()
                with LIVE_LOCK:
                    first_pass = LIVE_STATE["frame_count"] == 0
                if first_pass:
                    # The first pass is the slow one (first CUDA kernels, first VLM stage-2
                    # escalation), so show the plain camera picture meanwhile instead of a page
                    # stuck on "waiting for the first frame".
                    raw_preview = encode_jpg(im0, max_dim=960, quality=80)
                    with LIVE_LOCK:
                        LIVE_STATE.update({"ready": True, "error": None, "image": raw_preview,
                                           "image_ood": raw_preview, "warming": True,
                                           "resolution": f"{im0.shape[1]}x{im0.shape[0]}"})
                t0 = time.perf_counter()
                image_tensor, letterboxed_shape = ri.preprocess(im0)
                with torch.no_grad():
                    results = ri.run_inference(predictor, image_tensor, letterboxed_shape)
                boxes, labels, scores = ri.postprocess_energy(results, ARGS.energy_thres, ARGS.conf_thres)
                detect_ms = (time.perf_counter() - t0) * 1000

                ood_labels_by_row = {}
                flush_cache_on_motion("live", im0, CACHE_LIVE)
                if len(boxes):
                    boxes = ri.scale_boxes(letterboxed_shape, boxes, im0.shape).round()
                    ood_idx = (labels == ri.OOD_CLASS_ID).nonzero(as_tuple=True)[0]
                    if ood_idx.numel():
                        descriptions = ri.label_ood_detections(
                            im0, boxes[ood_idx], CACHE_LIVE, ARGS.verify_model_id, human_qa_enabled=False,
                            memory=MEMORY,
                        )
                        ood_labels_by_row = {int(row): desc for row, desc in zip(ood_idx.tolist(), descriptions)}
                frame_ms = (time.perf_counter() - t0) * 1000

            seen_unknown = []
            for row, (label, _src) in ood_labels_by_row.items():
                if ri._is_unknown(label):
                    x1, y1, x2, y2 = clip_box(im0, boxes[row])
                    crop = im0[y1:y2, x1:x2]
                    seen_unknown.append({
                        "key": cache_key_to_str(ri.ood_cache_key(boxes[row])),
                        "box": [x1, y1, x2, y2],
                        "crop": encode_jpg(crop, max_dim=220) if crop.size else None,
                        "row": int(row),
                    })

            preview = ri.annotate(im0.copy(), boxes, labels, scores, ood_labels_by_row)
            # Same frame, same boxes, annotated with an empty ood_labels_by_row -- every OOD
            # box falls back to annotate()'s "unknown" default (red, no name) instead of the
            # VLM's guess, so this is the raw energy-based OOD/ID split before any VLM call.
            # Lets the page show that split side by side with the final VLM-labeled view.
            ood_preview = ri.annotate(im0.copy(), boxes, labels, scores, {})
            stats, resolved_labels, known_labels = stats_and_labels(labels, ood_labels_by_row, boxes)
            preview_b64 = encode_jpg(preview)
            now = time.perf_counter()
            fps = update_fps(fps, last_done_t, now)
            last_done_t = now

            # Side by side with Qwen-direct, keep streaming: a stopped pipeline would leave the
            # comparison with only one side moving (unknowns still collect in LIVE_UNKNOWN).
            comparing = QWEN_LIVE_ENABLED.is_set() and not QWEN_SOLO.is_set()
            with LIVE_LOCK:
                if seen_unknown and not ARGS.no_freeze and not comparing and not assistant_active():
                    # Stop on this frame: the page keeps showing it, and asks about exactly these
                    # objects, until every one is named or dismissed (api_live_label).
                    LIVE_FROZEN = {
                        "im0": im0, "boxes": boxes, "labels": labels, "scores": scores,
                        "ood_labels_by_row": dict(ood_labels_by_row),
                        "pending": OrderedDict((it["key"], it) for it in seen_unknown),
                    }
                    pending = list(LIVE_FROZEN["pending"].values())
                else:
                    # Add/refresh this frame's still-unknown objects into the running list --
                    # never drop an entry here just because it's absent from *this* frame; see
                    # LIVE_UNKNOWN's docstring above for why. Only api_live_label removes one.
                    for item in seen_unknown:
                        LIVE_UNKNOWN[item["key"]] = item
                        LIVE_UNKNOWN.move_to_end(item["key"])
                    pending = list(LIVE_UNKNOWN.values())
                frame_count = LIVE_STATE["frame_count"] + 1
                LIVE_STATE.update({
                    "ready": True, "error": None,
                    "image": preview_b64, "image_ood": encode_jpg(ood_preview),
                    "resolution": f"{im0.shape[1]}x{im0.shape[0]}",
                    "stats": stats, "labels": resolved_labels, "known_labels": known_labels,
                    "pending": pending, "frozen": LIVE_FROZEN is not None, "solo": False,
                    "warming": False, "frame_count": frame_count,
                    "timing": {
                        "detect_ms": round(detect_ms, 1), "vlm_ms": round(frame_ms - detect_ms, 1),
                        "frame_ms": round(frame_ms, 1), "fps": round(fps, 2) if fps else None,
                    },
                })
            compare_log("live", "pipeline", LIVE_SOURCE, seq, frame_ms, stats["total"],
                        pipeline_label_names(known_labels, resolved_labels))
            assistant_observe(im0, boxes, labels, scores, ood_labels_by_row, preview_b64, depth_info, LIVE_SOURCE)
    except CameraPaused:
        return
    except Exception as e:
        print(f"[robot_web_app] live loop stopped: {e}")
        with LIVE_LOCK:
            LIVE_STATE["error"] = str(e)
        release_camera()  # switching it on again (hub / POST /api/camera) retries


def hybrid_step(predictor):
    """One frame of Part 3 (see HYBRID): detector + energy OOD split, then hybrid_pipeline does
    memory first and the VLM for the rest. Camera errors propagate to _run_live_loop; a failure
    of Part 3 itself only switches Part 3 off, with the message on its page."""
    global HYBRID_PIPELINE
    with GPU_LOCK:
        im0, depth_info, seq = grab_frame()
        try:
            if HYBRID_PIPELINE is None:
                with HYBRID_LOCK:
                    HYBRID_STATE["warming"] = "loading Module 1's memory and its embedder (DINOv2)…"
                import hybrid_pipeline
                memory = hybrid_pipeline.TaughtMemory(ARGS.hybrid_memory_dir, device=str(DEVICE),
                                                      accept_thres=ARGS.hybrid_accept_thres)
                HYBRID_PIPELINE = hybrid_pipeline.HybridPipeline(memory, ARGS.verify_model_id)
            t0 = time.perf_counter()
            image_tensor, letterboxed_shape = ri.preprocess(im0)
            with torch.no_grad():
                results = ri.run_inference(predictor, image_tensor, letterboxed_shape)
            boxes, labels, scores = ri.postprocess_energy(results, ARGS.energy_thres, ARGS.conf_thres)
            if len(boxes):
                boxes = ri.scale_boxes(letterboxed_shape, boxes, im0.shape).round()
            detect_ms = (time.perf_counter() - t0) * 1000
            flush_cache_on_motion("hybrid", im0, CACHE_HYBRID)
            result, ood_labels_by_row = HYBRID_PIPELINE.run(
                im0, boxes, labels, scores, CACHE_HYBRID,
                lambda img, max_dim: encode_jpg(img, max_dim=max_dim, quality=82))
        except Exception as e:
            print(f"[robot_web_app] Part 3 (OOD + memory + VLM) stopped: {e}")
            with HYBRID_LOCK:
                HYBRID["enabled"] = False
                HYBRID_STATE.update(error=str(e), warming=None)
            return
    timing = {"detect_ms": round(detect_ms, 1), **result["timing"]}
    timing["frame_ms"] = round(timing["detect_ms"] + timing["memory_ms"] + timing["vlm_ms"], 1)
    now = time.perf_counter()
    with HYBRID_LOCK:
        HYBRID["fps"] = update_fps(HYBRID["fps"], HYBRID["last_t"], now)
        HYBRID["last_t"] = now
        timing["fps"] = round(HYBRID["fps"], 2) if HYBRID["fps"] else None
        tot = HYBRID_STATE["totals"] or hybrid_totals()
        c = result["counts"]
        tot["frames"] += 1
        for k in ("ood", "memory", "vlm", "vlm_new", "unknown"):
            tot[k] += c[k]
        for k in ("detect_ms", "memory_ms", "vlm_ms"):
            tot[k] += timing[k]
        HYBRID_STATE.update(result)
        HYBRID_STATE.update({"ready": True, "error": None, "warming": None, "seq": seq, "totals": tot,
                             "frame_count": HYBRID_STATE["frame_count"] + 1, "timing": timing,
                             "resolution": f"{im0.shape[1]}x{im0.shape[0]}"})
    assistant_observe(im0, boxes, labels, scores, ood_labels_by_row, result["images"]["final"], depth_info, LIVE_SOURCE)


@app.route("/")
def index():
    return send_from_directory(PAGE_PATH.parent, PAGE_PATH.name)


@app.route("/api/sample-images")
def api_sample_images():
    out = []
    for name, entry in SAMPLES_BY_NAME.items():
        item = {"name": name, "kind": entry["kind"]}
        if entry["kind"] == "video":
            item["frames"] = entry["frames"]
        out.append(item)
    return jsonify(out)


@app.route("/api/live/status")
def api_live_status():
    return jsonify({
        "enabled": LIVE_SOURCE is not None, "source": LIVE_SOURCE, "topic": ARGS.ros_topic,
        "qwen_direct_model": qwen_direct_model_id(),
    })


@app.route("/api/camera", methods=["GET", "POST"])
def api_camera():
    """{"on": bool}: take or release the camera, models stay loaded. Releasing answers once the
    RealSense is closed, so the caller can hand it to another module right away."""
    if LIVE_SOURCE is None:
        return jsonify({"error": "server was not started with --ros-topic or --realsense"}), 400
    if request.method == "POST":
        if (request.get_json(silent=True) or {}).get("on"):
            with LIVE_LOCK:
                LIVE_STATE["error"] = None
            CAMERA_ON.set()
        else:
            release_camera()
    return jsonify({"camera": CAMERA["state"], "wanted": CAMERA_ON.is_set()})


@app.route("/api/live/latest")
def api_live_latest():
    if LIVE_SOURCE is None:
        return jsonify({"error": "server was not started with --ros-topic or --realsense"}), 400
    with LIVE_LOCK:
        snapshot = dict(LIVE_STATE)
    if snapshot["error"]:
        return jsonify({"error": snapshot["error"]}), 500
    if not snapshot["ready"]:
        return jsonify({"ready": False, "camera": CAMERA["state"]})
    snapshot["energy_thres"] = ARGS.energy_thres
    snapshot["camera"] = CAMERA["state"]
    return jsonify(snapshot)


@app.route("/api/live/label", methods=["POST"])
def api_live_label():
    if LIVE_SOURCE is None:
        return jsonify({"error": "server was not started with --ros-topic or --realsense"}), 400
    payload = request.get_json(silent=True) or {}
    key_str = payload.get("key")
    if not key_str:
        return jsonify({"error": "missing key"}), 400
    try:
        key = cache_key_from_str(key_str)
    except ValueError:
        return jsonify({"error": "malformed key"}), 400

    dismiss = bool(payload.get("dismiss"))
    typed = "" if dismiss else (payload.get("label") or "").strip().lower()
    if dismiss:
        # Operator dismissed it without naming it -- cache a sentinel (not "unknown") so this
        # exact spot doesn't immediately get re-queried by the VLM or bounce right back onto
        # the pending list next frame. Deliberately never touches MEMORY: dismissing isn't a name.
        CACHE_LIVE[key] = ("dismissed", "dismissed")
    else:
        CACHE_LIVE[key] = (typed, "human") if typed else ("unknown", "vlm2")
    CACHE_LIVE.move_to_end(key)
    if len(CACHE_LIVE) > ri.MAX_CACHE_ENTRIES:
        CACHE_LIVE.popitem(last=False)
    if typed or dismiss:
        # A real name or an explicit dismissal both clear the object from the persistent
        # unknown list -- leaving the input blank does neither: it still caches "unknown" (so
        # the loop stops re-querying the VLM for it), but the object stays visible in the QA
        # list until someone actually names or dismisses it.
        with LIVE_LOCK:
            entry = LIVE_UNKNOWN.pop(key_str, None)
            frozen_entry = LIVE_FROZEN["pending"].pop(key_str, None) if LIVE_FROZEN is not None else None
        entry = entry or frozen_entry
        if frozen_entry is not None:
            update_frozen_view(frozen_entry["row"], ("dismissed", "dismissed") if dismiss else (typed, "human"))
        if typed and MEMORY is not None and entry:
            crop = decode_jpg_b64(entry.get("crop"))
            if crop is not None:
                MEMORY.remember(crop, typed)
    return jsonify({"ok": True})


def update_frozen_view(row, resolved):
    """Draws a just-given name (or a dismissal) onto the frame the live view stopped on, and
    goes live again once nothing on it is left unnamed."""
    global LIVE_FROZEN
    with LIVE_LOCK:
        fz = LIVE_FROZEN
        if fz is None:
            return
        fz["ood_labels_by_row"][row] = resolved
        pending = list(fz["pending"].values())
        if not pending:
            LIVE_FROZEN = None
            LIVE_STATE.update({"frozen": False, "pending": list(LIVE_UNKNOWN.values())})
            return
        im0, boxes, labels, scores, ood = fz["im0"], fz["boxes"], fz["labels"], fz["scores"], dict(fz["ood_labels_by_row"])
    preview = encode_jpg(ri.annotate(im0.copy(), boxes, labels, scores, ood))
    stats, resolved_labels, known_labels = stats_and_labels(labels, ood, boxes)
    with LIVE_LOCK:
        if LIVE_FROZEN is fz:
            LIVE_STATE.update({"image": preview, "stats": stats, "labels": resolved_labels,
                               "known_labels": known_labels, "pending": pending})


@app.route("/api/pushed/latest")
def api_pushed_latest():
    with PUSHED_LOCK:
        snapshot = dict(PUSHED_STATE)
    if not snapshot["ready"]:
        return jsonify({"ready": False})
    return jsonify(snapshot)


@app.route("/api/pushed/label", methods=["POST"])
def api_pushed_label():
    """Same shape as /api/live/label, resolving by spatial key straight into the shared
    CACHE (not job_id/row -- a pushed frame's job_id can fall out of STATE's bounded LRU
    long before the operator gets to naming it, unlike the manual demo's /api/submit)."""
    payload = request.get_json(silent=True) or {}
    key_str = payload.get("key")
    if not key_str:
        return jsonify({"error": "missing key"}), 400
    try:
        key = cache_key_from_str(key_str)
    except ValueError:
        return jsonify({"error": "malformed key"}), 400

    dismiss = bool(payload.get("dismiss"))
    typed = "" if dismiss else (payload.get("label") or "").strip().lower()
    if dismiss:
        # See api_live_label's identical comment: sentinel, not "unknown", and never MEMORY.
        CACHE[key] = ("dismissed", "dismissed")
    else:
        CACHE[key] = (typed, "human") if typed else ("unknown", "vlm2")
    CACHE.move_to_end(key)
    if len(CACHE) > ri.MAX_CACHE_ENTRIES:
        CACHE.popitem(last=False)
    if typed or dismiss:
        # A real name or an explicit dismissal both clear it from the persistent unknown
        # list; see PUSHED_UNKNOWN. Leaving the input blank does neither.
        with PUSHED_LOCK:
            entry = PUSHED_UNKNOWN.pop(key_str, None)
        if typed and MEMORY is not None and entry:
            crop = decode_jpg_b64(entry.get("crop"))
            if crop is not None:
                MEMORY.remember(crop, typed)
    return jsonify({"ok": True})


@app.route("/api/memory/list")
def api_memory_list():
    if MEMORY is None:
        return jsonify({"enabled": False, "entries": []})
    return jsonify({"enabled": True, "entries": MEMORY.list_entries()})


@app.route("/api/memory/delete", methods=["POST"])
def api_memory_delete():
    """Deletes directly from the live MEMORY singleton (not just the JSON file), so a
    deletion here takes effect immediately -- no server restart needed, unlike editing
    ood_memory.json by hand while the server is running."""
    if MEMORY is None:
        return jsonify({"error": "memory is disabled (--no-memory)"}), 400
    payload = request.get_json(silent=True) or {}
    index = payload.get("index")
    name = payload.get("name")
    if index is not None:
        indices = [int(index)]
    elif name:
        target = (name or "").strip().lower()
        indices = [e["index"] for e in MEMORY.list_entries() if e["name"] == target]
    else:
        return jsonify({"error": "must provide index or name"}), 400
    removed = MEMORY.forget(indices)
    return jsonify({"removed": removed, "entries": MEMORY.list_entries()})


# --- Robot assistant ("Robot, where are my headphones?") -------------------------
@app.route("/assistant")
def assistant_page():
    return send_from_directory(ASSISTANT_PAGE_PATH.parent, ASSISTANT_PAGE_PATH.name)


@app.route("/api/assistant/ask", methods=["POST"])
def api_assistant_ask():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "missing text"}), 400
    return jsonify(ASSISTANT.ask(text))


@app.route("/api/assistant/state")
def api_assistant_state():
    ASSISTANT_POLL["last"] = time.time()
    return jsonify(ASSISTANT.state(request.args.get("since", default=0, type=int)))


@app.route("/api/assistant/snapshot/<path:key>")
def api_assistant_snapshot(key):
    jpg = ASSISTANT.scene.snapshot_jpg(key)
    if jpg is None:
        return jsonify({"error": f"haven't seen {key!r}"}), 404
    return Response(jpg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.route("/api/assistant/cancel", methods=["POST"])
def api_assistant_cancel():
    payload = request.get_json(silent=True) or {}
    return jsonify(ASSISTANT.cancel(payload.get("target")))


def load_request_frame():
    """Decodes the frame an /api/run-style request asks for: an uploaded file (multipart
    `image`, plus robot_client.py's `live`/`skip_vlm` form flags), a sample image, or one
    frame of a sample video (JSON `sample` + optional `frame`). Shared by /api/run and
    /api/qwen-direct/run so both answer for exactly the same frame. Returns (frame, None)
    -- frame a dict of im0/source_name/frame_idx/frame_count/is_live/skip_vlm -- or
    (None, error_response) for the route to return as-is."""
    frame = {"frame_idx": None, "frame_count": None, "is_live": False, "skip_vlm": False}

    if "image" in request.files and request.files["image"].filename:
        f = request.files["image"]
        data = np.frombuffer(f.read(), np.uint8)
        frame["im0"] = cv2.imdecode(data, cv2.IMREAD_COLOR)
        frame["source_name"] = f.filename
        frame["is_live"] = request.form.get("live") == "1"
        frame["skip_vlm"] = request.form.get("skip_vlm") == "1"
    else:
        payload = request.get_json(silent=True) or {}
        name = payload.get("sample")
        entry = SAMPLES_BY_NAME.get(name)
        if entry is None:
            return None, (jsonify({"error": f"unknown sample: {name!r}"}), 400)

        if entry["kind"] == "video":
            frame_count = entry["frames"]
            requested = payload.get("frame")
            frame_idx = frame_count // 2 if requested is None else int(requested)
            frame_idx = max(0, min(frame_count - 1, frame_idx))
            cap = cv2.VideoCapture(str(entry["path"]))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, im0 = cap.read()
            finally:
                cap.release()
            frame.update({
                "im0": im0 if ok else None, "frame_idx": frame_idx, "frame_count": frame_count,
                "source_name": f"{name} — frame {frame_idx}/{frame_count - 1}",
            })
        else:
            frame["im0"] = cv2.imread(str(entry["path"]))
            frame["source_name"] = name

    if frame["im0"] is None:
        return None, (jsonify({"error": "could not decode image/frame"}), 400)
    return frame, None


@app.route("/api/qwen-direct/run", methods=["POST"])
def api_qwen_direct_run():
    """The Qwen-direct baseline on one sample/upload -- same request shape as /api/run."""
    frame, error = load_request_frame()
    if error:
        return error
    im0 = frame["im0"]
    try:
        detections, raw_text, infer_ms = run_qwen_direct(im0)
    except Exception as e:  # e.g. vLLM server unreachable -- show it on the page, not as an HTML 500
        return jsonify({"error": f"Qwen direct failed: {e}"}), 500
    compare_log("image", "qwen", frame["source_name"], frame["frame_idx"], infer_ms, len(detections),
                [det["label"] for det in detections])
    return jsonify({
        "image": encode_jpg(annotate_qwen_direct(im0.copy(), detections)),
        "source": frame["source_name"],
        "resolution": f"{im0.shape[1]}x{im0.shape[0]}",
        "frame_idx": frame["frame_idx"],
        "frame_count": frame["frame_count"],
        "labels": qwen_direct_labels(detections),
        "count": len(detections),
        "timing": {"infer_ms": round(infer_ms, 1)},
        "raw_text": raw_text,
        "model_id": qwen_direct_model_id(),
    })


@app.route("/qwen-direct")
def qwen_direct_page():
    """Module 3's own page: the Qwen-direct baseline running alone on the live camera."""
    return send_from_directory(MODULE3_PAGE_PATH.parent, MODULE3_PAGE_PATH.name)


@app.route("/api/qwen-direct/live", methods=["POST"])
def api_qwen_direct_live():
    """{"enabled": bool, "solo": bool}. solo = Module 3 on its own: the detector + VLM pipeline
    pauses (see QWEN_SOLO) so the Qwen-direct baseline gets the GPU to itself. Without solo
    it's the Module 2 page's side-by-side comparison."""
    global LIVE_FROZEN
    if LIVE_SOURCE is None:
        return jsonify({"error": "server was not started with --ros-topic or --realsense"}), 400
    payload = request.get_json(silent=True) or {}
    if payload.get("enabled") and (payload.get("solo") or not QWEN_LIVE_ENABLED.is_set()):
        # Module 3 has nothing to name, and the comparison keeps the pipeline streaming (see
        # live_loop): drop a stopped frame; its unknown objects stay in the to-name list.
        with LIVE_LOCK:
            if LIVE_FROZEN is not None:
                for item in LIVE_FROZEN["pending"].values():
                    LIVE_UNKNOWN[item["key"]] = item
                LIVE_FROZEN = None
                LIVE_STATE.update({"frozen": False, "pending": list(LIVE_UNKNOWN.values())})
    if payload.get("enabled") and payload.get("solo"):
        QWEN_SOLO.set()
    else:
        QWEN_SOLO.clear()
    if payload.get("enabled"):
        if not QWEN_LIVE_ENABLED.is_set():
            # A fresh comparison: drop the previous one's last frame/timing so the page
            # doesn't show a stale result, and bump `session` so qwen_live_loop restarts its
            # fps average instead of counting the time it sat switched off.
            with QWEN_LIVE_LOCK:
                QWEN_LIVE_STATE.update({
                    "ready": False, "error": None, "image": None, "timing": None,
                    "session": QWEN_LIVE_STATE["session"] + 1,
                })
        QWEN_LIVE_ENABLED.set()
    else:
        QWEN_LIVE_ENABLED.clear()
    return jsonify({"enabled": QWEN_LIVE_ENABLED.is_set(), "solo": QWEN_SOLO.is_set()})


@app.route("/api/qwen-direct/live/latest")
def api_qwen_direct_live_latest():
    """?light=1 leaves out the image and the raw model reply (for status polling)."""
    if LIVE_SOURCE is None:
        return jsonify({"error": "server was not started with --ros-topic or --realsense"}), 400
    with QWEN_LIVE_LOCK:
        snapshot = dict(QWEN_LIVE_STATE)
    snapshot["enabled"] = QWEN_LIVE_ENABLED.is_set()
    snapshot["solo"] = QWEN_SOLO.is_set()
    if request.args.get("light"):
        snapshot.pop("image", None)
        snapshot.pop("raw_text", None)
    return jsonify(snapshot)


@app.route("/hybrid")
def hybrid_page():
    """Part 3's page: OOD detection, then Module 1's memory, then the VLM for the rest."""
    return send_from_directory(HYBRID_PAGE_PATH.parent, HYBRID_PAGE_PATH.name)


@app.route("/api/hybrid", methods=["POST"])
def api_hybrid():
    """{"enabled": bool}: the /hybrid page switches Part 3 on when it opens and off when it
    closes (navigator.sendBeacon, so the body may come as text/plain). {"reset": true} starts
    the session totals over."""
    if LIVE_SOURCE is None:
        return jsonify({"error": "server was not started with --ros-topic or --realsense"}), 400
    payload = request.get_json(silent=True, force=True) or {}
    with HYBRID_LOCK:
        if payload.get("reset"):
            HYBRID_STATE["totals"] = hybrid_totals()
        if "enabled" in payload:
            if payload["enabled"] and not HYBRID["enabled"]:
                HYBRID.update(fps=None, last_t=None)
                HYBRID_STATE.update(error=None, totals=hybrid_totals())
            HYBRID["enabled"] = bool(payload["enabled"])
            HYBRID["last_poll"] = time.time()
    if payload.get("enabled"):
        # Part 3 has the pipeline to itself: Module 3 and the side-by-side comparison stop.
        QWEN_SOLO.clear()
        QWEN_LIVE_ENABLED.clear()
    return jsonify({"enabled": HYBRID["enabled"]})


@app.route("/api/hybrid/latest")
def api_hybrid_latest():
    """Polled by the /hybrid page, which also keeps Part 3 switched on (see HYBRID_TIMEOUT_S).
    ?since=<seq>: if that is still the newest frame, leaves out the images and objects."""
    if LIVE_SOURCE is None:
        return jsonify({"error": "server was not started with --ros-topic or --realsense"}), 400
    since = request.args.get("since", default=-1, type=int)
    with HYBRID_LOCK:
        if HYBRID["enabled"]:
            HYBRID["last_poll"] = time.time()
        snapshot = dict(HYBRID_STATE)
        enabled = HYBRID["enabled"]
    if snapshot["seq"] == since:
        for k in ("images", "objects", "known"):
            snapshot.pop(k, None)
        snapshot["unchanged"] = True
    snapshot.update(enabled=enabled, camera=CAMERA["state"], module3=QWEN_SOLO.is_set(),
                    energy_thres=ARGS.energy_thres, conf_thres=ARGS.conf_thres, source=LIVE_SOURCE,
                    vlm_model=os.environ.get("QWEN_MODEL_ID"), verify_model=ARGS.verify_model_id)
    return jsonify(snapshot)


@app.route("/api/hybrid/memory")
def api_hybrid_memory():
    """The objects of Module 1's memory, as Part 3 last read it (one view each, for the page)."""
    mem = HYBRID_PIPELINE.memory if HYBRID_PIPELINE is not None else None
    return jsonify({"version": mem.version if mem else 0, "objects": mem.objects() if mem else []})


@app.route("/api/hybrid/memory/view/<path:file>")
def api_hybrid_memory_view(file):
    return send_from_directory(ARGS.hybrid_memory_dir, file)


@app.route("/api/compare/metrics")
def api_compare_metrics():
    with COMPARE_LOCK:
        n_rows = len(COMPARE_LOG)
    return jsonify({"methods": COMPARE_METHODS, "summary": compare_summary(), "rows": n_rows})


@app.route("/api/compare/metrics.csv")
def api_compare_metrics_csv():
    with COMPARE_LOCK:
        rows = list(COMPARE_LOG)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["time", "mode", "method", "source", "frame", "ms", "objects", "labels"])
    for r in rows:
        writer.writerow([r["time"], r["mode"], COMPARE_METHODS[r["method"]], r["source"], r["frame"], r["ms"],
                         r["objects"], "; ".join(r["labels"])])
    name = f"ood_vlm_vs_vlm_only_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    return Response("﻿" + buf.getvalue(), mimetype="text/csv",  # BOM: Excel reads it as UTF-8
                    headers={"Content-Disposition": f"attachment; filename={name}"})


@app.route("/api/compare/metrics/reset", methods=["POST"])
def api_compare_metrics_reset():
    with COMPARE_LOCK:
        COMPARE_LOG.clear()
    return jsonify({"ok": True})


@app.route("/api/run", methods=["POST"])
def api_run():
    frame, error = load_request_frame()
    if error:
        return error
    im0, source_name = frame["im0"], frame["source_name"]
    frame_idx, frame_count = frame["frame_idx"], frame["frame_count"]
    is_live, skip_vlm = frame["is_live"], frame["skip_vlm"]
    depth_info = parse_depth_upload() if is_live else None  # robot_client.py --depth

    ood_labels_by_row = {}
    pending_rows = []
    per_object_timing = []
    vlm_ms = 0.0
    # One GPU turn for the detector + VLM stages (see GPU_LOCK); timers start inside it.
    with GPU_LOCK:
        t_detect0 = time.perf_counter()
        image_tensor, letterboxed_shape = ri.preprocess(im0)
        with torch.no_grad():
            results = ri.run_inference(get_predictor(), image_tensor, letterboxed_shape)
        boxes, labels, scores = ri.postprocess_energy(results, ARGS.energy_thres, ARGS.conf_thres)
        detect_ms = (time.perf_counter() - t_detect0) * 1000  # preprocess + detector fwd pass + energy split

        if is_live:
            flush_cache_on_motion("pushed", im0, CACHE)
        if len(boxes):
            boxes = ri.scale_boxes(letterboxed_shape, boxes, im0.shape).round()
            ood_idx = (labels == ri.OOD_CLASS_ID).nonzero(as_tuple=True)[0]
            if ood_idx.numel() and skip_vlm:
                # Detector + energy-OOD split only -- no VLM calls, no pending/QA. Just labels
                # every OOD box "ood" so the stream stays fast and simple to watch; --skip-vlm
                # on robot_client.py sends this.
                ood_labels_by_row = {int(row): ("ood", "vlm1") for row in ood_idx.tolist()}
            elif ood_idx.numel() and is_live:
                # Live push path (robot_client.py): use the real batched production pipeline
                # (one VLM call per frame covering every OOD object in it, same as the
                # --ros-topic live loop below) instead of label_ood_detections_timed's
                # one-object-at-a-time calls. The per-object timing breakdown is a demo/debug
                # feature for manual sample-image runs in the browser -- nothing reads it for
                # pushed frames (robot_client.py only prints detect_ms/vlm_total_ms) -- so it
                # was pure overhead here: N sequential generate() calls instead of one batched
                # call, each paying its own tokenization + CUDA-launch + KV-cache-alloc cost.
                t_vlm0 = time.perf_counter()
                descriptions = ri.label_ood_detections(
                    im0, boxes[ood_idx], CACHE, ARGS.verify_model_id, human_qa_enabled=False,
                    memory=MEMORY,
                )
                vlm_ms = (time.perf_counter() - t_vlm0) * 1000
                ood_labels_by_row = {int(row): desc for row, desc in zip(ood_idx.tolist(), descriptions)}
                pending_rows = [row for row, (label, _src) in ood_labels_by_row.items() if ri._is_unknown(label)]
            elif ood_idx.numel():
                # Manual/demo path (sample-image button clicks, not a robot_client.py push):
                # stages 1 + 2 only, stage 3 (still unknown) comes back to the caller instead
                # of blocking on cv2/stdin -- same resolution logic as ri.label_ood_detections
                # with human_qa_enabled=False, but timed per-object (see the function
                # docstring) so the demo page can show a real single-query latency number.
                t_vlm0 = time.perf_counter()
                descriptions, timings = label_ood_detections_timed(
                    im0, boxes[ood_idx], CACHE, ARGS.verify_model_id, memory=MEMORY,
                )
                vlm_ms = (time.perf_counter() - t_vlm0) * 1000
                ood_labels_by_row = {int(row): desc for row, desc in zip(ood_idx.tolist(), descriptions)}
                pending_rows = [row for row, (label, _src) in ood_labels_by_row.items() if ri._is_unknown(label)]
                for row, desc, timing in zip(ood_idx.tolist(), descriptions, timings):
                    per_object_timing.append({
                        "row": int(row), "label": desc[0], "source": desc[1], **timing,
                    })

    job_id = uuid.uuid4().hex
    STATE[job_id] = {
        "im0": im0, "boxes": boxes, "labels": labels, "scores": scores,
        "ood_labels_by_row": ood_labels_by_row,
    }
    if len(STATE) > MAX_STATE_ENTRIES:
        STATE.popitem(last=False)

    # Crop thumbnails from the raw frame before annotating: Annotator draws onto
    # its input array in place, so annotating im0 itself would bake box/label
    # overlays into any crop taken afterwards (and into the frame this job keeps
    # for /api/submit to re-annotate later). Give annotate() a throwaway copy.
    pending = []
    for row in pending_rows:
        x1, y1, x2, y2 = clip_box(im0, boxes[row])
        crop = im0[y1:y2, x1:x2]
        pending.append({
            "row": row,
            "key": cache_key_to_str(ri.ood_cache_key(boxes[row])),
            "box": [x1, y1, x2, y2],
            "crop": encode_jpg(crop, max_dim=220) if crop.size else None,
        })

    preview = ri.annotate(im0.copy(), boxes, labels, scores, ood_labels_by_row)
    # Same frame, same boxes, annotated with an empty ood_labels_by_row -- every OOD box
    # falls back to annotate()'s "unknown" default (red, no name) instead of the VLM's
    # guess, so this is the raw energy-based OOD/ID split before any VLM call. Lets the
    # page show that split side by side with the final VLM-labeled view above.
    ood_preview = ri.annotate(im0.copy(), boxes, labels, scores, {})
    stats, resolved_labels, known_labels = stats_and_labels(labels, ood_labels_by_row, boxes)
    image_b64 = encode_jpg(preview)
    image_ood_b64 = encode_jpg(ood_preview)

    if is_live:
        with PUSHED_LOCK:
            # Same accumulate-until-named pattern as LIVE_UNKNOWN/live_loop above,
            # keyed by spatial position instead of this frame's job_id -- every pushed
            # frame gets a fresh job_id, so keying off it would wipe the list on every
            # single frame (that was the bug: the QA list never accumulated).
            for item in pending:
                PUSHED_UNKNOWN[item["key"]] = {"key": item["key"], "box": item["box"], "crop": item["crop"]}
                PUSHED_UNKNOWN.move_to_end(item["key"])
            PUSHED_STATE.update({
                "ready": True, "job_id": job_id, "source": source_name, "image": image_b64,
                "image_ood": image_ood_b64,
                "resolution": f"{im0.shape[1]}x{im0.shape[0]}", "stats": stats,
                "labels": resolved_labels, "known_labels": known_labels,
                "pending": list(PUSHED_UNKNOWN.values()),
                "energy_thres": ARGS.energy_thres,
                "frame_count": PUSHED_STATE["frame_count"] + 1,
            })
        assistant_observe(im0, boxes, labels, scores, ood_labels_by_row, image_b64, depth_info,
                          f"robot_client.py ({source_name})")

    if not is_live:  # sample/upload runs only; robot_client.py pushes aren't part of the comparison
        compare_log("image", "pipeline", source_name, frame_idx, detect_ms + vlm_ms, stats["total"],
                    pipeline_label_names(known_labels, resolved_labels))
    n_queried =sum(1 for t in per_object_timing if not t["cached"])
    return jsonify({
        "job_id": job_id,
        "source": source_name,
        "image": image_b64,
        "image_ood": image_ood_b64,
        "resolution": f"{im0.shape[1]}x{im0.shape[0]}",
        "energy_thres": ARGS.energy_thres,
        "stats": stats,
        "labels": resolved_labels,
        "known_labels": known_labels,
        "pending": pending,
        "frame_idx": frame_idx,
        "frame_count": frame_count,
        "timing": {
            "detect_ms": round(detect_ms, 1),
            "vlm_total_ms": round(vlm_ms, 1),
            "image_total_ms": round(detect_ms + vlm_ms, 1),
            "vlm_avg_per_object_ms": round(vlm_ms / n_queried, 1) if n_queried else None,
            "per_object": [
                {k: (round(v, 1) if isinstance(v, float) else v) for k, v in t.items()}
                for t in per_object_timing
            ],
        },
    })


@app.route("/api/submit/<job_id>", methods=["POST"])
def api_submit(job_id):
    job = STATE.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job_id (server restarted since this run?)"}), 404

    payload = request.get_json(silent=True) or {}
    typed_by_row = payload.get("labels", {})
    ood_labels_by_row = job["ood_labels_by_row"]
    for row_str, typed in typed_by_row.items():
        row = int(row_str)
        typed = (typed or "").strip().lower()
        resolved = (typed, "human") if typed else ("unknown", "vlm2")
        ood_labels_by_row[row] = resolved
        # Also write into the shared spatial cache (same one label_ood_detections_timed
        # reads from), so this name resolves the same physical object in whatever frame
        # comes in next -- without this, a live push session (robot_client.py) would ask
        # about the exact same object again on every single frame.
        cache_key = ri.ood_cache_key(job["boxes"][row])
        CACHE[cache_key] = resolved
        CACHE.move_to_end(cache_key)
        if len(CACHE) > ri.MAX_CACHE_ENTRIES:
            CACHE.popitem(last=False)
        if typed and MEMORY is not None:
            x1, y1, x2, y2 = clip_box(job["im0"], job["boxes"][row])
            crop = job["im0"][y1:y2, x1:x2]
            if crop.size:
                MEMORY.remember(crop, typed)

    final = ri.annotate(job["im0"].copy(), job["boxes"], job["labels"], job["scores"], ood_labels_by_row)
    stats, resolved_labels, known_labels = stats_and_labels(job["labels"], ood_labels_by_row, job["boxes"])

    return jsonify({
        "image": encode_jpg(final), "stats": stats, "labels": resolved_labels, "known_labels": known_labels,
    })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ri.DEFAULT_WEIGHTS)
    parser.add_argument("--cfg", type=str, default=ri.DEFAULT_CFG)
    parser.add_argument("--conf-thres", type=float, default=0.5)
    parser.add_argument("--energy-thres", type=float, default=-6.84)
    parser.add_argument("--verify-model-id", type=str, default=ri.DEFAULT_VERIFY_MODEL_ID)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--ros-topic", type=str, default=None,
                        help="ROS 2 sensor_msgs/Image topic to run continuously in the background "
                             "(e.g. /camera/camera/color/image_raw for realsense-ros). When set, the "
                             "page grows a 'Live (ROS)' section -- open http://<host>:<port>/ from any "
                             "browser on the robot's network to watch the stream and resolve unknowns "
                             "without needing a display on the robot itself. Requires rclpy + cv_bridge "
                             "on this Python path (source your ROS 2 setup.bash first).")
    parser.add_argument("--realsense", nargs="?", const="", default=None, metavar="SERIAL",
                        help="stream a USB-attached Intel RealSense's color camera straight into the "
                             "page's live section via pyrealsense2 (no ROS needed). Optionally give a "
                             "serial number to pick one camera when several are plugged in. Ignored "
                             "if --ros-topic is also set.")
    parser.add_argument("--ros-qos", choices=["sensor", "default"], default="sensor",
                        help="QoS profile for --ros-topic; see robot_implementation.py --ros-qos.")
    parser.add_argument("--vllm-url", type=str, default=ri.DEFAULT_VLLM_URL,
                        help="base URL of a running vLLM OpenAI-compatible server (see "
                             "start_vllm_server.sh) used for stage-1 (the primary VLM) instead of "
                             "loading that checkpoint in-process -- see robot_implementation.py "
                             "--vllm-url. Set to \"\" or pass --no-vllm to load it locally instead.")
    parser.add_argument("--no-vllm", action="store_true",
                        help='equivalent to --vllm-url ""; load stage 1 locally instead of using a '
                             'vLLM server')
    parser.add_argument("--memory-path", type=str, default=str(FILE.parent / "ood_memory.json"),
                        help="JSON file persisting appearance-based memory of human-given names for "
                             "OOD objects (default: robot_code/ood_memory.json). Shared with "
                             "robot_implementation.py's default of the same name -- names learned via "
                             "one entry point help the other too, unless overridden. Delete this file "
                             "to reset the memory.")
    parser.add_argument("--memory-sim-thres", type=float, default=ood_memory.DEFAULT_SIM_THRESHOLD,
                        help="cosine-similarity threshold (0-1) for accepting a stored memory name as "
                             "a match before falling back to the VLM stages; see "
                             "robot_implementation.py --memory-sim-thres / ood_memory.OODMemory")
    parser.add_argument("--no-memory", action="store_true",
                        help="disable the appearance-based memory lookup entirely")
    parser.add_argument("--no-freeze", action="store_true",
                        help="live view: keep streaming when a frame has objects nobody could name, instead "
                             "of stopping on that frame until each one is named or dismissed")
    parser.add_argument("--qwen-direct-model-id", type=str, default=None,
                        help="checkpoint for the page's 'Qwen3-VL direct' baseline (whole image -> "
                             "boxes + labels, no detector). Default: the stage-1 model "
                             "(QWEN_MODEL_ID), which is already loaded, so no extra VRAM.")
    parser.add_argument("--qwen-direct-max-tokens", type=int, default=1024,
                        help="generation budget for one Qwen-direct grounding reply (~25 tokens "
                             "per object found); output cut off at this limit still keeps "
                             "every complete box")
    parser.add_argument("--camera-off", action="store_true",
                        help="with --realsense: load everything but don't open the camera until "
                             "POST /api/camera {\"on\": true} (the hub hands it between modules this way)")
    parser.add_argument("--depth", action="store_true",
                        help="with --realsense: also stream aligned depth, so the /assistant page can say "
                             "how far away and in which direction a found object is. (For a RealSense on "
                             "another machine, pass --depth to robot_client.py instead.)")
    parser.add_argument("--motion-flush-thres", type=float, default=10.0,
                        help="clear a live path's spatial label cache when consecutive frames differ by "
                             "more than this (mean abs. difference of 160x90 grayscale thumbnails, 0-255) "
                             "-- i.e. the robot moved, so screen-position cache keys no longer point at "
                             "the same objects. 0 disables.")
    parser.add_argument("--no-grounding", action="store_true",
                        help="/assistant: don't fall back to asking Qwen3-VL to find an object in the "
                             "whole frame when the detector hasn't seen it")
    parser.add_argument("--hybrid-memory-dir", type=str, default=str(FILE.parent / "teach_mode" / "teach_memory"),
                        help="/hybrid (Part 3): Module 1's memory folder, only read here (default: Module 1's "
                             "default --memory-dir)")
    parser.add_argument("--hybrid-accept-thres", type=float, default=0.62,
                        help="/hybrid: memory match threshold, the same as Module 1's --accept-thres")
    return parser.parse_args()


def main():
    global ARGS, MEMORY, LIVE_SOURCE, ASSISTANT, LIVE_FRAMES_FACTORY
    ARGS = parse_args()
    MEMORY = None if ARGS.no_memory else ood_memory.OODMemory(ARGS.memory_path, ARGS.memory_sim_thres)
    ASSISTANT = robot_assistant.Assistant(
        voc_names=ri.VOC_NAME_MAPPING, ood_class_id=ri.OOD_CLASS_ID, grounding=not ARGS.no_grounding,
        gpu_lock=GPU_LOCK,
    )

    # See robot_implementation.main() -- same env var, same "must be set before the first
    # lazy describe_images() import" requirement. The warmup call a few lines down is
    # that first import, so this has to happen before it.
    vllm_url = "" if ARGS.no_vllm else (ARGS.vllm_url or "")
    if vllm_url:
        os.environ["QWEN_VLLM_BASE_URL"] = vllm_url
        print(f"[robot_web_app] stage 1 will use the vLLM server at {vllm_url} "
              f"(start it with start_vllm_server.sh if you haven't; --no-vllm to load it locally instead)")
    else:
        os.environ.pop("QWEN_VLLM_BASE_URL", None)

    print(f"[robot_web_app] loading detector on {DEVICE} ...")
    get_predictor()

    # Force the primary VLM's module import + from_pretrained() to happen now, not
    # inside the first timed /api/run: `import transformers` alone can take several
    # seconds, and that cost would otherwise land entirely on whichever object
    # happens to be first in the first request, badly skewing its per-object timing.
    # (The bigger --verify-model-id stays lazy -- same tradeoff as the CLI script --
    # so its load cost will still show up as a one-time spike on the first stage-2
    # escalation of a session.)
    print("[robot_web_app] warming up primary VLM (first import + model load is slow, one-time only) ...")
    t0 = time.perf_counter()
    from OOD_object_name_generation.qwen3_generator import describe_images
    describe_images([Image.new("RGB", (64, 64))], max_new_tokens=1, batch_size=1)
    print(f"[robot_web_app] VLM warm ({(time.perf_counter() - t0) * 1000:.0f} ms). "
          f"Open http://{ARGS.host}:{ARGS.port}/ in a browser "
          f"(robot assistant: http://{ARGS.host}:{ARGS.port}/assistant).")

    if ARGS.ros_topic:
        LIVE_SOURCE = f"ROS topic: {ARGS.ros_topic}"
        LIVE_FRAMES_FACTORY = lambda: ri.ros_iter_frames(ARGS.ros_topic, ARGS.ros_qos)  # noqa: E731
    elif ARGS.realsense is not None:
        LIVE_SOURCE = "RealSense camera" + (f" {ARGS.realsense}" if ARGS.realsense else "")
        LIVE_FRAMES_FACTORY = lambda: ri.realsense_iter_frames(ARGS.realsense or None,  # noqa: E731
                                                               with_depth=ARGS.depth)
    if LIVE_SOURCE:
        if not ARGS.camera_off:
            CAMERA_ON.set()
        print(f"[robot_web_app] starting live loop on {LIVE_SOURCE} ...")
        threading.Thread(target=live_loop, daemon=True).start()
        # Idle until the page's "Compare with Qwen3-VL direct" toggle sets QWEN_LIVE_ENABLED.
        threading.Thread(target=qwen_live_loop, daemon=True).start()

    app.run(host=ARGS.host, port=ARGS.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
