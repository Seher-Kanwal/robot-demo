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
"""
import argparse
import base64
import os
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(FILE.parent) not in sys.path:
    sys.path.append(str(FILE.parent))

import robot_implementation as ri  # noqa: E402 (sets QWEN_MODEL_ID / sys.path on import)
import ood_memory  # noqa: E402

PAGE_PATH = FILE.parent / "live_feed_demo.html"

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Live (pushed frames / robot_client.py) mode -------------------------------
# Frames arriving at /api/run with live=1 (robot_client.py always sends this) also get
# stashed here, so the page can poll /api/pushed/latest and show whatever a remote
# client is streaming in -- no rclpy needed on this side at all (contrast the --ros-topic
# / ros_live_loop path above, which needs rclpy in this same process). Each pushed frame
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

# --- Live (--ros-topic) mode --------------------------------------------------
# A background thread (ros_live_loop) continuously pulls frames from a ROS 2
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
    "stats": None, "labels": [], "known_labels": [], "pending": [], "frame_count": 0,
}
# Every still-unknown object ever seen on the ROS live loop, keyed by its spatial cache
# key -- unlike LIVE_STATE["pending"] (which used to just be "whatever is unknown in the
# *current* frame"), an entry here is never dropped just because the object isn't in the
# latest frame (camera panned away, object occluded, detector missed it on one pass). It's
# only removed once the operator actually types a name for it in api_live_label, so the
# QA list on the page keeps accumulating instead of losing entries the operator hasn't
# had a chance to name yet.
LIVE_UNKNOWN = OrderedDict()


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


def ros_live_loop(topic, qos):
    """Runs forever on a background thread: pulls frames from a ROS 2 topic, runs the
    real pipeline (batched, unlike label_ood_detections_timed above) with stage 3
    disabled, and publishes each result into LIVE_STATE. Never raises into the caller
    -- on error it records the message into LIVE_STATE['error'] and stops, so a camera
    hiccup surfaces on the page instead of silently killing the server."""
    predictor = get_predictor()
    try:
        for im0, _is_last in ri.ros_iter_frames(topic, qos):
            image_tensor, letterboxed_shape = ri.preprocess(im0)
            with torch.no_grad():
                results = ri.run_inference(predictor, image_tensor, letterboxed_shape)
            boxes, labels, scores = ri.postprocess_energy(results, ARGS.energy_thres, ARGS.conf_thres)

            ood_labels_by_row = {}
            if len(boxes):
                boxes = ri.scale_boxes(letterboxed_shape, boxes, im0.shape).round()
                ood_idx = (labels == ri.OOD_CLASS_ID).nonzero(as_tuple=True)[0]
                if ood_idx.numel():
                    descriptions = ri.label_ood_detections(
                        im0, boxes[ood_idx], CACHE_LIVE, ARGS.verify_model_id, human_qa_enabled=False,
                        memory=MEMORY,
                    )
                    ood_labels_by_row = {int(row): desc for row, desc in zip(ood_idx.tolist(), descriptions)}

            seen_unknown = []
            for row, (label, _src) in ood_labels_by_row.items():
                if ri._is_unknown(label):
                    x1, y1, x2, y2 = clip_box(im0, boxes[row])
                    crop = im0[y1:y2, x1:x2]
                    seen_unknown.append({
                        "key": cache_key_to_str(ri.ood_cache_key(boxes[row])),
                        "box": [x1, y1, x2, y2],
                        "crop": encode_jpg(crop, max_dim=220) if crop.size else None,
                    })

            preview = ri.annotate(im0.copy(), boxes, labels, scores, ood_labels_by_row)
            # Same frame, same boxes, annotated with an empty ood_labels_by_row -- every OOD
            # box falls back to annotate()'s "unknown" default (red, no name) instead of the
            # VLM's guess, so this is the raw energy-based OOD/ID split before any VLM call.
            # Lets the page show that split side by side with the final VLM-labeled view.
            ood_preview = ri.annotate(im0.copy(), boxes, labels, scores, {})
            stats, resolved_labels, known_labels = stats_and_labels(labels, ood_labels_by_row, boxes)

            with LIVE_LOCK:
                # Add/refresh this frame's still-unknown objects into the running list --
                # never drop an entry here just because it's absent from *this* frame; see
                # LIVE_UNKNOWN's docstring above for why. Only api_live_label removes one.
                for item in seen_unknown:
                    LIVE_UNKNOWN[item["key"]] = item
                    LIVE_UNKNOWN.move_to_end(item["key"])
                LIVE_STATE.update({
                    "ready": True, "error": None,
                    "image": encode_jpg(preview), "image_ood": encode_jpg(ood_preview),
                    "resolution": f"{im0.shape[1]}x{im0.shape[0]}",
                    "stats": stats, "labels": resolved_labels, "known_labels": known_labels,
                    "pending": list(LIVE_UNKNOWN.values()),
                    "frame_count": LIVE_STATE["frame_count"] + 1,
                })
    except Exception as e:
        print(f"[robot_web_app] live ROS loop stopped: {e}")
        with LIVE_LOCK:
            LIVE_STATE["error"] = str(e)


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
    return jsonify({"enabled": ARGS.ros_topic is not None, "topic": ARGS.ros_topic})


@app.route("/api/live/latest")
def api_live_latest():
    if ARGS.ros_topic is None:
        return jsonify({"error": "server was not started with --ros-topic"}), 400
    with LIVE_LOCK:
        snapshot = dict(LIVE_STATE)
    if snapshot["error"]:
        return jsonify({"error": snapshot["error"]}), 500
    if not snapshot["ready"]:
        return jsonify({"ready": False})
    snapshot["energy_thres"] = ARGS.energy_thres
    return jsonify(snapshot)


@app.route("/api/live/label", methods=["POST"])
def api_live_label():
    if ARGS.ros_topic is None:
        return jsonify({"error": "server was not started with --ros-topic"}), 400
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
        if typed and MEMORY is not None and entry:
            crop = decode_jpg_b64(entry.get("crop"))
            if crop is not None:
                MEMORY.remember(crop, typed)
    return jsonify({"ok": True})


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


@app.route("/api/run", methods=["POST"])
def api_run():
    frame_idx = None
    frame_count = None
    is_live = False

    if "image" in request.files and request.files["image"].filename:
        f = request.files["image"]
        data = np.frombuffer(f.read(), np.uint8)
        im0 = cv2.imdecode(data, cv2.IMREAD_COLOR)
        source_name = f.filename
        is_live = request.form.get("live") == "1"
        skip_vlm = request.form.get("skip_vlm") == "1"
    else:
        skip_vlm = False
        payload = request.get_json(silent=True) or {}
        name = payload.get("sample")
        entry = SAMPLES_BY_NAME.get(name)
        if entry is None:
            return jsonify({"error": f"unknown sample: {name!r}"}), 400

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
            if not ok:
                im0 = None
            source_name = f"{name} — frame {frame_idx}/{frame_count - 1}"
        else:
            im0 = cv2.imread(str(entry["path"]))
            source_name = name

    if im0 is None:
        return jsonify({"error": "could not decode image/frame"}), 400

    t_detect0 = time.perf_counter()
    image_tensor, letterboxed_shape = ri.preprocess(im0)
    with torch.no_grad():
        results = ri.run_inference(get_predictor(), image_tensor, letterboxed_shape)
    boxes, labels, scores = ri.postprocess_energy(results, ARGS.energy_thres, ARGS.conf_thres)
    detect_ms = (time.perf_counter() - t_detect0) * 1000  # preprocess + detector fwd pass + energy split

    ood_labels_by_row = {}
    pending_rows = []
    per_object_timing = []
    vlm_ms = 0.0
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
            # Same accumulate-until-named pattern as LIVE_UNKNOWN/ros_live_loop above,
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

    n_queried = sum(1 for t in per_object_timing if not t["cached"])
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
    return parser.parse_args()


def main():
    global ARGS, MEMORY
    ARGS = parse_args()
    MEMORY = None if ARGS.no_memory else ood_memory.OODMemory(ARGS.memory_path, ARGS.memory_sim_thres)

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
          f"Open http://{ARGS.host}:{ARGS.port}/ in a browser.")

    if ARGS.ros_topic:
        print(f"[robot_web_app] starting live ROS loop on '{ARGS.ros_topic}' (qos={ARGS.ros_qos}) ...")
        threading.Thread(target=ros_live_loop, args=(ARGS.ros_topic, ARGS.ros_qos), daemon=True).start()

    app.run(host=ARGS.host, port=ARGS.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
