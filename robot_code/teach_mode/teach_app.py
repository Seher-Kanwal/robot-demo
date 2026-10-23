"""
Module 1 -- "Ask & Remember": teach the robot unfamiliar objects by asking the user.

    1. camera image  ->  2. OOD detector  ->  3. robot asks "What is this called?"
    ->  4. user captures multiple views  ->  5. memory (name + multi-view embeddings)
    ->  6. the object is moved  ->  7. "I recognize this object from memory. It is the TV remote."

No VLM is involved (that's Module 2, robot_web_app.py): anything the energy-based OOD
detector flags that isn't already in memory goes straight to the user.

Usage (one process owns the RealSense, so stop robot_web_app.py first):

    .venv/Scripts/python.exe robot_code/teach_mode/teach_app.py
    # then open http://127.0.0.1:7861/ in Chrome or Edge (needed for the microphone)

    # replay a recorded clip / a still image instead of the live camera:
    .venv/Scripts/python.exe robot_code/teach_mode/teach_app.py --source demo.mp4 --loop

    # start in explore mode (after teaching): only VOC classes + OOD objects named by memory,
    # and only OOD boxes the detector scores >= 0.95; no questions. Also switchable on the page.
    .venv/Scripts/python.exe robot_code/teach_mode/teach_app.py --mode explore

    # teach and replay side by side: a second, explore-mode view of the same camera frames and
    # the same memory at /recognize, so an object is recognized there right after it's taught
    .venv/Scripts/python.exe robot_code/teach_mode/teach_app.py --dual-view

Memory lives in robot_code/teach_mode/teach_memory/ (memory.json + view JPGs); delete that
folder to start from scratch, or use the Memory card on the page to edit it live.
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, request, send_from_directory

FILE = Path(__file__).resolve()
TEACH_DIR = FILE.parent
ROBOT_CODE = TEACH_DIR.parent
ROOT = ROBOT_CODE.parent
for _p in (TEACH_DIR, ROBOT_CODE):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import memory_backup  # noqa: E402
import robot_implementation as ri  # noqa: E402 (detector helpers; does not load any VLM)
from embedder import DEFAULT_EMBEDDER, EMBEDDERS, Embedder  # noqa: E402
from imgutils import encode_jpg  # noqa: E402
from multiview_memory import DEFAULT_ACCEPT_THRES, DEFAULT_MARGIN, MultiViewMemory  # noqa: E402
from teacher import Teacher  # noqa: E402
from unknown_tracker import UnknownTracker  # noqa: E402

PAGE_PATH = TEACH_DIR / "teach_demo.html"
RECOGNIZE_PAGE_PATH = TEACH_DIR / "recognize.html"
PERSON_CLASS_ID = 0  # VOC_NAME_MAPPING[0]

app = Flask(__name__)
ARGS = None
TEACHER = None
EXPLORER = None  # --dual-view: a second Teacher in explore mode on the same frames and memory
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOOP = {"running": False, "error": None, "source": None}


class FrameHub:
    """Latest annotated JPGs for the MJPEG endpoints; every viewer waits on the same
    condition instead of polling."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpgs = {"robot": None, "ood": None, "recognize": None}
        self.seq = 0

    def publish(self, robot_jpg, ood_jpg, recognize_jpg=None):
        with self.cond:
            self.jpgs = {"robot": robot_jpg, "ood": ood_jpg, "recognize": recognize_jpg}
            self.seq += 1
            self.cond.notify_all()

    def stream(self, which):
        last = -1
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.seq != last, timeout=5.0)
                if self.seq == last or self.jpgs[which] is None:
                    continue
                last, jpg = self.seq, self.jpgs[which]
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode()
                   + b"\r\n\r\n" + jpg + b"\r\n")


FRAMES = FrameHub()


# ============================================================================ camera loop
def detect(predictor, im0):
    """Steps 1-2: the same detector + energy split as robot_web_app.live_loop."""
    image_tensor, letterboxed_shape = ri.preprocess(im0)
    with torch.no_grad():
        results = ri.run_inference(predictor, image_tensor, letterboxed_shape)
    boxes, labels, scores = ri.postprocess_energy(results, ARGS.energy_thres, ARGS.conf_thres)
    if len(boxes):
        boxes = ri.scale_boxes(letterboxed_shape, boxes, im0.shape).round()
    return (boxes.detach().cpu().numpy().reshape(-1, 4), labels.detach().cpu().numpy(),
            scores.detach().cpu().numpy())


def frame_source():
    if not ARGS.source:
        LOOP["source"] = "RealSense" + (f" {ARGS.realsense}" if ARGS.realsense else "")
        yield from ri.realsense_iter_frames(ARGS.realsense or None)
        return
    LOOP["source"] = ARGS.source
    suffix = Path(ARGS.source).suffix[1:].lower()
    if suffix in ("jpg", "jpeg", "png", "bmp", "webp"):
        im0 = cv2.imread(ARGS.source)
        if im0 is None:
            raise FileNotFoundError(ARGS.source)
        while True:  # a still image is replayed as a "live" scene so the tracker/ask flow still runs
            yield im0.copy(), False
            time.sleep(0.1)
    while True:
        yield from ri.iter_frames(ARGS.source, 0)
        if not ARGS.loop:
            return


def live_loop(predictor):
    recorder = None
    t_prev = time.perf_counter()
    try:
        LOOP["running"] = True
        for im0, _is_last in frame_source():
            boxes, labels, scores = detect(predictor, im0)
            robot, ood = TEACHER.process(im0, boxes, labels, scores)
            recognize_jpg = None
            if EXPLORER is not None:
                recognize, _ood = EXPLORER.process(im0, boxes, labels, scores)
                recognize_jpg = encode_jpg(recognize, max_dim=1280, quality=80)
            FRAMES.publish(encode_jpg(robot, max_dim=1280, quality=80), encode_jpg(ood, max_dim=1280, quality=80),
                           recognize_jpg)
            if ARGS.record:
                if recorder is None:
                    recorder = cv2.VideoWriter(ARGS.record, cv2.VideoWriter_fourcc(*"mp4v"), ARGS.record_fps,
                                               (im0.shape[1], im0.shape[0]))
                recorder.write(im0)
            t = time.perf_counter()
            inst_fps = 1.0 / max(t - t_prev, 1e-6)
            TEACHER.fps = inst_fps if not TEACHER.fps else 0.9 * TEACHER.fps + 0.1 * inst_fps
            if EXPLORER is not None:
                EXPLORER.fps = TEACHER.fps
            t_prev = t
        print("[teach_app] source ended")
    except Exception as e:
        print(f"[teach_app] camera loop stopped: {e!r}")
        LOOP["error"] = str(e)
    finally:
        LOOP["running"] = False
        if recorder is not None:
            recorder.release()


# ============================================================================ routes
def _err(msg, code=400):
    return jsonify({"error": msg}), code


@app.route("/")
def index():
    return send_from_directory(PAGE_PATH.parent, PAGE_PATH.name)


@app.route("/recognize")
def recognize_page():
    return send_from_directory(RECOGNIZE_PAGE_PATH.parent, RECOGNIZE_PAGE_PATH.name)


@app.route("/stream/<which>.mjpg")
def stream(which):
    if which not in ("robot", "ood", "recognize") or (which == "recognize" and EXPLORER is None):
        return _err("unknown stream", 404)
    return Response(FRAMES.stream(which), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store"})


@app.route("/api/state")
def api_state():
    since = request.args.get("since", default=0, type=int)
    snap = TEACHER.snapshot(since)
    snap.update({"ready": snap["frame_count"] > 0, "running": LOOP["running"], "error": LOOP["error"],
                 "source": LOOP["source"], "dual_view": EXPLORER is not None})
    return jsonify(snap)


@app.route("/api/recognize/state")
def api_recognize_state():
    """The --dual-view explore side, same shape as /api/state."""
    if EXPLORER is None:
        return _err("start teach_app.py with --dual-view for the side-by-side recognize view", 404)
    since = request.args.get("since", default=0, type=int)
    snap = EXPLORER.snapshot(since)
    snap.update({"ready": snap["frame_count"] > 0, "running": LOOP["running"], "error": LOOP["error"],
                 "source": LOOP["source"]})
    return jsonify(snap)


@app.route("/api/mode", methods=["POST"])
def api_mode():
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify({"ok": True, "mode": TEACHER.set_mode(payload.get("mode"))})
    except ValueError as e:
        return _err(str(e))


@app.route("/api/answer", methods=["POST"])
def api_answer():
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify({"ok": True, **TEACHER.answer(int(payload.get("item_id", -1)), payload.get("name", ""))})
    except KeyError:
        return _err("that object is no longer being asked about", 404)
    except ValueError as e:
        return _err(str(e))


@app.route("/api/skip", methods=["POST"])
def api_skip():
    payload = request.get_json(silent=True) or {}
    return jsonify({"ok": TEACHER.skip(int(payload.get("item_id", -1)))})


@app.route("/api/skip_rest", methods=["POST"])
def api_skip_rest():
    """Skips every object still waiting on the stopped image and goes live again."""
    return jsonify({"ok": True, "skipped": TEACHER.skip_rest()})


@app.route("/api/capture/start", methods=["POST"])
def api_capture_start():
    payload = request.get_json(silent=True) or {}
    try:
        TEACHER.start_capture(payload.get("object_id"))
    except KeyError:
        return _err("no such object", 404)
    except ValueError as e:
        return _err(str(e))
    return jsonify({"ok": True})


@app.route("/api/capture/shoot", methods=["POST"])
def api_capture_shoot():
    try:
        return jsonify({"ok": True, "view": TEACHER.request_shot()})
    except (ValueError, TimeoutError, RuntimeError) as e:
        return _err(str(e))


@app.route("/api/capture/delete", methods=["POST"])
def api_capture_delete():
    payload = request.get_json(silent=True) or {}
    return jsonify({"ok": TEACHER.delete_shot(payload.get("idx", -1))})


@app.route("/api/capture/done", methods=["POST"])
def api_capture_done():
    try:
        return jsonify({"ok": True, **TEACHER.finish_capture()})
    except ValueError as e:
        return _err(str(e))


@app.route("/api/capture/cancel", methods=["POST"])
def api_capture_cancel():
    return jsonify({"ok": TEACHER.cancel_capture()})


@app.route("/api/memory")
def api_memory():
    return jsonify({"version": TEACHER.memory.version, "objects": TEACHER.memory.list_objects(),
                    "accept_thres": TEACHER.memory.accept_thres, "embedder": TEACHER.memory.embedder.model_id})


@app.route("/memory/<path:rel>")
def memory_file(rel):
    if not rel.startswith("views/"):
        return _err("not found", 404)
    return send_from_directory(TEACHER.memory.dir, rel)


@app.route("/api/memory/rename", methods=["POST"])
def api_memory_rename():
    payload = request.get_json(silent=True) or {}
    return jsonify({"ok": TEACHER.rename_object(payload.get("object_id"), payload.get("name", ""))})


@app.route("/api/memory/delete", methods=["POST"])
def api_memory_delete():
    payload = request.get_json(silent=True) or {}
    return jsonify({"ok": TEACHER.delete_object(payload.get("object_id"))})


def _swap_memory(swap):
    """Runs a memory_backup folder swap and reloads the memory both Teachers share."""
    TEACHER.check_can_swap_memory()
    result = TEACHER.memory.reload(swap)
    for teacher in (TEACHER, EXPLORER):
        if teacher is not None:
            teacher.memory_swapped()
    return result


@app.route("/api/memory/backups")
def api_memory_backups():
    return jsonify({"backups": memory_backup.list_backups(TEACHER.memory.dir),
                    "objects": [o["name"] for o in TEACHER.memory.list_objects()]})


@app.route("/api/memory/reset", methods=["POST"])
def api_memory_reset():
    """Fresh demo: back up the memory and start with an empty one."""
    try:
        backup = _swap_memory(lambda: memory_backup.backup_and_clear(TEACHER.memory.dir))
    except (ValueError, OSError) as e:
        return _err(str(e))
    return jsonify({"ok": True, "backup": backup})


@app.route("/api/memory/restore", methods=["POST"])
def api_memory_restore():
    payload = request.get_json(silent=True) or {}
    try:
        saved = _swap_memory(lambda: memory_backup.restore(TEACHER.memory.dir, str(payload.get("name", ""))))
    except (ValueError, OSError) as e:
        return _err(str(e))
    return jsonify({"ok": True, "backup": saved, "objects": [o["name"] for o in TEACHER.memory.list_objects()]})


# ============================================================================ main
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("camera")
    src.add_argument("--realsense", type=str, default="", metavar="SERIAL",
                     help="RealSense serial number to use when several are plugged in (default: the first one)")
    src.add_argument("--source", type=str, default=None,
                     help="video file, image, or stream URL to use instead of the RealSense")
    src.add_argument("--loop", action="store_true", help="restart --source video when it ends")
    src.add_argument("--record", type=str, default=None, metavar="OUT.mp4",
                     help="also save the raw camera frames to this video (for replay tests / a backup demo video)")
    src.add_argument("--record-fps", type=float, default=10.0)

    det = p.add_argument_group("detector (steps 1-2)")
    det.add_argument("--weights", type=str, default=str(ROOT / "model" / "model_final.pth"))
    det.add_argument("--cfg", type=str, default=ri.DEFAULT_CFG)
    det.add_argument("--conf-thres", type=float, default=0.9)
    det.add_argument("--energy-thres", type=float, default=-6.84)

    mode = p.add_argument_group("mode (can also be switched on the page)")
    mode.add_argument("--mode", choices=["teach", "explore"], default="teach",
                      help="teach: ask about unknown objects and learn them. explore: only show VOC classes and "
                           "OOD objects recognized from memory, never ask")
    mode.add_argument("--explore-ood-conf", type=float, default=0.95,
                      help="explore mode: min detector score for an OOD box to be shown (if memory names it)")
    mode.add_argument("--dual-view", action="store_true",
                      help="also run an explore-mode view of the same frames and memory, shown at /recognize "
                           "(teach and replay side by side)")

    mem = p.add_argument_group("memory (steps 5, 7)")
    mem.add_argument("--memory-dir", type=str, default=str(TEACH_DIR / "teach_memory"))
    mem.add_argument("--embedder", choices=sorted(EMBEDDERS), default=DEFAULT_EMBEDDER)
    mem.add_argument("--accept-thres", type=float, default=DEFAULT_ACCEPT_THRES,
                     help="min cosine similarity to a stored view to count as recognized; tune with calibrate.py")
    mem.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                     help="how much the best object must beat the runner-up by")
    mem.add_argument("--match-known-boxes", action="store_true",
                     help="also check boxes the detector labeled as a VOC class (except person) against memory, "
                          "with a stricter threshold -- use if a taught object gets a VOC label after being moved")
    mem.add_argument("--capture-min-sim", type=float, default=0.4,
                     help="in the capture window, min similarity to the object's views so far to keep following "
                          "the box being taught (a new side may look less alike)")
    mem.add_argument("--capture-acquire-sim", type=float, default=0.55,
                     help="in the capture window, min similarity for any other box to become the object being "
                          "taught (the one nearest the center mark wins); 0.4 let ~14%% of other objects through")
    mem.add_argument("--same-thres", type=float, default=0.72,
                     help="similarity above which two unknown detections are treated as the same object")
    mem.add_argument("--show-scores", action="store_true",
                     help="print the closest memory score on unknown boxes too (threshold tuning)")

    web = p.add_argument_group("web page")
    web.add_argument("--host", type=str, default="127.0.0.1")
    web.add_argument("--port", type=int, default=7861)
    return p.parse_args()


def main():
    global ARGS, TEACHER, EXPLORER
    ARGS = parse_args()
    weights = Path(ARGS.weights)
    if not weights.is_absolute() and not weights.exists():
        weights = ROOT / weights
    if not weights.exists():
        sys.exit(f"[teach_app] detector weights not found: {weights}")

    print(f"[teach_app] loading detector on {DEVICE} ...")
    predictor = ri.build_model(ARGS.cfg, str(weights), ARGS.conf_thres, DEVICE)
    embedder = Embedder(ARGS.embedder, device=str(DEVICE))
    embedder.embed([np.zeros((64, 64, 3), np.uint8)])  # warm-up, so the first real frame isn't slow
    memory = MultiViewMemory(ARGS.memory_dir, embedder, accept_thres=ARGS.accept_thres, margin=ARGS.margin)
    TEACHER = Teacher(
        embedder, memory, UnknownTracker(same_thres=ARGS.same_thres), ri.VOC_NAME_MAPPING, ri.OOD_CLASS_ID,
        person_class_id=PERSON_CLASS_ID, negatives_dir=Path(ARGS.memory_dir) / "negatives",
        match_known_boxes=ARGS.match_known_boxes, capture_min_sim=ARGS.capture_min_sim,
        capture_acquire_sim=ARGS.capture_acquire_sim, show_scores=ARGS.show_scores,
        mode=ARGS.mode, explore_ood_conf=ARGS.explore_ood_conf,
    )
    if ARGS.dual_view:
        # Its own tracker, so it never shares the teach side's question state; it never asks anyway.
        EXPLORER = Teacher(
            embedder, memory, UnknownTracker(same_thres=ARGS.same_thres), ri.VOC_NAME_MAPPING, ri.OOD_CLASS_ID,
            person_class_id=PERSON_CLASS_ID, match_known_boxes=ARGS.match_known_boxes, show_scores=ARGS.show_scores,
            mode="explore", explore_ood_conf=ARGS.explore_ood_conf,
        )
        print(f"[teach_app] dual view: recognize side at http://{ARGS.host}:{ARGS.port}/recognize")
    threading.Thread(target=live_loop, args=(predictor,), daemon=True).start()
    print(f"[teach_app] open http://{ARGS.host}:{ARGS.port}/ in Chrome or Edge")
    app.run(host=ARGS.host, port=ARGS.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
