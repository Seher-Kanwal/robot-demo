"""
Remote camera client for robot_web_app.py.

Runs ON the robot (or wherever its camera is attached): grabs frames the same
way robot_implementation.py does -- webcam / video file / stream URL / single
image -- and POSTs each one to a remote robot_web_app.py server's /api/run
endpoint, so the detector + VLM inference stays on a GPU workstation instead of
running on the robot's own (likely much weaker) onboard compute.

Anything still "unknown" after both VLM stages shows up live in the browser --
open the server's page and start "Live (pushed frames)" to watch along and type
names in; a name typed there also resolves later frames near the same spot, via
the server's shared spatial cache.

Usage:
    # defaults to DEFAULT_SERVER_URL (127.0.0.1:7860) and DEFAULT_STREAM_URL (the
    # robot's web_video_server MJPEG stream) below -- just run it with no args:
    python robot_code/robot_client.py

    # workstation/server elsewhere on the network, still using the default stream:
    python robot_code/robot_client.py --server http://<workstation-ip>:7860

    # local webcam instead of the default stream (pass --source '' to opt out of it):
    python robot_code/robot_client.py --source '' --cam-index 0

    # dry run against a sample video instead of the live stream:
    python robot_code/robot_client.py --source Videos/15_seconds.mp4
"""
import argparse
import base64
import sys
import time
from pathlib import Path

import cv2
import requests

FILE = Path(__file__).resolve()
if str(FILE.parent) not in sys.path:
    sys.path.append(str(FILE.parent))

from robot_implementation import iter_frames  # reuse the exact same source-handling logic

# The robot's web_video_server -- serves /camera/camera/color/image_raw as MJPEG over
# HTTP, so it can be read here the same way as any other network stream (see the
# http(s)/rtsp branch of iter_frames in robot_implementation.py), no rclpy needed.
# Default --server/--source so a bare `python robot_client.py` streams this by default.
DEFAULT_STREAM_URL = "http://10.254.90.201:8081/stream?topic=/camera/camera/color/image_raw&type=mjpeg"
DEFAULT_SERVER_URL = "http://127.0.0.1:7860"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=DEFAULT_SERVER_URL,
                        help=f"robot_web_app.py base URL (default: {DEFAULT_SERVER_URL})")
    parser.add_argument("--source", type=str, default=DEFAULT_STREAM_URL,
                        help="image/video file or MJPEG stream URL; "
                             f"defaults to the robot's live camera stream ({DEFAULT_STREAM_URL}); "
                             "pass --source '' to fall back to a local webcam via --cam-index instead")
    parser.add_argument("--cam-index", type=int, default=0, help="camera index used when --source is omitted")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="seconds to wait between frames sent for a live camera (ignored for a single image; "
                             "throttles how often the workstation is hit -- start conservative, this is a full "
                             "detector+VLM pass per frame, not a lightweight ping)")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout per frame, in seconds")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="optional directory to save each annotated result frame locally")
    parser.add_argument("--skip-vlm", action="store_true",
                        help="skip both Qwen3-VL stages entirely -- OOD boxes are just labeled \"ood\", "
                             "with no per-frame VLM calls and nothing pending/needing a human. Faster and "
                             "simpler to watch a raw detector+energy-OOD stream with, e.g. while first "
                             "confirming the live view itself works before adding the VLM stages back in.")
    return parser.parse_args()


def send_frame(server, im0, timeout, source_label, skip_vlm=False):
    ok, buf = cv2.imencode(".jpg", im0, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("failed to encode frame")
    files = {"image": (source_label, buf.tobytes(), "image/jpeg")}
    # live=1 marks this as a robot_client.py push (vs. a manual browser upload) so the
    # server also stashes it for /api/pushed/latest -- see the "Live (pushed frames)"
    # section on live_feed_demo.html, which polls that and lets the operator resolve
    # anything still "unknown" right from the browser.
    data = {"live": "1", "skip_vlm": "1" if skip_vlm else "0"}
    resp = requests.post(f"{server}/api/run", files=files, data=data, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def main():
    args = parse_args()
    if args.source == "":
        args.source = None  # explicit opt-out of DEFAULT_STREAM_URL, falls back to --cam-index
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # A camera index and a network stream URL are both unbounded (iter_frames never
    # sets is_last for either), so --interval needs to throttle both the same way; only
    # a single image or a local video file is "finite" and should run through unthrottled.
    is_continuous_source = args.source is None or args.source.startswith(('http://', 'https://', 'rtsp://'))

    source_label = args.source or f"camera {args.cam_index}"
    print(f"[robot_client] streaming to {args.server} (source={source_label})")
    print(f"[robot_client] open {args.server}/ in a browser and start \"Live (pushed frames)\" to watch "
          f"along and resolve anything still unknown after both VLM stages.")
    for i, (im0, is_last) in enumerate(iter_frames(args.source, args.cam_index)):
        t0 = time.time()
        try:
            data = send_frame(args.server, im0, args.timeout, source_label, args.skip_vlm)
        except requests.RequestException as e:
            print(f"[robot_client] frame {i}: request failed ({e})")
            if not is_continuous_source:
                break
            time.sleep(args.interval)
            continue
        round_trip_ms = (time.time() - t0) * 1000

        stats = data["stats"]
        timing = data.get("timing", {})
        print(
            f"[robot_client] frame {i}: {stats['total']} detections "
            f"({stats['known']} known, {stats['ood_vlm']} OOD/VLM, {stats['ood_human']} OOD/human, "
            f"{stats['pending']} pending) — round trip {round_trip_ms:.0f} ms "
            f"(server: detect {timing.get('detect_ms')} ms, vlm {timing.get('vlm_total_ms')} ms)"
        )
        if stats["pending"]:
            print(f"[robot_client]   -> {stats['pending']} object(s) still unknown; job {data['job_id']} "
                  f"needs a human — resolve it in the \"Live (pushed frames)\" section at {args.server}/")

        if save_dir:
            img_bytes = base64.b64decode(data["image"].split(",", 1)[1])
            (save_dir / f"frame_{i:05d}.jpg").write_bytes(img_bytes)

        if is_last:
            break
        if is_continuous_source:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
