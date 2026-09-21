"""
Live OOD-aware object detection for the robot: continuously reads frames from a
camera (a local webcam, a ROS 2 image topic, or a file for offline testing),
runs the Pascal-VOC energy-based OOD detector on each frame, and resolves the
identity of any out-of-distribution detection through a three-stage
uncertainty pipeline before displaying the annotated stream on screen:

  1. A fast primary VLM (Qwen3-VL-4B by default) proposes a general category
     name, or "unknown" if it isn't confident.
  2. Anything still "unknown" gets a second opinion from a bigger, more
     capable VLM (Qwen3-VL-8B by default) that looks again more carefully.
  3. Anything still "unknown" after that is treated as genuinely uncertain and
     handed to a human: the box is highlighted on screen and the operator is
     asked to type in the object's name (simulated QA).

To stream from a ROS 2 camera topic (e.g. a RealSense via realsense-ros) instead
of a local webcam, pass --ros-topic. This needs rclpy + cv_bridge on the Python
path used to run this script -- source your ROS 2 install first (or use a venv
created with --system-site-packages) before installing this repo's other
dependencies into it, e.g.:

    source /opt/ros/jazzy/setup.bash
    ros2 topic list  # find the color image topic, e.g. /camera/camera/color/image_raw
    python robot_code/robot_implementation.py --ros-topic /camera/camera/color/image_raw

Alternatively, if the ROS 2 side already runs `web_video_server` (no rclpy/cv_bridge
needed here at all -- OpenCV reads the MJPEG stream directly like any video source):

    python robot_code/robot_implementation.py \
        --source "http://<host>:8081/stream?topic=/camera/camera/color/image_raw&type=mjpeg"

For a live stream, stage 1 (the primary VLM) is the hottest path -- it runs on every
OOD object in every frame -- so by default this script talks to a persistent vLLM
server for it instead of loading the checkpoint in-process (see --vllm-url / --no-vllm
below). Start that server once, in a separate terminal, before running this script:

    robot_code/start_vllm_server.sh
    python robot_code/robot_implementation.py --ros-topic /camera/camera/color/image_raw

Pass --no-vllm to skip that and load stage 1 locally instead (the old behavior; no
server needed, but pays per-call model/KV-cache setup cost on every request).
"""

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # repo root (robot_code/ is one level down)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# qwen3_generator.py reads QWEN_MODEL_ID at import time; the 30B default doesn't
# comfortably fit on a single 24GB GPU alongside the detector, so default this
# script to the 4B checkpoint. setdefault() still lets an explicit env var win.
# The bigger verification checkpoint (--verify-model-id, Qwen3-VL-8B by default)
# is loaded lazily and only if/when stage 1 actually reports "unknown", so its
# memory cost isn't paid unless the pipeline needs the second opinion.
os.environ.setdefault("QWEN_MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")

# qwen3_generator.py also reads QWEN_DEVICE_MAP at import time and defaults it to
# "auto" (accelerate's balanced placement), which readily splits a model across
# every visible GPU even when it would comfortably fit on one -- great for batch
# throughput, terrible for this script's interactive per-frame latency, since a
# split model pays a cross-GPU transfer on every decode step. Both the 4B primary
# and 8B verification checkpoints fit on a single 24GB card, so pin each one to
# whichever GPU is freest at load time instead (see qwen3_generator._build_bundle).
# Only scoped to this script -- other callers (e.g. Lvis_evalution.py's default
# 30B-A3B eval, which genuinely needs multi-GPU sharding to fit) are unaffected.
os.environ.setdefault("QWEN_DEVICE_MAP", "single")

from detectron2.config import get_cfg

from core.setup import add_probabilistic_config
from inference.inference_utils import build_predictor
from utils.augmentations import letterbox
from utils.dataloaders import IMG_FORMATS, VID_FORMATS
from utils.general import scale_boxes
from utils.plots import Annotator

import ood_memory

VOC_NAME_MAPPING = {0: 'person', 1: 'bird', 2: 'cat', 3: 'cow', 4: 'dog', 5: 'horse', 6: 'sheep', 7: 'airplane',
                    8: 'bicycle', 9: 'boat', 10: 'bus', 11: 'car', 12: 'motorcycle', 13: 'train', 14: 'bottle',
                    15: 'chair', 16: 'dining table', 17: 'potted plant', 18: 'couch', 19: 'tv', 20: ''}
OOD_CLASS_ID = 20

DEFAULT_WEIGHTS = "1401_seher/model_final.pth"
DEFAULT_CFG = "configs/PascalVOC-Detection/default/default.yaml"
DEFAULT_VERIFY_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_VLLM_URL = "http://127.0.0.1:8000/v1"  # see start_vllm_server.sh
IMGSZ = 800
STRIDE = 32
WINDOW_NAME = "robot_ood_detection"
QA_WINDOW_NAME = "robot_ood_qa_crop"
KNOWN_COLOR = (255, 0, 0)     # BGR blue: known VOC class
OOD_COLOR = (0, 0, 255)       # BGR red: still unresolved -- only the energy-based OOD detector
                              # has flagged it, no VLM/memory/human has named it yet
VLM_COLOR = (35, 166, 245)    # BGR orange: named by a VLM stage (vlm1/vlm2); matches --vlm in
                              # live_feed_demo.html
MEMORY_COLOR = (224, 120, 180)  # BGR violet: matched a previously human-named object by
                                # appearance (ood_memory.OODMemory); matches --memory
QA_COLOR = (0, 255, 255)      # BGR yellow: the human operator confirmed this name directly

CACHE_QUANT_PX = 32       # spatial quantization stride for the OOD label cache key
MAX_CACHE_ENTRIES = 2048  # bounded LRU so a long-running stream doesn't grow unbounded

# The second (bigger) VLM pass reuses this prompt instead of qwen3_generator's
# DEFAULT_PROMPT, since it should read as "look again, more carefully" rather
# than a plain first attempt.
VERIFY_PROMPT = (
    "You are a second, more capable vision expert double-checking a case the first "
    "model could not confidently identify. Look very closely at the image and identify "
    "the object. Reply with its general category name as a single word or short phrase. "
    "Do not add explanations or punctuation. "
    "If you truly cannot tell what the object is, reply with exactly: unknown."
)


def _is_unknown(label):
    return (label or "").strip().lower() in ("", "unknown")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=None,
                        help='image/video file, or an http(s)/rtsp stream URL (e.g. a web_video_server '
                             'MJPEG stream like http://<host>:8081/stream?topic=...&type=mjpeg); '
                             'omit (and omit --ros-topic) to stream from the local camera')
    parser.add_argument('--cam-index', type=int, default=0, help='camera index used when --source is omitted')
    parser.add_argument('--ros-topic', type=str, default=None,
                        help='ROS 2 sensor_msgs/Image topic to stream frames from (e.g. '
                             '/camera/camera/color/image_raw for realsense-ros). Takes priority '
                             'over --source/--cam-index when set; requires rclpy + cv_bridge on '
                             'the Python path (source your ROS 2 setup.bash first).')
    parser.add_argument('--ros-qos', choices=['sensor', 'default'], default='sensor',
                        help="QoS profile for the --ros-topic subscription. 'sensor' (best-effort, "
                             "volatile) matches realsense-ros and most camera drivers and is usually "
                             "required to receive any frames from them; 'default' (reliable, depth 10) "
                             "suits topics republished with reliable QoS.")
    parser.add_argument('--weights', type=str, default=DEFAULT_WEIGHTS, help='detectron2 checkpoint (.pth)')
    parser.add_argument('--cfg', type=str, default=DEFAULT_CFG, help='detectron2 config yaml')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold for kept detections')
    parser.add_argument('--energy-thres', type=float, default=-6.84, help='energy score threshold for OOD/unknown')
    parser.add_argument('--verify-model-id', type=str, default=DEFAULT_VERIFY_MODEL_ID,
                        help='bigger VLM used for the stage-2 second opinion on OOD objects the '
                             'primary (QWEN_MODEL_ID) model could not name')
    parser.add_argument('--no-human-qa', action='store_true',
                        help='disable the stage-3 human QA fallback; objects still unknown after '
                             'both VLM stages stay labeled "unknown" instead of prompting the operator')
    parser.add_argument('--vllm-url', type=str, default=DEFAULT_VLLM_URL,
                        help='base URL of a running vLLM OpenAI-compatible server (see '
                             'start_vllm_server.sh) used for stage-1 (the primary VLM) instead of '
                             'loading that checkpoint in-process. Set to "" or pass --no-vllm to '
                             'load it locally instead.')
    parser.add_argument('--no-vllm', action='store_true',
                        help='equivalent to --vllm-url ""; load stage 1 locally instead of using a '
                             'vLLM server')
    parser.add_argument('--memory-path', type=str, default=str(FILE.parent / "ood_memory.json"),
                        help='JSON file persisting appearance-based memory of human-given names for '
                             'OOD objects (default: robot_code/ood_memory.json next to this script). '
                             'Delete this file to reset the memory.')
    parser.add_argument('--memory-sim-thres', type=float, default=ood_memory.DEFAULT_SIM_THRESHOLD,
                        help='cosine-similarity threshold (0-1) for accepting a stored memory name as '
                             'a match before falling back to the VLM stages; see ood_memory.OODMemory '
                             'docstring for tuning guidance')
    parser.add_argument('--no-memory', action='store_true',
                        help='disable the appearance-based memory lookup entirely (VLM stage-1 runs '
                             'on every never-before-cached OOD object, as before this feature)')
    return parser.parse_args()


def build_model(cfg_path, weights_path, conf_thres, device):
    cfg_path = cfg_path if os.path.isabs(cfg_path) else str(ROOT / cfg_path)
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20 + 1
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thres
    cfg.MODEL.DEVICE = str(device)
    add_probabilistic_config(cfg)
    return build_predictor(cfg)


def ros_iter_frames(topic, qos='sensor'):
    """Yields (im0, False) pairs read from a ROS 2 sensor_msgs/Image topic, forever (until the
    process is interrupted). im0 is a BGR HWC uint8 ndarray, converted from whatever encoding the
    publisher uses via cv_bridge. rclpy is spun on a background thread; only the newest frame is
    kept (a bounded queue of size 1) so a slow detector loop falls behind the camera instead of
    building up latency."""
    import queue
    import threading

    import rclpy
    from cv_bridge import CvBridge
    from rclpy.qos import QoSProfile, qos_profile_sensor_data
    from sensor_msgs.msg import Image as RosImage

    bridge = CvBridge()
    frame_q = queue.Queue(maxsize=1)

    def _on_image(msg):
        im0 = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if frame_q.full():
            try:
                frame_q.get_nowait()  # drop the stale frame; we only ever want the latest
            except queue.Empty:
                pass
        frame_q.put(im0)

    rclpy.init(args=None)
    node = rclpy.create_node('samood_camera_subscriber')
    qos_profile = qos_profile_sensor_data if qos == 'sensor' else QoSProfile(depth=10)
    node.create_subscription(RosImage, topic, _on_image, qos_profile)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print(f"[ros] subscribed to '{topic}' (qos={qos}); waiting for frames ...")
    try:
        while rclpy.ok():
            yield frame_q.get(), False  # blocks until a frame arrives
    finally:
        node.destroy_node()
        rclpy.shutdown()


def iter_frames(source, cam_index, ros_topic=None, ros_qos='sensor'):
    """Yields (im0, is_last) pairs. im0 is a BGR HWC uint8 ndarray (original resolution)."""
    if ros_topic is not None:
        yield from ros_iter_frames(ros_topic, ros_qos)
        return
    if source is not None:
        if source.startswith(('http://', 'https://', 'rtsp://')):
            # A network video stream -- e.g. a web_video_server MJPEG URL such as
            # http://<host>:8081/stream?topic=/camera/camera/color/image_raw&type=mjpeg.
            # No rclpy/cv_bridge needed on this side: OpenCV/ffmpeg reads it like any
            # other video source. Treated as endless, like the local camera branch below.
            #
            # Read continuously on a background thread and always hand the caller the
            # newest frame, dropping any older ones still queued -- cv2.VideoCapture
            # buffers frames as they arrive regardless of how often the caller asks for
            # one, and read() returns the *oldest* buffered frame first. A caller that
            # only pulls one frame every few seconds (e.g. robot_client.py's --interval)
            # would otherwise silently fall further and further behind the live feed
            # instead of staying in sync with it -- the same problem ros_iter_frames()
            # above already avoids with its own bounded queue, mirrored here for a URL.
            import queue
            import threading

            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open stream: {source}")

            frame_q = queue.Queue(maxsize=1)
            stopped = threading.Event()

            def _reader():
                while not stopped.is_set():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        frame_q.put(None)  # signal failure to the consumer below
                        return
                    if frame_q.full():
                        try:
                            frame_q.get_nowait()  # drop the stale frame; keep only the newest
                        except queue.Empty:
                            pass
                    frame_q.put(frame)

            threading.Thread(target=_reader, daemon=True).start()
            try:
                while True:
                    frame = frame_q.get()
                    if frame is None:
                        raise RuntimeError("Stream ended or failed to capture a frame")
                    yield frame, False
            finally:
                stopped.set()
                cap.release()
            return
        suffix = Path(source).suffix[1:].lower()
        if suffix in IMG_FORMATS:
            im0 = cv2.imread(source)
            if im0 is None:
                raise FileNotFoundError(f"Could not read image: {source}")
            yield im0, True
            return
        elif suffix in VID_FORMATS:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {source}")
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        return
                    yield frame, False
            finally:
                cap.release()
        else:
            raise ValueError(f"Unsupported source file type: {source}")
    else:
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {cam_index}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise RuntimeError("Camera opened but failed to capture a frame")
                yield frame, False
        finally:
            cap.release()


def preprocess(im0):
    im = letterbox(im0, new_shape=IMGSZ, stride=STRIDE, auto=True)[0]  # stays BGR, no color conversion
    im = im.transpose(2, 0, 1)  # HWC -> CHW
    im = np.ascontiguousarray(im)
    image_tensor = torch.as_tensor(im)
    letterboxed_shape = image_tensor.shape[1:]  # (H1, W1)
    return image_tensor, letterboxed_shape


def run_inference(predictor, image_tensor, letterboxed_shape):
    data_dict = {
        'file_name': '',
        'height': letterboxed_shape[0],
        'width': letterboxed_shape[1],
        'image_id': 0,
        'image': image_tensor,
    }
    return predictor([data_dict])


def postprocess_energy(results, energy_threshold, conf_threshold):
    predicted_boxes = results.pred_boxes.tensor
    labels = results.det_labels.clone()
    scores = results.scores
    inter_feat = results.inter_feat

    energy = torch.logsumexp(inter_feat[:, :-2], dim=1)
    labels[energy < energy_threshold] = OOD_CLASS_ID

    keep = scores >= conf_threshold
    return predicted_boxes[keep], labels[keep], scores[keep]


def ood_cache_key(box_xyxy):
    return tuple((box_xyxy // CACHE_QUANT_PX).long().tolist())


def ask_user_for_label(im0, box, crop_pil):
    """Stage-3 QA fallback: highlights `box` on the live window, shows a zoomed-in crop in a
    second window, and blocks on stdin asking the operator to type the object's name.
    Returns the typed name (lowercased), or '' if the operator left it blank."""
    x1, y1, x2, y2 = box
    preview = np.ascontiguousarray(im0.copy())
    cv2.rectangle(preview, (x1, y1), (x2, y2), QA_COLOR, 3)
    cv2.putText(preview, "unknown - awaiting operator input", (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, QA_COLOR, 2)
    cv2.imshow(WINDOW_NAME, preview)
    cv2.imshow(QA_WINDOW_NAME, cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)  # pump the GUI event loop so both windows actually paint before we block on input()

    try:
        typed = input(
            f"[QA] Both VLM stages are unsure about the highlighted object (see {QA_WINDOW_NAME}). "
            "Type its name, or press Enter to leave it 'unknown': "
        ).strip()
    except EOFError:
        typed = ""

    cv2.destroyWindow(QA_WINDOW_NAME)
    return typed.lower()


def label_ood_detections(im0, ood_boxes_xyxy, cache, verify_model_id, human_qa_enabled=True, memory=None):
    """Returns a list[(label, source)] aligned 1:1 with ood_boxes_xyxy rows. `source` is
    "vlm1" (primary model), "vlm2" (bigger verification model), "memory" (matched a
    previously human-named object by appearance, see ood_memory.OODMemory), or "human"
    (operator typed it in), so callers can render provenance differently. `cache` avoids
    re-running the whole pipeline for objects that keep reappearing near the same image
    location; `memory`, if given, additionally recognizes the same physical object by
    appearance regardless of where it is on screen or which process/session it's in."""
    h0, w0 = im0.shape[:2]
    results = [None] * len(ood_boxes_xyxy)
    pending_indices = []
    pending_keys = []
    pending_boxes = []
    pending_crops = []

    for i, box in enumerate(ood_boxes_xyxy):
        key = ood_cache_key(box)
        if key in cache:
            cache.move_to_end(key)
            results[i] = cache[key]
            continue

        x1, y1, x2, y2 = (int(round(v)) for v in box.tolist())
        x1, x2 = max(0, min(w0, x1)), max(0, min(w0, x2))
        y1, y2 = max(0, min(h0, y1)), max(0, min(h0, y2))
        crop = im0[y1:y2, x1:x2]
        if crop.size == 0:
            results[i] = ("unknown", "vlm1")
            cache[key] = results[i]
            continue

        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        if memory is not None:
            remembered = memory.lookup(crop_pil)
            if remembered:
                results[i] = (remembered, "memory")
                cache[key] = results[i]  # later frames near this spot hit the cheap spatial cache
                continue

        pending_indices.append(i)
        pending_keys.append(key)
        pending_boxes.append((x1, y1, x2, y2))
        pending_crops.append(crop_pil)

    if pending_crops:
        # Lazy import: only load Qwen3-VL once a genuinely new OOD object shows up.
        from OOD_object_name_generation.qwen3_generator import describe_images

        # Stage 1: fast primary model (already resident; see QWEN_MODEL_ID above).
        stage1_labels = describe_images(pending_crops, max_new_tokens=6, batch_size=len(pending_crops))
        final = [(label, "vlm1") for label in stage1_labels]

        # Stage 2: bigger verifier model, only for what stage 1 couldn't name. Loaded lazily
        # on first use and kept resident afterwards (see qwen3_generator's model cache).
        unsure = [j for j, label in enumerate(stage1_labels) if _is_unknown(label)]
        if unsure:
            print(f"[OOD] stage-1 unsure about {len(unsure)} object(s); asking {verify_model_id} for a second opinion")
            verify_crops = [pending_crops[j] for j in unsure]
            stage2_labels = describe_images(
                verify_crops, prompt=VERIFY_PROMPT, max_new_tokens=6,
                batch_size=len(verify_crops), model_id=verify_model_id,
            )
            for j, label in zip(unsure, stage2_labels):
                final[j] = (label, "vlm2")

        # Stage 3: still unknown after the verifier -> ask the operator (if enabled).
        still_unsure = [j for j, (label, _) in enumerate(final) if _is_unknown(label)]
        if still_unsure and human_qa_enabled:
            print(f"[OOD] verifier still unsure about {len(still_unsure)} object(s); asking the operator")
            for j in still_unsure:
                typed = ask_user_for_label(im0, pending_boxes[j], pending_crops[j])
                if typed:
                    final[j] = (typed, "human")
                    if memory is not None:
                        memory.remember(pending_crops[j], typed)
                else:
                    final[j] = ("unknown", "vlm2")

        for j, i in enumerate(pending_indices):
            results[i] = final[j]
            cache[pending_keys[j]] = final[j]
            if len(cache) > MAX_CACHE_ENTRIES:
                cache.popitem(last=False)

    return results


def annotate(im0, boxes, labels, scores, ood_labels_by_row):
    im0 = np.ascontiguousarray(im0)
    annotator = Annotator(im0, line_width=3, example=str(VOC_NAME_MAPPING))
    for i in range(len(boxes)):
        cls = int(labels[i])
        if cls == OOD_CLASS_ID:
            desc, source = ood_labels_by_row.get(i, ("unknown", "vlm1"))
            if source == "dismissed":
                continue  # operator dismissed it without naming it -- don't draw it at all
            # Color (and an explicit tag in the overlay text) shows *how* this name was
            # arrived at, not just that it's OOD -- still-unknown, VLM-guessed,
            # memory-matched, and human-confirmed all look meaningfully different to an
            # operator deciding whether to trust the label.
            if _is_unknown(desc):
                color, tag = OOD_COLOR, "ood"
            elif source == "human":
                color, tag = QA_COLOR, "confirmed"
            elif source == "memory":
                color, tag = MEMORY_COLOR, "memory"
            else:
                color, tag = VLM_COLOR, "vlm"
            text = f"{desc} [{tag}] {scores[i]:.2f}"
        else:
            text = f"{VOC_NAME_MAPPING[cls]} {scores[i]:.2f}"
            color = KNOWN_COLOR
        annotator.box_label(boxes[i].tolist(), text, color=color)
    return annotator.result()


def main():
    args = parse_args()

    # qwen3_generator.py reads QWEN_VLLM_BASE_URL at call time (not import time, unlike
    # QWEN_MODEL_ID/QWEN_DEVICE_MAP above), so setting it here -- before the loop below
    # ever triggers describe_images()'s lazy import -- is early enough.
    vllm_url = "" if args.no_vllm else (args.vllm_url or "")
    if vllm_url:
        os.environ["QWEN_VLLM_BASE_URL"] = vllm_url
        print(f"[robot_implementation] stage 1 will use the vLLM server at {vllm_url} "
              f"(start it with start_vllm_server.sh if you haven't; --no-vllm to load it locally instead)")
    else:
        os.environ.pop("QWEN_VLLM_BASE_URL", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = build_model(args.cfg, args.weights, args.conf_thres, device)
    cache = OrderedDict()
    memory = None if args.no_memory else ood_memory.OODMemory(args.memory_path, args.memory_sim_thres)
    human_qa_enabled = not args.no_human_qa

    try:
        for im0, is_last in iter_frames(args.source, args.cam_index, args.ros_topic, args.ros_qos):
            image_tensor, letterboxed_shape = preprocess(im0)
            with torch.no_grad():
                results = run_inference(predictor, image_tensor, letterboxed_shape)
            boxes, labels, scores = postprocess_energy(results, args.energy_thres, args.conf_thres)

            ood_labels_by_row = {}
            if len(boxes):
                boxes = scale_boxes(letterboxed_shape, boxes, im0.shape).round()
                ood_idx = (labels == OOD_CLASS_ID).nonzero(as_tuple=True)[0]
                if ood_idx.numel():
                    descriptions = label_ood_detections(
                        im0, boxes[ood_idx], cache, args.verify_model_id, human_qa_enabled, memory=memory,
                    )
                    ood_labels_by_row = {int(row): desc for row, desc in zip(ood_idx.tolist(), descriptions)}
            else:
                print("No detections above threshold.")

            frame = annotate(im0, boxes, labels, scores, ood_labels_by_row)
            cv2.imshow(WINDOW_NAME, frame)

            if is_last:
                cv2.waitKey(0)
                break
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 'q' or Esc
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
