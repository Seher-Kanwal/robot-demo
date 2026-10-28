"""
Camera / video frame sources shared by robot_implementation.py, robot_web_app.py and
robot_client.py.

Split out of robot_implementation.py so robot_client.py (which runs on the laptop riding
on the robot) only needs cv2 + numpy (+ pyrealsense2 for a RealSense, rclpy/cv_bridge for
a ROS topic) -- not torch/detectron2/the rest of the detector stack, which only the GPU
workstation needs. robot_implementation.py re-exports everything here, so existing
`ri.iter_frames` / `ri.realsense_iter_frames` callers keep working unchanged.
"""
import os
import time
from pathlib import Path

import cv2
import numpy as np

# Same suffix lists as utils/dataloaders.py, inlined so importing this module doesn't pull
# in torch/torchvision through that file.
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'


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


def _start_realsense(rs, serial, width, height, fps, with_depth, wait_s=30.0, retry_every_s=2.0):
    """Starts a pipeline with the color stream (+ a depth stream when with_depth). The D435
    can do 1280x720 depth over USB 3; if that combination is refused (USB 2 port, other
    model) retry depth at 848x480 -- the aligned depth comes out at color resolution
    either way, so nothing downstream cares which one was picked.

    If every combination is refused, keeps retrying for up to wait_s: the camera is often
    still held for a few seconds by a process that was just stopped (the hub kills the old
    module right before starting the next one), or by the RealSense Viewer / Windows Camera
    app, which shows up as "Device or resource busy" or "Couldn't resolve requests"."""
    depth_sizes = [(width, height), (848, 480)] if with_depth else [None]
    deadline = time.monotonic() + wait_s
    attempt = 0
    while True:
        attempt += 1
        last_err = None
        for depth_size in depth_sizes:
            pipeline = rs.pipeline()
            config = rs.config()
            if serial:
                config.enable_device(serial)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            if depth_size:
                config.enable_stream(rs.stream.depth, depth_size[0], depth_size[1], rs.format.z16, fps)
            try:
                return pipeline, pipeline.start(config), depth_size
            except RuntimeError as e:
                last_err = e
                print(f"[realsense] could not start with depth {depth_size}: {e}")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{last_err} (camera still unavailable after {wait_s:.0f} s -- is another "
                               f"program, e.g. the RealSense Viewer or another module, using it?)")
        print(f"[realsense] camera busy or not ready (attempt {attempt}), retrying in {retry_every_s:.0f} s ...")
        time.sleep(retry_every_s)


def _equirect_to_view_maps(src_w, src_h, out_w, out_h, fov_deg, yaw_deg, pitch_deg):
    """cv2.remap maps that cut a normal (pinhole) view out of an equirectangular panorama.
    fov_deg is the view's horizontal field of view; yaw_deg turns it right of the panorama's
    centre, pitch_deg tilts it up."""
    f = 0.5 * out_w / np.tan(np.radians(fov_deg) / 2)
    xx, yy = np.meshgrid(np.arange(out_w) - (out_w - 1) / 2, np.arange(out_h) - (out_h - 1) / 2)
    rays = np.stack([xx, yy, np.full_like(xx, f)], axis=-1)  # x right, y down, z forward
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    p, y = np.radians(pitch_deg), np.radians(yaw_deg)
    rot_pitch = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rot_yaw = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rays = rays @ (rot_yaw @ rot_pitch).T
    lon = np.arctan2(rays[..., 0], rays[..., 2])  # 0 = panorama centre
    lat = np.arcsin(np.clip(rays[..., 1], -1, 1))  # positive = down
    map_x = ((lon / (2 * np.pi) + 0.5) * src_w).astype(np.float32)
    map_y = ((lat / np.pi + 0.5) * src_h).astype(np.float32)
    return map_x, map_y


def _open_theta(width, height, wait_s=30.0, retry_every_s=2.0):
    """Opens the RICOH THETA's UVC stream (live mode + the RICOH THETA UVC driver). OpenCV
    can't list camera names, so it takes the first camera index (or THETA_INDEX) that gives
    a 2:1 equirectangular frame. Retries for up to wait_s, like _start_realsense()."""
    indices = [int(os.environ["THETA_INDEX"])] if os.environ.get("THETA_INDEX") else range(6)
    deadline = time.monotonic() + wait_s
    while True:
        for i in indices:
            cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                ok, frame = cap.read()
                if ok and frame is not None and frame.shape[1] == 2 * frame.shape[0]:
                    return cap, i
            cap.release()
        if time.monotonic() >= deadline:
            raise RuntimeError("no RICOH THETA found -- is it plugged in, switched to LIVE mode, and is the "
                               "RICOH THETA UVC driver installed?")
        print(f"[theta] camera not found or busy, retrying in {retry_every_s:.0f} s ...")
        time.sleep(retry_every_s)


def theta_iter_frames(width=1280, height=720, with_depth=False):
    """Yields (im0, False) pairs from a USB RICOH THETA (Z1/V/X): only the front view, cut out
    of its 360-degree panorama at the RealSense's resolution, so nothing downstream can tell
    the difference. The view can be aimed with the THETA_YAW / THETA_PITCH env vars (degrees,
    default 0 = the panorama's centre) and widened with THETA_FOV (horizontal degrees, default
    90). The THETA has no depth: with_depth=True yields (im0, False, None). Like the other
    live sources, always hands the caller the newest frame."""
    import queue
    import threading

    fov = float(os.environ.get("THETA_FOV", 90))
    yaw = float(os.environ.get("THETA_YAW", 0))
    pitch = float(os.environ.get("THETA_PITCH", 0))
    cap, index = _open_theta(3840, 1920)
    src_h, src_w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    map_x, map_y = _equirect_to_view_maps(src_w, src_h, width, height, fov, yaw, pitch)
    print(f"[theta] streaming {src_w}x{src_h} panorama from camera index {index}; front view "
          f"{width}x{height}, fov {fov:g}, yaw {yaw:g}, pitch {pitch:g}")

    frame_q = queue.Queue(maxsize=1)
    stopped = threading.Event()

    def _reader():
        while not stopped.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                frame_q.put(None)
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
                raise RuntimeError("the THETA stopped sending frames")
            im0 = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            yield (im0, False, None) if with_depth else (im0, False)
    finally:
        stopped.set()
        cap.release()


def _use_theta(serial):
    """--realsense theta picks the THETA; with no serial, it's used when no RealSense is plugged in."""
    if serial:
        return serial.lower() == "theta"
    try:
        import pyrealsense2 as rs
    except ImportError:
        return True
    return len(rs.context().query_devices()) == 0


def realsense_iter_frames(serial=None, width=1280, height=720, fps=30, with_depth=False):
    """Yields (im0, False) pairs straight from a USB-attached Intel RealSense (D4xx) color
    stream via pyrealsense2 -- no ROS/realsense-ros needed. `serial` picks one camera when
    several are plugged in; None takes the first. Like ros_iter_frames(), always hands the
    caller the newest frame: anything that queued up while the caller was busy (detector +
    VLM pass) is drained first, so a slow loop doesn't drift behind the live camera.

    with_depth=True yields (im0, False, depth_info) triples instead, where depth_info is
    {"depth_mm": uint16 HxW depth aligned to im0 (millimetres, 0 = no reading),
     "fx", "fy", "cx", "cy": the color camera's intrinsics in im0 pixels}.

    With no RealSense plugged in (or serial "theta") this streams the front view of a RICOH
    THETA instead; see theta_iter_frames()."""
    if _use_theta(serial):
        yield from theta_iter_frames(width, height, with_depth)
        return
    import pyrealsense2 as rs

    pipeline, profile, depth_size = _start_realsense(rs, serial, width, height, fps, with_depth)
    device = profile.get_device()
    print(f"[realsense] streaming color {width}x{height}@{fps}"
          + (f" + depth {depth_size[0]}x{depth_size[1]} (aligned to color)" if depth_size else "")
          + f" from {device.get_info(rs.camera_info.name)} (serial {device.get_info(rs.camera_info.serial_number)})")

    align = intrinsics = None
    depth_scale_mm = 1.0
    if depth_size:
        align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        intrinsics = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy}
        depth_scale_mm = device.first_depth_sensor().get_depth_scale() * 1000.0  # 1.0 on a D435

    try:
        for _ in range(15):  # let auto-exposure settle before handing out the first frame
            pipeline.wait_for_frames()
        while True:
            frames = pipeline.wait_for_frames()
            while True:
                newer = pipeline.poll_for_frames()
                if not newer:
                    break
                frames = newer
            if align is not None:
                frames = align.process(frames)  # only the frame we keep pays for alignment
            color = frames.get_color_frame()
            if not color:
                continue
            im0 = np.asanyarray(color.get_data()).copy()
            if align is None:
                yield im0, False
                continue
            depth = frames.get_depth_frame()
            if not depth:
                continue
            depth_raw = np.asanyarray(depth.get_data())
            if abs(depth_scale_mm - 1.0) > 1e-6:
                depth_mm = np.clip(depth_raw * depth_scale_mm, 0, 65535).astype(np.uint16)
            else:
                depth_mm = depth_raw.copy()
            yield im0, False, {"depth_mm": depth_mm, **intrinsics}
    finally:
        pipeline.stop()


def parse_realsense_source(source):
    """'realsense' -> (True, None); 'realsense:<serial>' -> (True, '<serial>'); else (False, None)."""
    if source is None:
        return False, None
    head, _, serial = source.partition(':')
    if head.lower() != 'realsense':
        return False, None
    return True, (serial or None)


def iter_frames(source, cam_index, ros_topic=None, ros_qos='sensor'):
    """Yields (im0, is_last) pairs. im0 is a BGR HWC uint8 ndarray (original resolution)."""
    if ros_topic is not None:
        yield from ros_iter_frames(ros_topic, ros_qos)
        return
    is_realsense, rs_serial = parse_realsense_source(source)
    if is_realsense:
        yield from realsense_iter_frames(rs_serial)
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
