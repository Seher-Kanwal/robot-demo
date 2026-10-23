"""
Main page for the robot demo: start, stop and open every part of it from one browser page
instead of juggling terminals.

    .venv/Scripts/python.exe robot_code/hub/hub_app.py
    # then open http://127.0.0.1:7800/ in Chrome or Edge

What it manages (each process is the normal script, started as a subprocess with the same
Python that runs the hub; output goes to robot_code/hub/logs/<name>.log):

  module1  robot_code/teach_mode/teach_app.py   Ask & Remember (port 7861)
  module2  robot_code/robot_web_app.py          VLM naming (port 7860, always --no-vllm here)
  module3  Qwen3-VL direct -- not a process of its own: it runs inside the module2 server
           (same Qwen weights) and is switched on there with the pipeline paused

It also recognizes copies that were started by hand in a terminal (by script name), so
the page shows the real state, and Stop works on those too. Only one process can own the
RealSense: starting something that needs it while another service holds it is refused
until the page confirms stopping the other one first.

Tool buttons run one-shot commands (tests, threshold calibration, memory listing, camera
and GPU checks) and stream their output into the page.
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import psutil
from flask import Flask, jsonify, request, send_from_directory

FILE = Path(__file__).resolve()
HUB_DIR = FILE.parent
ROBOT_CODE = HUB_DIR.parent
ROOT = ROBOT_CODE.parent
PY = sys.executable
LOG_DIR = HUB_DIR / "logs"
TEACH_MEMORY_DIR = ROBOT_CODE / "teach_mode" / "teach_memory"  # Module 1's default --memory-dir
REC_DIR = HUB_DIR / "recordings"
WEIGHTS = "model/model_final.pth"
VERIFY_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"  # Module 2's stage-2 model, same as its stage 1
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console window pops up per service

SERVICES = {
    "module1": {"title": "Module 1 · Ask & Remember", "script": "teach_app.py",
                "path": "robot_code/teach_mode/teach_app.py", "port": 7861},
    "module2": {"title": "Module 2 · VLM naming", "script": "robot_web_app.py",
                "path": "robot_code/robot_web_app.py", "port": 7860},
}
MODULE3_TITLE = "Module 3 · Qwen3-VL direct"
MODULE1_URL = f"http://127.0.0.1:{SERVICES['module1']['port']}"

sys.path.append(str(ROBOT_CODE / "teach_mode"))
import memory_backup  # noqa: E402 (folder moves only; the same code Module 1 runs on its live memory)
MODULE2_URL = f"http://127.0.0.1:{SERVICES['module2']['port']}"
MODULE3_START_TIMEOUT_S = 300  # module2's first start loads the detector + Qwen

CAMERA_CHECK = """
import pyrealsense2 as rs
devs = rs.context().query_devices()
print(f"{len(devs)} RealSense device(s) connected")
for d in devs:
    info = lambda k: d.get_info(k) if d.supports(k) else "?"
    print(f"  {info(rs.camera_info.name)}  serial {info(rs.camera_info.serial_number)}  "
          f"USB {info(rs.camera_info.usb_type_descriptor)}  firmware {info(rs.camera_info.firmware_version)}")
if len(devs) == 0:
    print("  none found -- check the USB cable (use a USB 3 port)")
"""

MEMORY1_LIST = """
import json, pathlib
p = pathlib.Path("robot_code/teach_mode/teach_memory/memory.json")
if not p.exists():
    print("Module 1 memory is empty (nothing taught yet)")
else:
    d = json.loads(p.read_text(encoding="utf-8"))
    print(f"{len(d['objects'])} object(s), embedder {d['embedder']}")
    for o in d["objects"]:
        origins = {}
        for v in o["views"]:
            origins[v["origin"]] = origins.get(v["origin"], 0) + 1
        print(f"  {o['name']:30s} {len(o['views']):3d} views  {origins}  (updated {o['updated_at'][:19]})")
"""

TOOLS = {
    "tests": {"title": "Run Module 1 tests", "cmd": [PY, "-m", "pytest", "robot_code/teach_mode/tests", "-q"]},
    "calibrate": {"title": "Calibrate Module 1 thresholds", "cmd": [PY, "-u", "robot_code/teach_mode/calibrate.py"]},
    "memory1": {"title": "List Module 1 memory", "cmd": [PY, "-c", MEMORY1_LIST]},
    "memory2": {"title": "List Module 2 memory", "cmd": [PY, "robot_code/manage_memory.py"]},
    "camera": {"title": "Check RealSense", "cmd": [PY, "-c", CAMERA_CHECK]},
    "gpu": {"title": "GPU status", "cmd": ["nvidia-smi"]},
}

app = Flask(__name__)
LOCK = threading.RLock()
MANAGED = {}  # name -> {"proc", "cmd", "started", "log", "stopped", "options"}
JOB = {"name": None, "proc": None, "output": [], "exit_code": None, "started": None}
MODULE3 = {"waiting": False, "token": 0, "error": None}  # waiting = module2 still starting, enable when up
_GPU_CACHE = {"at": 0.0, "value": None}


# ============================================================================ processes
def _port_open(port):
    if not port:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _scan():
    """name -> list of {pid, cmdline, create_time} for every running copy of each service,
    however it was started. On Windows a venv's python.exe is a launcher that runs the real
    interpreter as a child, so one launch shows up as two processes with the same command
    line; only the top one is reported."""
    found = {n: [] for n in SERVICES}
    for p in psutil.process_iter(["pid", "ppid", "cmdline", "create_time"]):
        cl = p.info["cmdline"] or []
        if not cl or p.info["pid"] == os.getpid():
            continue
        for n, s in SERVICES.items():
            if any(Path(arg).name == s["script"] for arg in cl[1:5]):  # script follows interpreter flags
                found[n].append({"pid": p.info["pid"], "ppid": p.info["ppid"], "cmdline": cl,
                                 "create_time": p.info["create_time"]})
    for n, procs in found.items():
        pids = {p["pid"] for p in procs}
        found[n] = [p for p in procs if p["ppid"] not in pids]
    return found


def _uses_camera(name, cmdline):
    if name == "module1":
        return "--source" not in cmdline
    if name == "module2":
        return "--realsense" in cmdline
    return False


def _log_tail(name, lines=40):
    path = LOG_DIR / f"{name}.log"
    if not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 20000))
        text = f.read().decode("utf-8", errors="replace")
    # progress bars rewrite one line with \r -- keep only the last state of each line
    out = [ln.rsplit("\r", 1)[-1] for ln in text.splitlines()]
    return [ln for ln in out if ln.strip()][-lines:]


def _http_json(url, payload=None, timeout=2.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="GET" if payload is None else "POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def _status():
    found = _scan()
    out = {}
    for n, s in SERVICES.items():
        insts = found[n]
        m = MANAGED.get(n)
        port_up = _port_open(s["port"])
        if insts:
            state = "running" if (port_up or s["port"] is None) else "starting"
        elif m and m["proc"].poll() not in (None, 0) and not m["stopped"]:
            state = "crashed"
        else:
            state = "stopped"
        mine = bool(m and m["proc"].poll() is None)
        out[n] = {
            "name": n, "title": s["title"], "state": state, "port": s["port"],
            "url": f"http://127.0.0.1:{s['port']}/" if s["port"] else None,
            "pids": [p["pid"] for p in insts], "instances": len(insts),
            "camera": any(_uses_camera(n, p["cmdline"]) for p in insts),
            "external": bool(insts) and not mine,
            "started": insts[0]["create_time"] if insts else None,
            "exit_code": m["proc"].poll() if m else None,
            "options": m["options"] if m else None,
            "log": _log_tail(n) if m else [],
        }
    out["module3"] = _module3_status(out["module2"])
    return out


def _module3_status(m2):
    """Module 3 lives inside the module2 server: it's "running" when that server has the
    Qwen-direct baseline switched on in solo mode."""
    q = None
    if m2["state"] == "running" and m2["camera"]:
        try:
            q = _http_json(f"{MODULE2_URL}/api/qwen-direct/live/latest?light=1", timeout=1.0)
        except (OSError, ValueError):
            q = None
    if q and q.get("enabled") and q.get("solo"):
        state = "crashed" if q.get("error") else "running"
        timing = q.get("timing") or {}
        detail = (f"{q.get('count', 0)} objects · {timing['infer_ms'] / 1000:.1f} s per pass"
                  if q.get("ready") and timing.get("infer_ms") else "first pass running…")
        if q.get("error"):
            detail = q["error"]
    elif MODULE3["waiting"] and m2["state"] in ("starting", "running"):
        state, detail = "starting", "waiting for Module 2's server to load the models…"
    else:
        state = "crashed" if MODULE3["error"] else "stopped"
        detail = MODULE3["error"] or ""
    return {
        "name": "module3", "title": MODULE3_TITLE, "state": state, "port": SERVICES["module2"]["port"],
        "url": f"{MODULE2_URL}/qwen-direct", "pids": m2["pids"], "instances": m2["instances"],
        "camera": False,  # the camera belongs to the module2 process -- counted there
        "external": m2["external"], "started": m2["started"], "exit_code": None, "options": None,
        "log": m2["log"], "detail": detail, "server_state": m2["state"],
    }


def _kill_tree(pids, timeout=8.0):
    procs = []
    for pid in pids:
        try:
            p = psutil.Process(pid)
            procs += p.children(recursive=True) + [p]
        except psutil.NoSuchProcess:
            pass
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _gone, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def stop_service(name):
    with LOCK:
        if name == "module2":
            MODULE3.update(waiting=False, token=MODULE3["token"] + 1)
        insts = _scan()[name]
        had_camera = any(_uses_camera(name, p["cmdline"]) for p in insts)
        _kill_tree([p["pid"] for p in insts])
        m = MANAGED.get(name)
        if m:
            m["stopped"] = True
            if m["proc"].poll() is None:
                _kill_tree([m["proc"].pid])
        if had_camera:
            time.sleep(1.0)  # give the RealSense driver a moment to release the device
        return len(insts)


# ============================================================================ commands
def _num(opts, key, default):
    try:
        return str(float(opts.get(key, default)))
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number")


def build_cmd(name, opts):
    s = SERVICES[name]
    cmd = [PY, "-u", s["path"]]
    if name == "module1":
        cmd += ["--port", str(s["port"]), "--weights", WEIGHTS,
                "--conf-thres", _num(opts, "conf_thres", 0.9), "--energy-thres", _num(opts, "energy_thres", -6.84),
                "--accept-thres", _num(opts, "accept_thres", 0.62),
                "--mode", "explore" if opts.get("mode") == "explore" else "teach",
                "--explore-ood-conf", _num(opts, "explore_ood_conf", 0.95)]
        if opts.get("record"):
            REC_DIR.mkdir(parents=True, exist_ok=True)
            cmd += ["--record", str(REC_DIR / f"teach_{datetime.now():%Y%m%d_%H%M%S}.mp4")]
        if opts.get("show_scores"):
            cmd += ["--show-scores"]
        if opts.get("match_known_boxes"):
            cmd += ["--match-known-boxes"]
        if opts.get("dual_view"):
            cmd += ["--dual-view"]
    elif name == "module2":
        cmd += ["--no-vllm", "--port", str(s["port"]), "--weights", WEIGHTS,
                "--conf-thres", _num(opts, "conf_thres", 0.9), "--energy-thres", _num(opts, "energy_thres", -6.84),
                # Stage 2 reuses the already-loaded 4B: the default 8B doesn't fit next to it on the
                # 24 GB card, spills into shared memory, and the first live frame then takes minutes.
                "--verify-model-id", VERIFY_MODEL_ID]
        if opts.get("realsense", True):
            cmd += ["--realsense"]
    return cmd


def start_service(name, opts, force=False):
    """Returns (status_code, payload). 409 + {"conflict": [...]} when the camera is taken."""
    with LOCK:
        st = _status()
        if st[name]["state"] in ("running", "starting"):
            return 409, {"error": f"{SERVICES[name]['title']} is already running -- stop or restart it"}
        try:
            cmd = build_cmd(name, opts)
        except ValueError as e:
            return 400, {"error": str(e)}
        if _uses_camera(name, cmd):
            owners = [n for n, v in st.items() if n != name and v["camera"]]
            if owners and not force:
                titles = ", ".join(SERVICES[o]["title"] + (" (and Module 3)" if o == "module2" and
                                                           st["module3"]["state"] == "running" else "")
                                   for o in owners)
                return 409, {"conflict": owners,
                             "error": f"The RealSense is in use by {titles}. Stop it and start {SERVICES[name]['title']}?"}
            for o in owners:
                stop_service(o)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = open(LOG_DIR / f"{name}.log", "w", encoding="utf-8")
        log.write(f"[hub] {datetime.now():%Y-%m-%d %H:%M:%S} starting: {subprocess.list2cmdline(cmd)}\n")
        log.flush()
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                                creationflags=CREATE_NO_WINDOW)
        log.close()  # the child keeps its own handle
        MANAGED[name] = {"proc": proc, "cmd": cmd, "started": time.time(), "stopped": False, "options": opts}
        return 200, {"ok": True, "pid": proc.pid}


def _set_module3(enabled):
    return _http_json(f"{MODULE2_URL}/api/qwen-direct/live", {"enabled": enabled, "solo": True}, timeout=5.0)


def _enable_module3_when_ready(token):
    """Module 2's server takes a while to start (detector + Qwen): keep trying until it accepts
    the switch, unless Module 3 was stopped (token changed) in the meantime."""
    deadline = time.time() + MODULE3_START_TIMEOUT_S
    while time.time() < deadline:
        with LOCK:
            if MODULE3["token"] != token:
                return
        try:
            _set_module3(True)
            with LOCK:
                if MODULE3["token"] == token:
                    MODULE3["waiting"] = False
            return
        except (OSError, ValueError):
            time.sleep(2.0)
    with LOCK:
        if MODULE3["token"] == token:
            MODULE3.update(waiting=False, error="Module 2's server didn't come up in time -- see its Log")


def start_module3(module2_opts, force=False, restart=False):
    with LOCK:
        st = _status()
        m2 = st["module2"]
        MODULE3["error"] = None
        if m2["state"] in ("running", "starting") and not m2["camera"]:
            if not restart:
                return 409, {"restart": True, "error": "Module 2's server is running without the RealSense, which "
                                                       "Module 3 needs. Restart it with the camera?"}
            stop_service("module2")
            m2 = _status()["module2"]
        if m2["state"] in ("stopped", "crashed"):
            code, body = start_service("module2", {**module2_opts, "realsense": True}, force=force)
            if code != 200:
                return code, body
        elif m2["state"] == "running":
            try:
                _set_module3(True)
                return 200, {"ok": True}
            except (OSError, ValueError) as e:
                return 502, {"error": f"Module 2's server didn't accept the switch: {e}"}
        MODULE3["token"] += 1
        MODULE3["waiting"] = True
        threading.Thread(target=_enable_module3_when_ready, args=(MODULE3["token"],), daemon=True).start()
        return 200, {"ok": True, "waiting": True}


def stop_module3():
    with LOCK:
        MODULE3.update(waiting=False, error=None, token=MODULE3["token"] + 1)
    try:
        _set_module3(False)
    except (OSError, ValueError):
        pass  # server not running -- nothing to switch off


# ============================================================================ tools
def _run_job(name, cmd):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                creationflags=CREATE_NO_WINDOW)
    except OSError as e:
        with LOCK:
            JOB.update(output=[f"could not start: {e}"], exit_code=-1, proc=None)
        return
    with LOCK:
        JOB["proc"] = proc
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n").rsplit("\r", 1)[-1]
        with LOCK:
            JOB["output"].append(line)
            del JOB["output"][:-2000]
    with LOCK:
        JOB["exit_code"] = proc.wait()


# ============================================================================ routes
@app.route("/")
def index():
    return send_from_directory(HUB_DIR, "hub.html")


@app.route("/compare")
def compare():
    """Module 1 and Module 2 side by side (the "Both modules" quick-start button opens it)."""
    return send_from_directory(HUB_DIR, "compare.html")


@app.route("/api/status")
def api_status():
    now = time.time()
    if now - _GPU_CACHE["at"] > 3.0:
        _GPU_CACHE["at"] = now
        try:
            used, total = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, creationflags=CREATE_NO_WINDOW,
            ).stdout.strip().splitlines()[0].split(",")
            _GPU_CACHE["value"] = {"used_mb": int(used), "total_mb": int(total)}
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            _GPU_CACHE["value"] = None
    with LOCK:
        st = _status()
        job = {"name": JOB["name"], "title": TOOLS.get(JOB["name"], {}).get("title"),
               "running": JOB["proc"] is not None and JOB["exit_code"] is None, "exit_code": JOB["exit_code"],
               "lines": len(JOB["output"])}
    camera_owner = next((v["title"] for v in st.values() if v["camera"]), None)
    if st["module3"]["state"] == "running":
        camera_owner = MODULE3_TITLE
    return jsonify({"services": st, "gpu": _GPU_CACHE["value"], "camera_owner": camera_owner, "job": job,
                    "tools": {k: v["title"] for k, v in TOOLS.items()}})


@app.route("/api/service/<name>/start", methods=["POST"])
def api_start(name):
    payload = request.get_json(silent=True) or {}
    if name == "module3":
        code, body = start_module3(payload.get("options") or {}, force=bool(payload.get("force")),
                                   restart=bool(payload.get("restart")))
        return jsonify(body), code
    if name not in SERVICES:
        return jsonify({"error": "unknown service"}), 404
    code, body = start_service(name, payload.get("options") or {}, force=bool(payload.get("force")))
    return jsonify(body), code


@app.route("/api/service/<name>/stop", methods=["POST"])
def api_stop(name):
    if name == "module3":
        stop_module3()
        return jsonify({"ok": True})
    if name not in SERVICES:
        return jsonify({"error": "unknown service"}), 404
    return jsonify({"ok": True, "stopped": stop_service(name)})


@app.route("/api/service/<name>/log")
def api_log(name):
    name = "module2" if name == "module3" else name
    if name not in SERVICES:
        return jsonify({"error": "unknown service"}), 404
    return jsonify({"lines": _log_tail(name, lines=request.args.get("lines", default=300, type=int))})


@app.route("/api/module1/mode", methods=["POST"])
def api_module1_mode():
    """Switches a running Module 1 between teach and explore without restarting it
    (forwarded server-side: the page can't POST JSON to another port itself)."""
    mode = (request.get_json(silent=True) or {}).get("mode")
    try:
        return jsonify(_http_json(f"{MODULE1_URL}/api/mode", {"mode": mode}, timeout=5.0))
    except urllib.error.HTTPError as e:
        return jsonify(json.loads(e.read() or b"{}") or {"error": f"HTTP {e.code}"}), e.code
    except OSError:
        return jsonify({"error": "Module 1 is not running (or is still starting)"}), 409


@app.route("/api/module1/memory/<action>", methods=["GET", "POST"])
def api_module1_memory(action):
    """Fresh demo: backups (GET), reset (POST) and restore (POST {name}) of Module 1's memory.
    A running Module 1 does it itself (it has the memory loaded); otherwise the hub moves the
    folders, and Module 1 reads the result on its next start."""
    if action not in ("backups", "reset", "restore"):
        return jsonify({"error": "unknown action"}), 404
    payload = request.get_json(silent=True) or {}
    with LOCK:
        state = _status()["module1"]["state"]
    if state == "starting":
        return jsonify({"error": "Module 1 is still starting -- try again in a moment"}), 409
    if state == "running":
        try:
            return jsonify(_http_json(f"{MODULE1_URL}/api/memory/{action}",
                                      None if action == "backups" else payload, timeout=10.0))
        except urllib.error.HTTPError as e:
            if e.code == 404:  # a Module 1 started before these routes existed
                return jsonify({"error": "restart Module 1 to use the fresh-demo buttons"}), 409
            try:
                return jsonify(json.loads(e.read() or b"{}") or {"error": f"HTTP {e.code}"}), e.code
            except ValueError:
                return jsonify({"error": f"Module 1 answered HTTP {e.code}"}), e.code
        except OSError as e:
            return jsonify({"error": f"Module 1 did not answer: {e}"}), 502
    try:
        if action == "backups":
            return jsonify({"backups": memory_backup.list_backups(TEACH_MEMORY_DIR),
                            "objects": memory_backup.summary(TEACH_MEMORY_DIR)[0]})
        if action == "reset":
            return jsonify({"ok": True, "backup": memory_backup.backup_and_clear(TEACH_MEMORY_DIR)})
        saved = memory_backup.restore(TEACH_MEMORY_DIR, str(payload.get("name", "")))
        return jsonify({"ok": True, "backup": saved, "objects": memory_backup.summary(TEACH_MEMORY_DIR)[0]})
    except (ValueError, OSError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/tool/<name>", methods=["POST"])
def api_tool(name):
    if name not in TOOLS:
        return jsonify({"error": "unknown tool"}), 404
    with LOCK:
        if JOB["proc"] is not None and JOB["exit_code"] is None:
            return jsonify({"error": f"{TOOLS[JOB['name']]['title']} is still running"}), 409
        JOB.update(name=name, proc=None, output=[f"$ {TOOLS[name]['title']}"], exit_code=None, started=time.time())
    threading.Thread(target=_run_job, args=(name, TOOLS[name]["cmd"]), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/tool/output")
def api_tool_output():
    since = request.args.get("since", default=0, type=int)
    with LOCK:
        return jsonify({"name": JOB["name"], "lines": JOB["output"][since:], "total": len(JOB["output"]),
                        "running": JOB["proc"] is not None and JOB["exit_code"] is None,
                        "exit_code": JOB["exit_code"]})


@app.route("/api/tool/stop", methods=["POST"])
def api_tool_stop():
    with LOCK:
        proc = JOB["proc"]
    if proc is not None and proc.poll() is None:
        _kill_tree([proc.pid], timeout=3)
    return jsonify({"ok": True})


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7800)
    args = parser.parse_args()
    print(f"[hub] open http://{args.host}:{args.port}/ in Chrome or Edge")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
