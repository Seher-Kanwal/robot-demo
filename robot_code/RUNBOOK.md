# Running the robot OOD demo

Three processes, started in order, each in its **own terminal** (they're
long-running and print live logs — don't background them into one shell).

## Environments

Two separate conda envs are involved — don't mix them up:

| Env | Used for | Why separate |
|---|---|---|
| `vllm_serve` | `robot_code/start_vllm_server.sh` only | vLLM pins its own torch/CUDA build (torch 2.13.0+cu130 as of vllm 0.29.0). Installing it into `sam_oed` would force a torch reinstall there and break `sam_oed`'s compiled detectron2 CUDA extensions (built against torch 2.11.0+cu128). |
| `sam_oed` | `robot_web_app.py`, `robot_client.py` | Has detectron2 + the detector/OOD stack. Talks to the vLLM server over plain HTTP (`httpx`), so it never needs vllm itself installed. |

Both envs already exist on this machine (`conda env list` to confirm). You
generally don't need to touch env setup — just make sure step 2/3 run inside
`sam_oed` (`start_vllm_server.sh` activates `vllm_serve` internally, you
don't need to `conda activate` it yourself). If `sam_oed` is ever missing or
broken, it needs: detectron2 built for CUDA + torch 2.11.0+cu128, plus the
packages in `requirements.txt` at the repo root (`transformers`, `qwen-vl`,
etc.) — there's no committed `environment.yml`, so rebuilding it from scratch
means matching whatever the existing detectron2 build was compiled against.

## 1. vLLM server (stage-1 VLM)

**Terminal 1:**

```bash
cd /home/seher/Projects/object_detection/robot_demo
robot_code/start_vllm_server.sh
```

Serves `Qwen/Qwen3-VL-4B-Instruct` at `http://127.0.0.1:8000/v1` (GPU 1 by
default; override with `--gpu`). First start downloads/compiles and can take
a few minutes; subsequent starts are fast. Runs in the separate `vllm_serve`
conda env (see comments at the top of the script for why).

The script also exports `LD_LIBRARY_PATH` to the `vllm_serve` env's lib dir —
needed because one of vllm's transitive deps (sqlite3's ICU extension) links
against the system libstdc++ at runtime, which is too old on this machine
(missing `CXXABI_1.3.15+`) and fails with an `ImportError` on `import
sqlite3` otherwise. **Always start vLLM through this script**, not a raw
`vllm serve ...` command — a hand-run command skips this fix.

Wait for `Application startup complete` (or similar Uvicorn-ready line) in
the logs, or check readiness from another terminal:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

Skip this whole step and pass `--no-vllm` to `robot_web_app.py` (step 2) to
load stage-1 locally instead — no server needed, but slower per-request.

## 2. `robot_web_app.py` (detector + OOD split + web UI)

**Terminal 2:**

```bash
cd /home/seher/Projects/object_detection/robot_demo
conda activate sam_oed
python robot_code/robot_web_app.py --host 127.0.0.1 --port 7860 --conf-thres 0.9 --energy-thres -8.0
```

Open **http://127.0.0.1:7860/** in a browser. This serves
`live_feed_demo.html` and exposes `/api/run`, `/api/pushed/latest`,
`/api/submit/<job_id>`. Confirm it's actually up before moving on:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7860/
```

should print `200`.

Two ways to get a *live* feed into the page's "Live (pushed frames)" panel —
pick one, not both:

- **`--ros-topic <topic>`**: the server subscribes via `rclpy` itself
  (`ros_live_loop`), no separate client needed. **Not usable on this
  machine** — no ROS 2 / `rclpy` install found in any conda env or
  `/opt/ros` as of 2026-09-15. Use option 3 (below) instead.
- **`robot_client.py`** (option 3): a separate process pushes frames to
  `/api/run` over plain HTTP. Works from anywhere on the network, no rclpy
  required on this machine.

## 3. `robot_client.py` (pushes frames into the live view)

**Terminal 3** (only start this after step 2's server is confirmed up):

```bash
cd /home/seher/Projects/object_detection/robot_demo
conda activate sam_oed
python robot_code/robot_client.py
```

Defaults (no flags needed) — both are now hardcoded in the script:

- `--server` → `http://127.0.0.1:7860` (the `robot_web_app.py` from step 2)
- `--source` → `http://10.254.90.201:8081/stream?topic=/camera/camera/color/image_raw&type=mjpeg`
  — the robot's `web_video_server` MJPEG stream (ROS topic
  `/camera/camera/color/image_raw`). Full topic list:
  `http://10.254.90.201:8081/`.

Then in the browser, open the **"Live (pushed frames)"** panel — it polls
`/api/pushed/latest`. Anything still "unknown" after both VLM stages shows up
there for you to type a name in; that name also resolves later frames near
the same spot via the server's shared spatial cache.

Useful overrides:

- `--source ''` — ignore the default stream and use a local webcam via
  `--cam-index` instead.
- `--source Videos/15_seconds.mp4` — dry-run against a repo sample video.
- `--skip-vlm` — detector + energy-OOD only, no VLM calls per frame; fast
  smoke test that the live pipe itself works before trusting VLM output.
- `--interval` — seconds between frames sent (default 1.0). Each frame is a
  full detector+VLM pass on the server, not a lightweight ping — don't set
  this too low.

## Shutting down

Stop in reverse start order, `Ctrl+C` in each terminal: `robot_client.py`
(3) → `robot_web_app.py` (2) → `start_vllm_server.sh` (1). Stopping the web
app first just means the client will start failing its pushes (harmless,
just Ctrl+C it too); stopping vLLM first only matters if the web app isn't
running with `--no-vllm`.

To confirm everything actually stopped:

```bash
ps aux | grep -iE "robot_client|robot_web_app|vllm" | grep -v grep
```

should print nothing.

## Troubleshooting

- **Nothing shows up in the browser / connection refused on 7860**: step 2
  (`robot_web_app.py`) isn't actually running — check its terminal for a
  crash, or `ps aux | grep robot_web_app`.
- **`robot_client.py` can't reach the camera stream**: the robot's IP is
  DHCP-assigned and has changed before (`10.254.90.96` → `10.254.90.201` as
  of 2026-09-17). Verify with `curl http://<ip>:8081/`; if it moved, update
  `DEFAULT_STREAM_URL` at the top of `robot_code/robot_client.py`.
- **vLLM server fails on `import sqlite3` at startup**: you probably ran
  `vllm serve` by hand instead of `start_vllm_server.sh` — see step 1's
  `LD_LIBRARY_PATH` note.
- **`--ros-topic` on `robot_web_app.py`**: not usable on this machine — no
  ROS 2 / `rclpy` install found in any conda env or `/opt/ros` (checked
  2026-09-15). Use `robot_client.py` (step 3) instead.

## Notes

- Stage-2 (the bigger `--verify-model-id` checkpoint, Qwen3-VL-8B by default)
  only runs when stage-1 says "unknown" — it stays on a local `transformers`
  path (lazy-loaded), not served by vLLM.
- The robot camera stream address (`10.254.90.201:8081`) is whatever network
  the robot happens to be on — if it moves, update `DEFAULT_STREAM_URL` at
  the top of `robot_code/robot_client.py`.
