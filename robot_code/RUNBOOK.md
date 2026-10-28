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
python robot_code/robot_web_app.py --host 127.0.0.1 --port 7860 --conf-thres 0.9 --energy-thres -6.84
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

### Baseline: Qwen3-VL direct (no detector)

The page can put the pipeline next to the "just ask the VLM" baseline. Here
Qwen3-VL gets the **whole** image and has to find and name every object
itself (it returns a JSON list of `bbox_2d` boxes and labels), with no
Faster R-CNN or energy OOD split in front of it.

- **Single image:** pick a sample or upload one, click **Run detection**, then
  **Run Qwen direct** (`/api/qwen-direct/run`). The green boxes are Qwen's.
  The *Pipeline vs Qwen direct* card compares the time and object count. The
  model's raw reply is under *Raw model output*.
- **Live** (needs `--realsense` or `--ros-topic`): click **Compare with … direct**
  in the live view's titlebar. It works in the default camera-only view too.
  The view splits into *Our pipeline* | *Qwen3-VL direct*, and each side shows
  a badge with ms/frame, fps and object count. Click it again (or stop the live
  view, or close the tab) to switch the Qwen worker off.
- The two take turns on the GPU (`GPU_LOCK` in `robot_web_app.py`), so
  **ms/frame** is each path's own compute time, and **fps** is what you
  actually see. Expect Qwen direct to take several seconds per frame on the
  3090. While the comparison is on, the pipeline's fps drops too, because it
  waits for Qwen's turn.
- Flags: `--qwen-direct-model-id` (default: the stage-1 model, already loaded,
  so no extra VRAM) and `--qwen-direct-max-tokens` (default 1024, about 25
  tokens per object; a reply cut off at the limit still keeps every complete
  box).
- If the green boxes come out shifted or scaled: `qwen3_generator.py`
  assumes Qwen3-VL's 0–1000 normalized box coordinates
  (`GROUNDING_COORD_SCALE`). Compare against the raw output.

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

---

# Main page — run everything from one browser tab (`robot_code/hub/`)

```powershell
cd robot-demo
.venv\Scripts\python.exe robot_code\hub\hub_app.py
```

Open **http://127.0.0.1:7800/** in Chrome or Edge. The page is laid out as the
demo is presented, in two parts:

**Part 1 · OOD (Module 1, Ask & Remember, no VLM)**
1. **Teach**: the robot stops on an unknown object, asks its name, and you capture views.
2. **Replay**: explore mode. Taught objects are recognized from memory and nothing is asked.
3. **Teach & Replay**: side by side at `/compare?set=ood`. Module 1 is started with
   `--dual-view`, which runs a second, explore-mode `Teacher` on the same camera frames
   and the same memory and shows it at `http://127.0.0.1:7861/recognize`. An object
   shows up as recognized on the right as soon as its capture is done on the left.
- **Fresh demo: back up & clear** moves `teach_mode/teach_memory/` to
  `teach_mode/teach_memory_backups/<YYYYmmdd_HHMMSS>/` and starts with an empty memory.
  **Restore** puts a backup back (backing up the current memory first). Both work with
  Module 1 running (it reloads its memory) or stopped (the hub moves the folders).

**Part 2 · OED (Modules 2 and 3, with Qwen3-VL)**
1. **OOD + VLM**: Module 2 on the RealSense (our pipeline).
2. **VLM only**: Module 3, Qwen3-VL direct on the whole image.
3. **OOD + VLM vs VLM**: Module 2 opened with `?compare=1`, which switches its
   "Compare with Qwen3-VL direct" view on. A **Comparison results** table under the
   frames shows each method's mean/median ms per frame, fps (1000 / mean ms), objects
   per frame and the names given, since the last **Reset**. **Download CSV**
   (`/api/compare/metrics.csv`) has one row per frame, including sample-image runs of
   "Run detection" / "Run Qwen direct" on Module 2's full page (`?full=1`).
4. **Robot assistant**: Module 2's `/assistant` page ("where are my X?").

**Part 3 · OOD + Memory + OED (Part 1's memory in front of Part 2's VLM)**
1. **OOD + Memory + VLM**: Module 2's `/hybrid` page (`hybrid_pipeline.py` + `.html`).
   Every OOD box is first matched against Module 1's memory (`teach_mode/teach_memory/`,
   the same DINOv2 embedder, 0.62 threshold and 0.05 margin as Module 1). Only the boxes
   memory can't name go to Qwen3-VL (stage 1, then stage 2 if unsure; no human step).
   The page shows two screens (*OOD detection only* | *OOD + Memory + VLM*), then each of
   the three steps on its own image, with per-box crops, similarities and VLM sources, plus
   session totals and the memory gallery. It runs inside the Module 2 server and replaces
   Module 2's pipeline on the live feed while the page is open. The page turns it on when
   it opens, and turns it off when it closes or after 8 s without a poll. Module 1's memory
   is only read here, and it is read again whenever `memory.json` changes, so objects taught
   in Part 1 are used from the next frame on. Flags: `--hybrid-memory-dir`,
   `--hybrid-accept-thres`.

**Load all models** (top right) starts Module 1 and Module 2 once, about a minute,
~13 GB of GPU memory together. After that no scene reloads anything: both modules stay
loaded, and a scene button only hands the RealSense over (`POST /api/service/<name>/camera`
on the hub, `/api/camera` on each module, which closes or reopens the camera without
exiting). A handover takes about 1.5–3 s. Without Load all models, a scene starts the
module it needs the first time, and keeps it loaded afterwards. Modules can also be started
loaded but without the camera with `--camera-off`.

A scene button takes the RealSense from whatever else holds it without asking, so going
from one scene to another is one click. The module pages start by
themselves; Chrome only lets a page speak after it has been clicked, so Module 1 shows a
"click anywhere to turn on the robot's voice" bar until then (start Chrome with
`--autoplay-policy=no-user-gesture-required` should skip even that; not tested here). Each part shows its
modules' state with a **Stop** button (Part 2's stops Module 3 and Module 2), and
**Stop everything** (top right) stops all modules and frees the camera and GPU.

Under **Advanced** are one card per service (Module 1, Module 2, Module 3) with their
options, thresholds and Start / Restart / Stop / Open / Log buttons, the module
windows, and **Tools** (Module 1 tests, threshold calibration, memory listings,
RealSense check, GPU status). Copies started by hand in a terminal are detected
too, and Stop works on them.

Only one service can own the RealSense: the page asks before stopping the
current owner. Watch the GPU meter. Module 2 loads Qwen (~9 GB, plus ~17 GB for
the 8B second opinion on first use), so running both modules can run out of
GPU memory when Module 2's stage 2 kicks in.

Service output goes to `robot_code/hub/logs/<name>.log`. Videos with non-ASCII
names are copied once to `robot_code/hub/cache/` (OpenCV can't open them
directly), and "record session" saves to `robot_code/hub/recordings/`, which
also shows up in the replay list.

---

# Module 1 — "Ask & Remember" (`robot_code/teach_mode/`)

A separate, VLM-free pipeline: the energy-based OOD detector flags an unfamiliar
object → the robot asks the user what it is (speech or typing) → the user captures
several views in a capture window → the object is stored in a multi-view memory
→ after it's moved, the robot recognizes it and says so. It doesn't load Qwen, so
it needs no vLLM server and only ~2–3 GB of GPU memory.

## Run (Windows, the venv at `robot-demo/.venv`)

Only one process can own the RealSense: stop any `robot_web_app.py --realsense`
first.

```powershell
cd robot-demo
.venv\Scripts\python.exe robot_code\teach_mode\teach_app.py
```

Then open **http://127.0.0.1:7861/** in **Chrome or Edge** (speech input uses the
browser's Web Speech API, which needs internet) and click **Start**.
The first run downloads `facebook/dinov2-small` (~90 MB) from Hugging Face.

Defaults match the tested Module 2 settings: `--weights model/model_final.pth
--conf-thres 0.9 --energy-thres -6.84`.

## Demo flow (the 7 steps on the page's step bar)

1–2. Put objects on the table. The left view shows the detector's VOC (blue) vs
     Unknown (red) split.
3.   The robot asks about one unknown at a time ("What is this called?"). Say
     "it's a TV remote" (hands-free mic listens after the question), or type it.
     **Skip** = never ask about that object again this session.
4.   The **capture window** opens: hold the object up, press **Capture** (or
     <kbd>Space</kbd>, or say "capture") for front / side / back / angled.
     The purple box is what gets saved. If the object isn't detected, the dashed
     guide box is used instead. Delete bad shots with ✕. **Done** (<kbd>Enter</kbd>)
     saves.
5.   The Memory card shows the object with all its views.
6.   Move the object somewhere else.
7.   Once it comes to rest, the robot says "I recognize this object from memory.
     It is the TV remote." (green box).

## Explore mode (after teaching)

Click **Explore** at the top of the Module 1 page (or the **Explore** quick-start
button on the main page). No restart is needed; **Teach** switches back. Then look
around with the camera:

- Known VOC objects are shown as usual.
- An OOD box is shown **only** if the detector's score is ≥ 0.95 *and* memory
  recognizes it (green, "memory: light 0.91"). Every other OOD box is hidden and
  never asked about.
- Each remembered object is announced once when it comes into view, and again
  when it reappears after being out of sight.

Change the 0.95 with `--explore-ood-conf` (or "explore OOD ≥" on the main page).
Start directly in this mode with `--mode explore`.

## Tuning

- **Wrong object recognized / right object missed:** teach 3–4 objects with
  several views each, press Skip on a few non-objects, then run
  `.venv\Scripts\python.exe robot_code\teach_mode\calibrate.py` (add
  `--embedders dinov2-small dinov2-base clip-b32` to compare). It suggests an
  `--accept-thres`; restart the app with it. `--show-scores` prints the closest
  memory score on unknown boxes too.
- **Taught object gets a VOC label after being moved** (e.g. "couch"): add
  `--match-known-boxes`.
- **Detector misses the object:** lower `--conf-thres` (e.g. 0.7).

## Files / data

- Memory: `robot_code/teach_mode/teach_memory/` (`memory.json` + `views/*.jpg`;
  `negatives/` holds crops you skipped). Delete the folder to start over, or
  edit live from the Memory card (rename / delete / add views).
- Replay without a camera: `--source clip.mp4 --loop` (record one with
  `--record clip.mp4`). OpenCV can't open non-ASCII paths on Windows, so
  rename/copy files like `Samples/화면 녹화 ….mp4` first.
- Tests (no GPU needed): `.venv\Scripts\python.exe -m pytest robot_code\teach_mode\tests`

---

# Robot assistant — "Robot, where are my headphones?" (`/assistant`)

A page on Module 2's server (`robot_web_app.py`): you ask, by voice or by typing, where
something is, and the robot answers on the page and out loud, e.g. *"I found your
headphones on the sofa, next to the teddy bear, about 1.6 m ahead, slightly to my
left"*, with a snapshot. It answers from what it sees **right now**; failing that, from
**where it last saw it**; failing that, it asks **Qwen3-VL to search the whole frame**
for objects the VOC detector never boxes. Every asked-for object stays on a watch list,
so the robot also speaks up on its own: *"There they are! I found your headphones on the
table."* when an object shows up, or *"…they are now on the table"* when it moves. It
only finds and locates things. You drive the robot yourself.

Code: `robot_assistant.py` (parsing, matching, answers, watch list, grounding fallback),
`scene_memory.py` (last sighting per object, "on X / next to Y", depth position),
`robot_assistant.html`.

## Two machines: laptop on the robot, GPU desktop on the desk

1. **Desktop** (RTX 3090), from `robot-demo/`:

   ```powershell
   .venv\Scripts\python.exe robot_code\robot_web_app.py --weights model\model_final.pth --no-vllm `
       --host 0.0.0.0 --conf-thres 0.9 --energy-thres -6.84 --verify-model-id Qwen/Qwen3-VL-4B-Instruct
   ```

   `--host 0.0.0.0` lets the laptop connect. Allow inbound TCP 7860 when Windows Firewall
   asks, or as admin run
   `New-NetFirewallRule -DisplayName robot_web_app -Direction Inbound -Protocol TCP -LocalPort 7860 -Action Allow`.
   Find the desktop's IP with `ipconfig`.
   `--verify-model-id Qwen/Qwen3-VL-4B-Instruct` reuses the 4B for stage 2 too. The
   default 8B (~17 GB) doesn't fit next to the 4B on a 24 GB card: Windows spills it
   into shared memory and every Qwen call, including the assistant's, crawls.

2. **Laptop** (RealSense D435 on USB 3). Copy over just `robot_code/robot_client.py`
   and `robot_code/frame_sources.py`. Neither imports torch or detectron2.

   ```powershell
   pip install -r requirements-laptop.txt   # opencv-python numpy requests pyrealsense2
   python robot_client.py --server http://<desktop-ip>:7860 --source realsense --depth --interval 0.3
   ```

   `--depth` sends the aligned depth (16-bit PNG, half resolution) with each frame,
   which is where the distance and direction in the answers come from.

3. **Laptop browser.** Chrome only allows the microphone on https or localhost, so
   do one of these once:
   - `chrome://flags/#unsafely-treat-insecure-origin-as-secure`: add
     `http://<desktop-ip>:7860`, set it to Enabled, and relaunch Chrome. Then open
     `http://<desktop-ip>:7860/assistant`.
   - or (Windows laptop, admin) `netsh interface portproxy add v4tov4 listenaddress=127.0.0.1 listenport=7860 connectaddress=<desktop-ip> connectport=7860`,
     then open `http://127.0.0.1:7860/assistant`.

   Chrome's speech recognition also needs internet on the laptop. Click the page once,
   so the browser lets it talk.

On the page, **Hands-free** is on by default: anything you say starting with "Robot, …"
is a question. Or hold **🎤** / **Space**, or type. Click an object card to ask about it.
✕ on a "Looking for" chip stops watching for that object. Name still-unknown objects on
`/?full=1` ("Needs your input"); until then they can't be found by name.

## One machine (development)

With the RealSense on the desktop instead: add `--realsense --depth` to the desktop
command above and open `http://127.0.0.1:7860/assistant`.

## Assistant flags (`robot_web_app.py`)

- `--depth`: with `--realsense`, also stream aligned depth (for `robot_client.py`
  it's that script's own `--depth`).
- `--motion-flush-thres 10`: when consecutive frames differ by more than this
  (0–255 mean diff of 160x90 grayscale), the live path's position-keyed name cache is
  cleared. Otherwise, once the robot moves, a different object landing on the same
  pixels would inherit the old name. 0 disables it (static camera).
- `--no-grounding`: never ask Qwen to search the whole frame. Answers are then "not
  seen yet" unless the detector/VLM pipeline has named the object.

## Checking it from the command line (server running)

`curl.exe -X POST http://127.0.0.1:7860/api/assistant/ask -H "Content-Type: application/json" -d "{\"text\":\"where is the bottle\"}"`
asks from the command line. `GET /api/assistant/state` shows what the page polls:
camera, watch list, the scene memory, and events.

## Known limits

- Relations come from one 2D view, checked with depth: "on" needs the target to rest
  inside a surface's box at about the same distance.
- Stage-1/2 names are one word (`_normalize_label`), so a teddy bear may be
  remembered as "teddy". Matching tolerates that ("teddy bear" ↔ "teddy").
- The `/?full=1` "Needs your input" list for pushed frames grows while driving,
  because it's keyed by screen position.
