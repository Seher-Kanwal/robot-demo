# Setting up robot-demo on a Windows laptop

This rebuilds the environment that runs on the lab desktop (RTX 3090). It was checked there on 2026-10-23:

| Component | Version |
|---|---|
| Windows | 10 / 11, 64-bit |
| Python | 3.12.5 (venv at `robot-demo\.venv`) |
| PyTorch | 2.11.0+cu128, torchvision 0.26.0+cu128 |
| detectron2 | 0.6, source at `third_party\detectron2` (commit `b0fe2f0`), **not compiled** |
| ImageMagick | 7.1.2-34 Q16-HDRI x64 (DLL build, used by Wand) |
| transformers | 5.7.0 (Qwen3-VL, DINOv2) |
| OpenCV | 5.0.0.93 |
| pyrealsense2 | 2.59.1 |
| CLIP | openai/CLIP @ `d05afc4` |

You do **not** need the CUDA Toolkit, Visual Studio/MSVC or conda. The torch wheel includes the CUDA runtime. detectron2 runs from source without its compiled `_C` extension, which only provides deformable conv and fast COCO eval. The demo uses neither.

---

## 1. What the laptop can run

All modules fall back to CPU when no NVIDIA GPU is found, but Qwen3-VL on CPU takes minutes per frame.

| Laptop GPU | Module 1 (teach, port 7861) | Module 2 / 3 (Qwen3-VL, port 7860) |
|---|---|---|
| None / Intel / AMD | Works on CPU (`-Cpu` setup), slow (a few fps at best) | Not usable live. Use the two-machine setup in `robot_code\RUNBOOK.md` |
| NVIDIA 6–8 GB | Works (~2–3 GB) | Does not fit (the 4B model alone is ~9 GB) |
| NVIDIA 12 GB | Works | Tight, Module 2 alone with the 4B model only |
| NVIDIA 16 GB+ | Works | Works with `--verify-model-id Qwen/Qwen3-VL-4B-Instruct` (the hub already passes this) |

Never let the 8B model load on a laptop (~17 GB). Windows spills it into shared memory and every Qwen call crawls.

Check the GPU with `nvidia-smi`. It needs **driver R570 or newer** for cu128 wheels. Update from nvidia.com/drivers if it's older.

---

## 2. Install the prerequisites

1. **Python 3.12** (64-bit) from python.org. Tick *"py launcher"* in the installer. Check: `py -3.12 --version`.
2. **Git for Windows** (git-scm.com). It's used to fetch CLIP and, if needed, detectron2.
3. **NVIDIA driver** R570+ (only if the laptop has an NVIDIA GPU).
4. **Chrome or Edge**. The pages use the browser's speech API, which needs internet.
5. **RealSense D435**: plug it into a **USB 3** port (blue / SS). The `pyrealsense2` wheel includes the driver. Intel RealSense Viewer is optional, for checking the camera.

---

## 3. Copy the project from the desktop

Copy the **whole `Robot demonstration` folder, except `robot-demo\.venv`**. A venv can't be moved between machines. A plain `git clone` is not enough because:

- `model\model_final.pth` (316 MB, detector weights) is git-ignored.
- The desktop has many changes that aren't committed yet (hub, teach_mode, hybrid pipeline, taught memory).
- `third_party\` (detectron2 source + ImageMagick DLLs) is outside the repo.

Keep this layout, with `third_party` **next to** `robot-demo`:

```
Robot demonstration\
├─ robot-demo\              ← code, model\model_final.pth, robot_code\teach_mode\teach_memory\
│                              (skip .venv\ ; Samples\ is optional, 127 MB of sample images)
└─ third_party\
   ├─ detectron2\           ← 14 MB, source only
   └─ imagemagick\          ← 61 MB, ImageMagick DLL build
```

That's about 600 MB in total. A USB stick or a network share works. With robocopy from the desktop:

```powershell
robocopy "E:\Presentation\LBA Project\Robot demonstration" "F:\Robot demonstration" /E /XD .venv __pycache__ .pytest_cache
```

**Optional, saves ~10 GB of downloads:** also copy these model caches from the desktop to the same place on the laptop (`%USERPROFILE%` = `C:\Users\<you>`):

| Copy from desktop | Size | Needed by |
|---|---|---|
| `%USERPROFILE%\.cache\huggingface\hub\models--facebook--dinov2-small` | ~90 MB | Module 1, Part 3 |
| `%USERPROFILE%\.cache\huggingface\hub\models--Qwen--Qwen3-VL-4B-Instruct` | ~9 GB | Modules 2 and 3 |
| `%USERPROFILE%\.cache\clip\ViT-B-32.pt` | 340 MB | Module 2's OOD memory |

Without them, the models download automatically on first run.

> If the project path on the laptop has non-ASCII characters (e.g. Korean folder names), move it to a plain path such as `C:\robot\Robot demonstration`. OpenCV and some DLL loaders fail on non-ASCII paths.

---

## 4. Build the environment (one command)

Open **PowerShell** in the laptop's `robot-demo` folder:

```powershell
# laptop with an NVIDIA GPU
powershell -ExecutionPolicy Bypass -File setup_windows.ps1

# laptop without an NVIDIA GPU
powershell -ExecutionPolicy Bypass -File setup_windows.ps1 -Cpu
```

It takes 10–20 minutes, mostly downloading torch (~3 GB for CUDA). The script:

1. Creates `.venv` with Python 3.12.
2. Installs torch 2.11.0 / torchvision 0.26.0 (cu128, or the CPU build with `-Cpu`).
3. Uses `..\third_party\detectron2` (clones it at `b0fe2f0` if missing) and installs detectron2's pure-Python dependencies (fvcore, iopath, pycocotools, yacs, omegaconf, hydra-core, …).
4. Installs `requirements.txt` + `accelerate`.
5. Installs OpenAI CLIP at the same commit as the desktop.
6. Writes two files into `.venv\Lib\site-packages\`:
   - `detectron2_src.pth`: one line, the full path of `third_party\detectron2`. This is how `import detectron2` works without building it.
   - `zz_magick_home.pth`: sets `MAGICK_HOME` to `third_party\imagemagick` so Wand finds the ImageMagick DLLs.
7. Prints an import check. It should look like this (desktop output):

```
python       3.12.5
torch        2.11.0+cu128 | CUDA: True | NVIDIA GeForce RTX 3090
detectron2   0.6 E:\...\third_party\detectron2\detectron2\__init__.py
ImageMagick  ImageMagick 7.1.2-34 Q16-HDRI x64 a0c4bf7:20261014 https://imagemagick.org
transformers 5.7.0 | opencv 5.0.0
```

On a GPU laptop, `CUDA: False` means the driver is too old or the CPU torch got installed. Rerun step 2 by hand (section 7).

You can rerun the script. Steps that are already done are skipped or pip reports "already satisfied".

---

## 5. Run

From `robot-demo` with the RealSense plugged in:

**Everything from one page (hub):**

```powershell
.venv\Scripts\python.exe robot_code\hub\hub_app.py
```

Open http://127.0.0.1:7800/. On a laptop GPU under 16 GB, **don't press "Load all models"** (~13 GB). Start only Part 1 (Module 1).

**Module 1 only (teach mode), the lightest:**

```powershell
.venv\Scripts\python.exe robot_code\teach_mode\teach_app.py
```

Open http://127.0.0.1:7861/ in Chrome/Edge and click **Start**. Defaults are already `--weights model\model_final.pth --conf-thres 0.9 --energy-thres -6.84`. Without a camera, use a video: `--source path\to\video.mp4 --loop`.

**Module 2 (VLM) by hand** (16 GB+ GPU):

```powershell
.venv\Scripts\python.exe robot_code\robot_web_app.py --weights model\model_final.pth --no-vllm `
    --realsense --conf-thres 0.9 --energy-thres -6.84 --verify-model-id Qwen/Qwen3-VL-4B-Instruct
```

Always pass `--no-vllm` on Windows (vLLM doesn't run there) and `--weights model\model_final.pth` (the built-in default path doesn't exist).

**Rules that apply on any machine:**

- Only **one** process can own the RealSense. Stop the other module or use the hub's scene buttons.
- Only run **one** `robot_web_app.py`. Two copies each load Qwen and run out of memory.
- `.venv\Scripts\python.exe` shows up as **two** `python.exe` processes (launcher + real interpreter). That's one run, not two. To stop it, close the terminal or use the hub's Stop button.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: MagickWand shared library not found` | `third_party\imagemagick` is missing or not next to `robot-demo`. Copy it, or install ImageMagick 7 Q16-HDRI x64 **DLL** build from imagemagick.org (tick "Install development headers and libraries for C and C++"). Then edit `.venv\Lib\site-packages\zz_magick_home.pth` to point to its install folder. |
| `ModuleNotFoundError: detectron2` | The path in `.venv\Lib\site-packages\detectron2_src.pth` is wrong. It must be the folder that *contains* the inner `detectron2\` folder. |
| `... detectron2._C` error | A config is using deformable conv / custom ops. Install VS 2022 Build Tools (C++) + CUDA Toolkit 12.8, then from an "x64 Native Tools" prompt in `third_party\detectron2`: `..\..\robot-demo\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation`, and delete `detectron2_src.pth`. Not needed for the demo. |
| `FileNotFoundError: ...model_final.pth` | Copy `model\model_final.pth` from the desktop. |
| RealSense: "Device or resource busy" / "Couldn't resolve requests" | Another process holds the camera. Stop it. The apps retry for 30 s. Also try a different USB 3 port, not a hub. |
| RealSense not found at all | Check it's on USB 3 (RealSense Viewer shows the USB type). Unplug and replug it. |
| `CUDA out of memory`, or Qwen calls take minutes | The GPU is too small for that module (table in section 1). Run Module 1 only, or use the two-machine setup. |
| "The paging file is too small" (os error 1455) | Two copies of Module 2, or too little virtual memory. Run one copy, and set Windows virtual memory to "System managed" on a drive with 30+ GB free. |
| Speech input does nothing | Use Chrome/Edge on `http://127.0.0.1`, with internet, and click the page once. |
| Hugging Face download stalls | Copy the cache folders from the desktop (section 3). |

---

## 7. Manual setup (what the script does)

From `robot-demo`, if you prefer to type it yourself:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools==78.1.0
.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
#   (no NVIDIA GPU: --index-url https://download.pytorch.org/whl/cpu)
.venv\Scripts\python.exe -m pip install fvcore==0.1.5.post20221221 iopath==0.1.10 pycocotools==2.0.11 yacs==0.1.8 tabulate cloudpickle tensorboard omegaconf==2.3.1 hydra-core==1.3.7 black packaging
.venv\Scripts\python.exe -m pip install -r requirements.txt accelerate==1.15.0
.venv\Scripts\python.exe -m pip install git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6

# detectron2 on sys.path (no build) and MAGICK_HOME for Wand. Use your real full paths:
Set-Content -Encoding ascii .venv\Lib\site-packages\detectron2_src.pth "C:\robot\Robot demonstration\third_party\detectron2"
Set-Content -Encoding ascii .venv\Lib\site-packages\zz_magick_home.pth "import os; os.environ.setdefault('MAGICK_HOME', r'C:\robot\Robot demonstration\third_party\imagemagick')"
```

If `third_party\detectron2` wasn't copied:
`git clone https://github.com/facebookresearch/detectron2.git ..\third_party\detectron2` then `git -C ..\third_party\detectron2 checkout b0fe2f0`.
