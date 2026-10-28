# Rebuilds the robot-demo Python environment on a Windows machine (laptop or desktop),
# matching the tested setup: Python 3.12, torch 2.11.0+cu128, detectron2 from source
# (not compiled), ImageMagick for Wand. See SETUP_WINDOWS_LAPTOP.md.
#
# Run from the robot-demo folder:
#   powershell -ExecutionPolicy Bypass -File setup_windows.ps1          # NVIDIA GPU
#   powershell -ExecutionPolicy Bypass -File setup_windows.ps1 -Cpu     # no NVIDIA GPU
param(
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$Repo = $PSScriptRoot
$ThirdParty = Join-Path (Split-Path $Repo -Parent) "third_party"
$D2 = Join-Path $ThirdParty "detectron2"
$Magick = Join-Path $ThirdParty "imagemagick"
$VenvPy = Join-Path $Repo ".venv\Scripts\python.exe"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# 1. venv (Python 3.12)
Step "1/7 Python 3.12 venv at .venv"
if (-not (Test-Path $VenvPy)) {
    & py -3.12 -m venv (Join-Path $Repo ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Python 3.12 not found. Install it from python.org (tick 'py launcher') and rerun." }
}
& $VenvPy -m pip install --upgrade pip wheel "setuptools==78.1.0"

# 2. torch / torchvision
Step "2/7 PyTorch 2.11.0 + torchvision 0.26.0"
if ($Cpu) {
    & $VenvPy -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
} else {
    & $VenvPy -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
}
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

# 3. detectron2 source (pure Python, no build) + its dependencies
Step "3/7 detectron2 source at $D2"
if (-not (Test-Path (Join-Path $D2 "detectron2\__init__.py"))) {
    New-Item -ItemType Directory -Force $ThirdParty | Out-Null
    git clone https://github.com/facebookresearch/detectron2.git $D2
    if ($LASTEXITCODE -ne 0) { throw "git clone failed (is Git for Windows installed?)" }
    git -C $D2 checkout b0fe2f0
}
& $VenvPy -m pip install "fvcore==0.1.5.post20221221" "iopath==0.1.10" "pycocotools==2.0.11" `
    "yacs==0.1.8" tabulate cloudpickle tensorboard "omegaconf==2.3.1" "hydra-core==1.3.7" black packaging
if ($LASTEXITCODE -ne 0) { throw "detectron2 dependency install failed" }

# 4. repo requirements (torch line is already satisfied by step 2)
Step "4/7 requirements.txt + accelerate"
& $VenvPy -m pip install -r (Join-Path $Repo "requirements.txt") "accelerate==1.15.0"
if ($LASTEXITCODE -ne 0) { throw "requirements.txt install failed" }

# 5. OpenAI CLIP (same commit as the desktop)
Step "5/7 OpenAI CLIP"
& $VenvPy -m pip install "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
if ($LASTEXITCODE -ne 0) { throw "CLIP install failed" }

# 6. .pth files: detectron2 on sys.path, MAGICK_HOME for Wand
Step "6/7 site-packages .pth files"
$SitePkgs = (& $VenvPy -c "import sysconfig; print(sysconfig.get_paths()['purelib'])").Trim()
Set-Content -Encoding ascii (Join-Path $SitePkgs "detectron2_src.pth") $D2
if (Test-Path (Join-Path $Magick "CORE_RL_MagickWand_.dll")) {
    Set-Content -Encoding ascii (Join-Path $SitePkgs "zz_magick_home.pth") `
        "import os; os.environ.setdefault('MAGICK_HOME', r'$Magick')"
    Write-Host "MAGICK_HOME -> $Magick"
} else {
    Write-Warning "No ImageMagick DLLs in $Magick. Copy that folder from the desktop, or install ImageMagick 7 (Q16-HDRI x64 DLL build) system-wide. See SETUP_WINDOWS_LAPTOP.md."
}

# 7. check
Step "7/7 import check"
& $VenvPy -c @"
import torch, detectron2, wand.version, clip, transformers, pyrealsense2, cv2
print('python      ', __import__('sys').version.split()[0])
print('torch       ', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')
print('detectron2  ', detectron2.__version__, detectron2.__file__)
print('ImageMagick ', wand.version.MAGICK_VERSION)
print('transformers', transformers.__version__, '| opencv', cv2.__version__)
"@
if ($LASTEXITCODE -ne 0) { throw "import check failed" }

if (-not (Test-Path (Join-Path $Repo "model\model_final.pth"))) {
    Write-Warning "model\model_final.pth is missing (git-ignored). Copy it from the desktop."
}
Write-Host "`nDone. Start with:  .venv\Scripts\python.exe robot_code\hub\hub_app.py" -ForegroundColor Green
