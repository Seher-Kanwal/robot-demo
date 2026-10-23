"""
Pick recognition thresholds from real data instead of guessing.

Reads the views already stored in teach_memory/ (teach a few objects in the app first,
3+ objects with 4+ views each is a good start), re-embeds them with each embedder you
list, and reports for each one:

  same object      -- each view vs. the *other* views of its own object (should be high)
  other objects    -- each view vs. the views of every other object (should be low)
  skipped crops    -- objects you pressed Skip on in the app (teach_memory/negatives/),
                      vs. every taught object (should be low)
  leave-one-out    -- how often a view's nearest other object is its own

and suggests --accept-thres halfway between the low end of "same object" and the high
end of "other objects / skipped crops". Read-only: it never modifies memory.json.

    .venv/Scripts/python.exe robot_code/teach_mode/calibrate.py
    .venv/Scripts/python.exe robot_code/teach_mode/calibrate.py --embedders dinov2-small dinov2-base clip-b32
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

FILE = Path(__file__).resolve()
if str(FILE.parent) not in sys.path:
    sys.path.append(str(FILE.parent))

from embedder import DEFAULT_EMBEDDER, EMBEDDERS, Embedder  # noqa: E402


def describe(name, values):
    if not len(values):
        print(f"  {name:16s} (no data)")
        return
    v = np.asarray(values)
    p5, p50, p95 = np.percentile(v, [5, 50, 95])
    print(f"  {name:16s} n={len(v):4d}  min={v.min():.3f}  p5={p5:.3f}  median={p50:.3f}  p95={p95:.3f}  max={v.max():.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--memory-dir", default=str(FILE.parent / "teach_memory"))
    parser.add_argument("--embedders", nargs="+", default=[DEFAULT_EMBEDDER], choices=sorted(EMBEDDERS))
    args = parser.parse_args()

    mem_dir = Path(args.memory_dir)
    if not (mem_dir / "memory.json").exists():
        sys.exit(f"nothing to calibrate yet: {mem_dir / 'memory.json'} doesn't exist -- teach a few objects "
                 f"in Module 1 first")
    with open(mem_dir / "memory.json", "r", encoding="utf-8") as f:
        objects = json.load(f)["objects"]
    images, owners = [], []
    for k, obj in enumerate(objects):
        for v in obj["views"]:
            img = cv2.imread(str(mem_dir / v["file"]))
            if img is not None:
                images.append(img)
                owners.append(k)
    owners = np.asarray(owners)
    negatives = [img for img in (cv2.imread(str(p)) for p in sorted((mem_dir / "negatives").glob("*.jpg")))
                 if img is not None]
    print(f"{len(objects)} object(s), {len(images)} view(s), {len(negatives)} skipped crop(s) in {mem_dir}")
    if len(objects) < 2:
        print("teach at least 2 objects first -- 'other objects' needs something to compare against")

    for name in args.embedders:
        emb = Embedder(name)
        E = emb.embed(images)
        S = E @ E.T
        np.fill_diagonal(S, -np.inf)
        same, other, correct = [], [], 0
        for i in range(len(images)):
            mine = owners == owners[i]
            if mine.sum() > 1:
                same.append(S[i, mine].max())
            if (~mine).any():
                other.append(S[i, ~mine].max())
            per_obj = [S[i, owners == k].max() if (owners == k).any() else -np.inf for k in range(len(objects))]
            correct += int(np.argmax(per_obj) == owners[i])
        neg = (emb.embed(negatives) @ E.T).max(axis=1) if negatives and len(images) else []

        print(f"\n== {name} ({emb.model_id})")
        describe("same object", same)
        describe("other objects", other)
        describe("skipped crops", neg)
        if len(images):
            print(f"  leave-one-out nearest object: {correct}/{len(images)} correct")
        impostors = list(other) + list(neg)
        if len(same) and len(impostors):
            lo, hi = float(np.percentile(same, 5)), float(np.percentile(impostors, 95))
            verdict = "good separation" if lo > hi else "OVERLAP -- add more views or try another embedder"
            print(f"  suggested --accept-thres {(lo + hi) / 2:.2f}   (same p5 {lo:.3f} vs impostor p95 {hi:.3f}: {verdict})")


if __name__ == "__main__":
    main()
