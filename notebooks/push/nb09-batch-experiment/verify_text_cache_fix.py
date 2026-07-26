"""Correctness check: pre-fix segmentor (re-encodes text per crop) vs
post-fix segmentor (text encoding cached once in __init__) must produce
numerically identical seg_logits for the same crops.

Loads the pre-fix segmentor class from git history (segmentor_old.py, a
frozen copy of segearthov3_segmentor.py from before the caching commit) and
the current one from the repo, runs both on the same crops of one tile, and
diffs _inference_single_view's output.

Run:  PYTHONPATH=. python notebooks/push/nb09-batch-experiment/verify_text_cache_fix.py
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).parent


def load_segmentor_class(py_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod.SegEarthOV3Segmentation


def crops_from_tile(img_path, crop, n_crops):
    arr = np.array(Image.open(img_path).convert("RGB"))
    H, W = arr.shape[:2]
    ys = np.linspace(0, max(H - crop, 0), n_crops, dtype=int)
    xs = np.linspace(0, max(W - crop, 0), n_crops, dtype=int)
    return [Image.fromarray(arr[y:y + crop, x:x + crop]) for y, x in zip(ys, xs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="dop20_32_476_5524_1_he")
    ap.add_argument("--crop", type=int, default=768)
    ap.add_argument("--crops", type=int, default=3)
    ap.add_argument("--classnames", default="configs/cls_hessen.txt")
    args = ap.parse_args()

    hits = sorted(Path("/kaggle/input").rglob(f"{args.tile}.jpg")) if Path("/kaggle/input").exists() else []
    if not hits:
        hits = sorted(Path(".").rglob(f"{args.tile}.jpg"))
    if not hits:
        print(f"ERROR: {args.tile}.jpg not found", flush=True)
        return 1
    img_path = hits[0]
    print(f"Tile: {img_path}", flush=True)

    OldSeg = load_segmentor_class(HERE / "segmentor_old.py", "segmentor_old")
    from segearthov3_segmentor import SegEarthOV3Segmentation as NewSeg

    print("Building OLD (pre-fix, re-encodes text per crop) segmentor...", flush=True)
    old = OldSeg(classname_path=args.classnames, slide_stride=0, slide_crop=0)
    print("Building NEW (post-fix, cached text encoding) segmentor...", flush=True)
    new = NewSeg(classname_path=args.classnames, slide_stride=0, slide_crop=0)

    crops = crops_from_tile(img_path, args.crop, args.crops)
    maxdiffs = []
    for i, crop_pil in enumerate(crops):
        l_old = old._inference_single_view(crop_pil)
        l_new = new._inference_single_view(crop_pil)
        d = (l_old - l_new).abs()
        maxdiffs.append(d.max().item())
        agree = (l_old.argmax(0) == l_new.argmax(0)).float().mean().item()
        print(f"crop {i}: max|diff| {d.max().item():.3e}  argmax agree {agree:.6f}", flush=True)

    print(f"\nOverall max|diff| across {len(crops)} crops: {max(maxdiffs):.3e}", flush=True)
    if max(maxdiffs) < 1e-4:
        print("PASS: outputs match (within bf16 rounding).", flush=True)
        return 0
    else:
        print("FAIL: outputs diverge beyond expected bf16 rounding.", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
