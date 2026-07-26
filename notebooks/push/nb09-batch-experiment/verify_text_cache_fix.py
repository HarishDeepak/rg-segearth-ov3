"""Correctness + speed check: pre-fix segmentor (re-encodes text per crop) vs
post-fix segmentor (text encoding cached once in __init__).

Must produce numerically identical seg_logits for the same crops (correctness),
and the fix should show up as a real wall-clock win (fewer redundant text-encoder
calls) once averaged over multiple crops.

Loads the pre-fix segmentor class from a frozen copy (segmentor_old.py, taken
from before the caching commit) and the current one from the repo, runs both
on the same crops of one tile, diffs _inference_single_view's output, times
both, and saves PNG comparisons (old argmax | new argmax | abs-diff heatmap)
so results are visible, not just numeric.

Run:  PYTHONPATH=. python notebooks/push/nb09-batch-experiment/verify_text_cache_fix.py
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# Must be set before matplotlib is imported anywhere (including transitively
# via mmseg/mmcv) -- Kaggle's papermill/Jupyter parent process sets MPLBACKEND
# to a notebook-only backend that a plain script process can't use.
os.environ["MPLBACKEND"] = "Agg"

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


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return r, time.perf_counter() - t0


def save_comparison_png(out_path, crop_pil, argmax_old, argmax_new, diff, class_names):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n_classes = len(class_names)
    cmap = plt.get_cmap("tab20", max(n_classes, 1))

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    axes[0].imshow(crop_pil)
    axes[0].set_title("input crop")

    axes[1].imshow(argmax_old, cmap=cmap, vmin=0, vmax=max(n_classes - 1, 1))
    axes[1].set_title("OLD (re-encode/crop)\nargmax")
    axes[2].imshow(argmax_new, cmap=cmap, vmin=0, vmax=max(n_classes - 1, 1))
    axes[2].set_title("NEW (cached text)\nargmax")

    disagree = (argmax_old != argmax_new)
    axes[3].imshow(crop_pil)
    axes[3].imshow(np.ma.masked_where(~disagree, disagree), cmap="autumn", alpha=0.6)
    axes[3].set_title(f"argmax disagreement\n({disagree.mean()*100:.2f}% of px)")

    im = axes[4].imshow(diff, cmap="hot")
    axes[4].set_title(f"|logit diff| heatmap\n(max={diff.max():.3e})")
    plt.colorbar(im, ax=axes[4], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    legend_handles = [Patch(color=cmap(i), label=name) for i, name in enumerate(class_names)]
    fig.legend(handles=legend_handles, loc="lower center", ncol=min(n_classes, 8),
               bbox_to_anchor=(0.5, -0.05), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="dop20_32_476_5524_1_he")
    ap.add_argument("--crop", type=int, default=768)
    ap.add_argument("--crops", type=int, default=3)
    ap.add_argument("--classnames", default="configs/cls_hessen.txt")
    ap.add_argument("--out-dir", default="/kaggle/working/verify_text_cache")
    args = ap.parse_args()

    hits = sorted(Path("/kaggle/input").rglob(f"{args.tile}.jpg")) if Path("/kaggle/input").exists() else []
    if not hits:
        hits = sorted(Path(".").rglob(f"{args.tile}.jpg"))
    if not hits:
        print(f"ERROR: {args.tile}.jpg not found", flush=True)
        return 1
    img_path = hits[0]
    print(f"Tile: {img_path}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    OldSeg = load_segmentor_class(HERE / "segmentor_old.py", "segmentor_old")
    from segearthov3_segmentor import SegEarthOV3Segmentation as NewSeg

    print("Building OLD (pre-fix, re-encodes text per crop) segmentor...", flush=True)
    old = OldSeg(classname_path=args.classnames, slide_stride=0, slide_crop=0)
    print("Building NEW (post-fix, cached text encoding) segmentor...", flush=True)
    new = NewSeg(classname_path=args.classnames, slide_stride=0, slide_crop=0)
    query_words = old.query_words  # argmax is per-query, not per-class (num_queries >= num_cls)

    crops = crops_from_tile(img_path, args.crop, args.crops)
    maxdiffs, meandiffs, old_times, new_times, per_crop = [], [], [], [], []

    for i, crop_pil in enumerate(crops):
        l_old, t_old = timed(lambda: old._inference_single_view(crop_pil))
        l_new, t_new = timed(lambda: new._inference_single_view(crop_pil))
        old_times.append(t_old)
        new_times.append(t_new)

        d = (l_old - l_new).abs()
        maxdiffs.append(d.max().item())
        meandiffs.append(d.mean().item())
        argmax_old, argmax_new = l_old.argmax(0), l_new.argmax(0)
        agree = (argmax_old == argmax_new).float().mean().item()
        print(f"crop {i}: old {t_old:.3f}s  new {t_new:.3f}s  "
              f"max|diff| {d.max().item():.3e}  mean|diff| {d.mean().item():.3e}  "
              f"argmax agree {agree:.6f}", flush=True)

        # per-query pixel histogram: catches "a class silently vanished" bugs
        # that an averaged/max diff can hide (e.g. one query flips to always-losing
        # the argmax even if its raw logit diff is individually small).
        per_query = []
        old_np, new_np = argmax_old.cpu().numpy(), argmax_new.cpu().numpy()
        for qi, word in enumerate(query_words):
            old_px = int((old_np == qi).sum())
            new_px = int((new_np == qi).sum())
            per_query.append(dict(query=word, old_px=old_px, new_px=new_px,
                                  max_diff=float(d[qi].max().item()),
                                  mean_diff=float(d[qi].mean().item())))
            print(f"    q{qi} {word[:35]!r}: old_px={old_px} new_px={new_px} "
                  f"max|diff|={d[qi].max().item():.3e}", flush=True)
        per_crop.append(dict(crop=i, argmax_agreement=agree, per_query=per_query))

        png_path = out_dir / f"crop{i}_comparison.png"
        save_comparison_png(
            png_path, crop_pil, old_np, new_np,
            d.amax(0).cpu().numpy(), query_words,
        )
        print(f"    saved {png_path}", flush=True)

    old_t, new_t = np.array(old_times), np.array(new_times)
    res = dict(
        tile=str(img_path), crop=args.crop, n_crops=len(crops), n_queries=len(query_words),
        old_mean_s=float(old_t.mean()), old_std_s=float(old_t.std()),
        new_mean_s=float(new_t.mean()), new_std_s=float(new_t.std()),
        speedup=float(old_t.mean() / new_t.mean()),
        max_abs_diff=float(max(maxdiffs)), mean_abs_diff=float(np.mean(meandiffs)),
        per_crop=per_crop,
    )
    print("\n" + json.dumps(res, indent=2), flush=True)
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    print(f"\nWrote {out_dir / 'results.json'}", flush=True)

    print(f"\nOverall max|diff| across {len(crops)} crops: {max(maxdiffs):.3e}"
          f"  mean|diff|: {np.mean(meandiffs):.3e}", flush=True)
    print(f"Mean per-crop time: old {old_t.mean():.3f}s  new {new_t.mean():.3f}s  "
          f"speedup {old_t.mean() / new_t.mean():.2f}x", flush=True)
    if max(maxdiffs) < 1e-4:
        print("PASS: outputs match (within bf16 rounding).", flush=True)
        return 0
    else:
        print("FAIL: outputs diverge beyond expected bf16 rounding.", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
