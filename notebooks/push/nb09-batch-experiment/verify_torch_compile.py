"""torch.compile experiment: does build_sam3_image_model(compile=True) speed up
inference in this pipeline, and does it change the output?

Per research (docs/project/2026-07-26_v1_sam3-prompt-batching-experiment.md's sibling
investigation): compile=True today only wires compile_mode into the ViT backbone and
PixelDecoder -- NOT the transformer decoder that runs once per class/prompt per crop.
So this measures "does compiling the image encoder help", not "does compiling the
whole hot loop help". Cheap to test regardless (one kwarg flip).

Timing methodology: crop 1 on the compiled model triggers Inductor compilation and is
reported separately as one-time cost, never averaged into steady-state numbers. Crops
2+ are steady-state.

Correctness: same rigor as verify_text_cache_fix.py -- per-query pixel histogram (not
just an aggregate diff), since this project's known failure mode is argmax starvation
between classes, which a small logit diff can still flip.

Run:  PYTHONPATH=. python notebooks/push/nb09-batch-experiment/verify_torch_compile.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import torch
from PIL import Image

from config_local import SAM3_CHECKPOINT
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

HERE = Path(__file__).parent
DEVICE = "cuda"


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


def get_cls_idx(path):
    with open(path, "r") as f:
        name_sets = f.readlines()
    class_names, class_indices = [], []
    for idx, line in enumerate(name_sets):
        names_i = [n.strip() for n in line.split(",")]
        class_names += names_i
        class_indices += [idx for _ in names_i]
    class_names = [n.replace("\n", "") for n in class_names]
    return class_names


def build_model_and_cache(compile_flag, classnames_path):
    print(f"Building SAM3 (compile={compile_flag})...", flush=True)
    model = build_sam3_image_model(
        bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path=SAM3_CHECKPOINT, device=DEVICE, compile=compile_flag,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    processor = Sam3Processor(model, confidence_threshold=0.1, device=DEVICE)
    query_words = get_cls_idx(classnames_path)

    text_cache = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for word in query_words:
            te = model.backbone.forward_text([word], device=DEVICE)
            text_cache.append({k: v.cpu() for k, v in te.items()})
    return model, processor, query_words, text_cache


def inference_single_view(model, processor, text_cache, image):
    w, h = image.size
    import torch.nn.functional as F
    seg_logits = torch.zeros((len(text_cache), h, w), device=DEVICE)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        for qi, te_cpu in enumerate(text_cache):
            processor.reset_all_prompts(state)
            for k, v in te_cpu.items():
                state["backbone_out"][k] = v.to(DEVICE)
            state["geometric_prompt"] = model._get_dummy_prompt()
            state = processor._forward_grounding(state)
            if state["masks_logits"].shape[0] > 0:
                for i in range(state["masks_logits"].shape[0]):
                    il = state["masks_logits"][i].squeeze()
                    if il.shape != (h, w):
                        il = F.interpolate(il.view(1, 1, *il.shape), size=(h, w),
                                           mode="bilinear", align_corners=False).squeeze()
                    seg_logits[qi] = torch.max(seg_logits[qi], il * state["object_score"][i])
            sem = state["semantic_mask_logits"]
            if sem.shape != (h, w):
                sem = F.interpolate(sem, size=(h, w), mode="bilinear", align_corners=False).squeeze()
            seg_logits[qi] = torch.max(seg_logits[qi], sem)
            seg_logits[qi] = seg_logits[qi] * state["presence_score"]
    return seg_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="dop20_32_476_5524_1_he")
    ap.add_argument("--crop", type=int, default=768)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timed-crops", type=int, default=10)
    ap.add_argument("--classnames", default="configs/cls_hessen.txt")
    ap.add_argument("--out-dir", default="/kaggle/working/verify_torch_compile")
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

    n_crops = args.warmup + args.timed_crops
    crops = crops_from_tile(img_path, args.crop, n_crops)

    model_eager, proc_eager, words, cache_eager = build_model_and_cache(False, args.classnames)
    model_comp, proc_comp, _, cache_comp = build_model_and_cache(True, args.classnames)

    eager_times, comp_times = [], []
    compile_time = None
    maxdiffs, meandiffs = [], []
    per_query_all = []

    for i, crop_pil in enumerate(crops):
        l_eager, t_eager = timed(lambda: inference_single_view(model_eager, proc_eager, cache_eager, crop_pil))
        l_comp, t_comp = timed(lambda: inference_single_view(model_comp, proc_comp, cache_comp, crop_pil))

        if i < args.warmup:
            print(f"crop {i} (WARMUP): eager {t_eager:.3f}s  compiled {t_comp:.3f}s "
                  f"(compiled includes one-time compile cost)", flush=True)
            if i == 0:
                compile_time = t_comp
            continue

        eager_times.append(t_eager)
        comp_times.append(t_comp)

        d = (l_eager - l_comp).abs()
        maxdiffs.append(d.max().item())
        meandiffs.append(d.mean().item())
        argmax_eager, argmax_comp = l_eager.argmax(0), l_comp.argmax(0)
        agree = (argmax_eager == argmax_comp).float().mean().item()
        print(f"crop {i}: eager {t_eager:.3f}s  compiled {t_comp:.3f}s  "
              f"max|diff| {d.max().item():.3e}  argmax agree {agree:.6f}", flush=True)

        old_np, new_np = argmax_eager.cpu().numpy(), argmax_comp.cpu().numpy()
        per_query = []
        for qi, word in enumerate(words):
            old_px = int((old_np == qi).sum())
            new_px = int((new_np == qi).sum())
            per_query.append(dict(query=word, eager_px=old_px, compiled_px=new_px,
                                  max_diff=float(d[qi].max().item())))
        per_query_all.append(dict(crop=i, argmax_agreement=agree, per_query=per_query))

    eager_t, comp_t = np.array(eager_times), np.array(comp_times)
    speedup = float(eager_t.mean() / comp_t.mean())
    breakeven = (compile_time - (eager_t.mean())) / max(eager_t.mean() - comp_t.mean(), 1e-9) \
        if eager_t.mean() > comp_t.mean() else None

    res = dict(
        tile=str(img_path), crop=args.crop, n_queries=len(words),
        warmup_crops=args.warmup, timed_crops=args.timed_crops,
        compile_one_time_cost_s=compile_time,
        eager_mean_s=float(eager_t.mean()), eager_std_s=float(eager_t.std()),
        compiled_mean_s=float(comp_t.mean()), compiled_std_s=float(comp_t.std()),
        speedup=speedup,
        breakeven_crops=breakeven,
        max_abs_diff=float(max(maxdiffs)), mean_abs_diff=float(np.mean(meandiffs)),
        per_crop=per_query_all,
    )
    print("\n" + json.dumps({k: v for k, v in res.items() if k != "per_crop"}, indent=2), flush=True)
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    print(f"\nWrote {out_dir / 'results.json'}", flush=True)

    print(f"\nOne-time compile cost: {compile_time:.3f}s", flush=True)
    print(f"Steady-state: eager {eager_t.mean():.3f}s  compiled {comp_t.mean():.3f}s  "
          f"speedup {speedup:.3f}x", flush=True)
    if breakeven is not None:
        print(f"Break-even: ~{breakeven:.1f} crops before compile overhead pays for itself", flush=True)
    else:
        print("Compiled is not faster on average -- no break-even point (never pays off).", flush=True)

    print(f"Max|diff| across timed crops: {max(maxdiffs):.3e}  mean|diff|: {np.mean(meandiffs):.3e}", flush=True)
    if max(maxdiffs) < 1e-4:
        print("CORRECTNESS: PASS (within bf16-autocast rounding tolerance).", flush=True)
        return 0
    else:
        print("CORRECTNESS: FAIL (diverges beyond expected bf16 rounding).", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
