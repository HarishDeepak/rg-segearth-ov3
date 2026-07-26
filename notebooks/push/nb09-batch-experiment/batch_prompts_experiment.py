"""SAM3 prompt-batching experiment: N sequential grounding calls vs 1 batched call.

Answers: is batching correct, is it faster, is it worth adopting?
Baseline = NB09 cell 11's `collect_class_scores` (per-class loop), copied verbatim.
Batched  = same math, one `forward_grounding` with a batch dim of N prompts.

Run:  python batch_prompts_experiment.py [--tile <stem>] [--crop 768] [--reps 3]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config_local import SAM3_CHECKPOINT
from sam3 import build_sam3_image_model
from sam3.model.data_misc import FindStage
from sam3.model.sam3_image_processor import Sam3Processor

DEVICE = "cuda"

# Tile 3's class list, verbatim from NB09 cell 11 (9 prompts — a realistic N).
WORDS = [
    "water body, river, lake",
    "building, rooftop, residential roof",
    "sunlit paved road with lane markings and curbs, street with passing cars, not shadow, not dark shaded area",
    "football pitch with painted field markings encircled by an oval running track, athletics track and field",
    "clay sports court, tennis court, dirt sports ground",
    "tree, wooded canopy",
    "car, vehicle",
    "grass, lawn, low vegetation",
    "railway track, rail line",
]


# ─────────────────────── baseline: NB09's per-class loop ───────────────────────
# Copied from notebooks/push/nb09 cell 11 unmodified, so "sequential" is exactly
# what the sweeps run today, not a re-derivation of it.
def cache_text(model, words):
    cache = []
    with torch.no_grad():
        for word in words:
            te = model.backbone.forward_text([word], device=DEVICE)
            cache.append({k: v.cpu() for k, v in te.items()})
    return cache


def collect_class_scores(model, processor, state, h, w, te_cache, n_classes, device):
    logits = torch.zeros((n_classes, h, w), device=device)
    for cls_idx, te_cpu in enumerate(te_cache):
        processor.reset_all_prompts(state)
        for k, v in te_cpu.items():
            state["backbone_out"][k] = v.to(device)
        state["geometric_prompt"] = model._get_dummy_prompt()
        processor._forward_grounding(state)
        scores = torch.zeros((h, w), device=device)
        if state.get("masks_logits") is not None and state["masks_logits"].shape[0] > 0:
            for i in range(state["masks_logits"].shape[0]):
                il = state["masks_logits"][i].squeeze()
                if il.shape != (h, w):
                    il = F.interpolate(il.view(1, 1, *il.shape), size=(h, w),
                                       mode="bilinear", align_corners=False).squeeze()
                scores = torch.max(scores, il * state["object_score"][i])
        sem = state["semantic_mask_logits"].squeeze()
        if sem.shape != (h, w):
            sem = F.interpolate(sem.view(1, 1, *sem.shape), size=(h, w),
                                mode="bilinear", align_corners=False).squeeze()
        scores = torch.max(scores, sem) * state["presence_score"]
        logits[cls_idx] = torch.max(logits[cls_idx], scores)
    return logits


# ─────────────────────── batched: one grounding call for N prompts ───────────────────────
def cache_text_batched(model, words):
    """One forward_text over all N prompts -> language_* with batch dim N."""
    with torch.no_grad():
        te = model.backbone.forward_text(words, device=DEVICE)
    return {k: v.cpu() for k, v in te.items()}


def collect_class_scores_batched(model, state, h, w, te_batched, n_classes, device,
                                 conf_thd):
    """Same math as collect_class_scores, but N prompts in one forward_grounding.

    img_ids = [0]*N  -> _get_img_feats gathers the same crop features N times.
    text_ids = arange(N) -> one text slice per prompt.
    Everything downstream carries a leading batch dim of N; we slice it back out
    per class instead of letting the confidence filter flatten across prompts
    (which is what Sam3Processor._forward_grounding does, and why it can't be
    reused verbatim here).
    """
    backbone_out = dict(state["backbone_out"])
    for k, v in te_batched.items():
        backbone_out[k] = v.to(device)

    find_stage = FindStage(
        img_ids=torch.zeros(n_classes, dtype=torch.long, device=device),
        text_ids=torch.arange(n_classes, dtype=torch.long, device=device),
        input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
        input_points=None, input_points_mask=None,
    )

    out = model.forward_grounding(
        backbone_out=backbone_out,
        find_input=find_stage,
        geometric_prompt=model._get_dummy_prompt(num_prompts=n_classes),
        find_target=None,
    )

    # [N, Q, 1] -> [N, Q];  presence [N, 1] -> [N, 1, 1] to match processor's unsqueeze(1)
    presence = out["presence_logit_dec"].sigmoid().unsqueeze(1)          # [N, 1, 1]
    probs = (out["pred_logits"].sigmoid() * presence).squeeze(-1)        # [N, Q]
    masks = out["pred_masks"]                                           # [N, Q, h', w']
    sem_all = F.interpolate(out["semantic_seg"].float(), (h, w),
                            mode="bilinear", align_corners=False).sigmoid()  # [N, 1, h, w]

    logits = torch.zeros((n_classes, h, w), device=device)
    for cls_idx in range(n_classes):
        keep = probs[cls_idx] > conf_thd
        scores = torch.zeros((h, w), device=device)
        if keep.any():
            kept_masks = F.interpolate(masks[cls_idx][keep].unsqueeze(1).float(), (h, w),
                                       mode="bilinear", align_corners=False).sigmoid()
            # per-instance max of (mask_logit * object_score), same as the loop
            scores = (kept_masks.squeeze(1) * probs[cls_idx][keep][:, None, None]).amax(0)
        scores = torch.max(scores, sem_all[cls_idx, 0]) * presence[cls_idx].squeeze()
        logits[cls_idx] = scores
    return logits


# ─────────────────────── driver ───────────────────────
def crops_from_tile(img_path, crop, n_crops):
    """Deterministic centre-ish crops, so both paths see identical pixels."""
    arr = np.array(Image.open(img_path).convert("RGB"))
    H, W = arr.shape[:2]
    ys = np.linspace(0, max(H - crop, 0), n_crops, dtype=int)
    xs = np.linspace(0, max(W - crop, 0), n_crops, dtype=int)
    return [Image.fromarray(arr[y:y + crop, x:x + crop]) for y, x in zip(ys, xs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="dop20_32_476_5524_1_he")
    ap.add_argument("--crop", type=int, default=768)
    ap.add_argument("--crops", type=int, default=4, help="distinct crops to test")
    ap.add_argument("--reps", type=int, default=3, help="timed repetitions per crop")
    ap.add_argument("--conf-thd", type=float, default=0.1)
    ap.add_argument("--out", default="/kaggle/working/batch_experiment_results.json")
    args = ap.parse_args()

    hits = sorted(Path("/kaggle/input").rglob(f"{args.tile}.jpg")) if Path("/kaggle/input").exists() else []
    if not hits:
        hits = sorted(Path(".").rglob(f"{args.tile}.jpg"))
    if not hits:
        print(f"ERROR: {args.tile}.jpg not found", flush=True)
        return 1
    img_path = hits[0]
    print(f"Tile: {img_path}", flush=True)

    print("Loading SAM3...", flush=True)
    model = build_sam3_image_model(
        bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path=SAM3_CHECKPOINT, device=DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    processor = Sam3Processor(model, confidence_threshold=args.conf_thd, device=DEVICE)
    N = len(WORDS)
    te_seq = cache_text(model, WORDS)
    te_bat = cache_text_batched(model, WORDS)
    crops = crops_from_tile(img_path, args.crop, args.crops)
    print(f"N={N} prompts, {len(crops)} crops of {args.crop}px, {args.reps} reps\n", flush=True)

    def timed(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        return r, time.perf_counter() - t0

    seq_times, bat_times, maxdiffs, reldiffs = [], [], [], []
    seq_peak = bat_peak = 0

    for ci, crop_pil in enumerate(crops):
        w, h = crop_pil.size
        for rep in range(args.reps):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = processor.set_image(crop_pil)   # shared image encoder pass
                torch.cuda.reset_peak_memory_stats()
                l_seq, t_seq = timed(lambda: collect_class_scores(
                    model, processor, state, h, w, te_seq, N, DEVICE).float())
                seq_peak = max(seq_peak, torch.cuda.max_memory_allocated())

                state = processor.set_image(crop_pil)   # fresh state, same pixels
                torch.cuda.reset_peak_memory_stats()
                l_bat, t_bat = timed(lambda: collect_class_scores_batched(
                    model, state, h, w, te_bat, N, DEVICE, args.conf_thd).float())
                bat_peak = max(bat_peak, torch.cuda.max_memory_allocated())

            seq_times.append(t_seq)
            bat_times.append(t_bat)
            d = (l_seq - l_bat).abs()
            maxdiffs.append(d.max().item())
            denom = l_seq.abs().max().clamp(min=1e-6)
            reldiffs.append((d.max() / denom).item())
            print(f"crop {ci} rep {rep}: seq {t_seq:.3f}s  batched {t_bat:.3f}s  "
                  f"max|diff| {maxdiffs[-1]:.3e}  argmax agree "
                  f"{(l_seq.argmax(0) == l_bat.argmax(0)).float().mean().item():.6f}", flush=True)

        # per-class divergence on the last rep, to localise any mismatch
        if ci == 0:
            for k in range(N):
                print(f"    class {k} ({WORDS[k][:28]!r}): max|diff| "
                      f"{(l_seq[k] - l_bat[k]).abs().max().item():.3e}", flush=True)

    seq_t, bat_t = np.array(seq_times), np.array(bat_times)
    res = dict(
        tile=str(img_path), crop=args.crop, n_prompts=N, n_crops=len(crops), reps=args.reps,
        gpu=torch.cuda.get_device_name(0),
        seq_mean_s=float(seq_t.mean()), seq_std_s=float(seq_t.std()),
        bat_mean_s=float(bat_t.mean()), bat_std_s=float(bat_t.std()),
        speedup=float(seq_t.mean() / bat_t.mean()),
        seq_peak_mem_gb=seq_peak / 1e9, bat_peak_mem_gb=bat_peak / 1e9,
        max_abs_diff=float(max(maxdiffs)), max_rel_diff=float(max(reldiffs)),
        argmax_agreement=float((l_seq.argmax(0) == l_bat.argmax(0)).float().mean().item()),
    )
    print("\n" + json.dumps(res, indent=2), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nWrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
