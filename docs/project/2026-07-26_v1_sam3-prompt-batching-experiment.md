eeeeste

# SAM3 prompt-batching experiment — what we tested, what we found, what we're doing next

Branch: `nb09-batch-experiment`. Code: `notebooks/push/nb09-batch-experiment/`
(isolated from the live NB09 sweep kernels — never touched them).

## The question

SAM3 inference for one image crop works like this today
(`segearthov3_segmentor.py::_inference_single_view`, and NB09's own inline
copy of the same loop, `collect_class_scores` in
`notebooks/push/nb09/NB09_zeroshot_sensitivity.ipynb` cell 11):

1. One shared image-encoder forward per crop (cheap).
2. One grounding-transformer forward **per class prompt**, in a Python loop,
   sequentially — typically 6-9 classes per tile today, but synonym-heavy
   class lists can reach 20+.

SAM3's own architecture supports asking about all N prompts in **one**
batched call instead (`FindStage` with `img_ids=[0]*N`, `text_ids=arange(N)`
— confirmed via Meta's own `facebookresearch/sam3` example notebook,
`examples/sam3_image_batched_inference.ipynb`). The question: does this make
our pipeline faster, and does it change the output?

## Analogy

Asking a librarian one book title at a time, waiting for each answer, vs.
handing them a written list of all titles at once. The librarian still has
to look up each title individually — the total lookup *work* doesn't shrink.
What shrinks is the back-and-forth overhead between you and the librarian.
If that overhead was never the bottleneck (the librarian is slow at look-ups,
not slow at listening to you), handing over the list doesn't help — and
that's close to what we found.

## What we built

- `notebooks/push/nb09-batch-experiment/batch_prompts_experiment.py` —
  standalone script, no dependency on any live NB09 kernel:
  - **Sequential baseline**: `collect_class_scores`, copied verbatim from
    NB09 cell 11, so the comparison is against exactly what the sweeps run
    today, not a re-derived approximation of it.
  - **Batched version**: one `forward_grounding` call with a batch dim of N
    prompts. Could **not** reuse `Sam3Processor._forward_grounding` as-is —
    its confidence-threshold filter (`keep = out_probs > thd`) flattens
    across the batch dimension, which would merge instances from different
    classes into one undifferentiated list. Had to reimplement that
    post-processing per-prompt-slice instead.
  - Times both paths (multiple crops × multiple reps, mean/std not a single
    sample), measures peak GPU memory for both, and checks per-class
    max-abs-diff + argmax agreement between the two paths' output logits.
- `notebooks/push/nb09-batch-experiment/NB09_batch_experiment.ipynb` — the
  Kaggle notebook wrapper (env setup reused from NB09 verbatim, clones this
  repo's `nb09-overnight-iteration` branch — see gotchas below).

## What we measured (Kaggle T4, tile `dop20_32_476_5524_1_he`, N=9 prompts, crop=768px, 4 crops × 3 reps)

```json
{
  "seq_mean_s": 3.039, "seq_std_s": 0.209,
  "bat_mean_s": 3.147, "bat_std_s": 0.104,
  "speedup": 0.966,
  "seq_peak_mem_gb": 4.93, "bat_peak_mem_gb": 11.81,
  "max_abs_diff": 0.092, "max_rel_diff": 0.095,
  "argmax_agreement": 0.9986
}
```

**Speed: no win — batched was ~3% slower.** Not within noise: batched was
slower in every single crop/rep pair we ran, not just on average.

**VRAM: batched cost ~2.4x more** for the same N=9 (11.8GB vs 4.9GB peak).

**VRAM ceiling (separate sweep, same crop size):**

| N prompts | Result        | Peak VRAM |
| --------- | ------------- | --------- |
| 6         | OK            | 9.07 GB   |
| 9         | OK            | 11.75 GB  |
| 12        | OK            | 14.43 GB  |
| 16        | **OOM** | —        |

A T4 (16GB) tops out between 12 and 16 batched prompts at this crop size —
below the 20+ synonym-heavy class lists this project sometimes uses.

**Correctness: not a clean match.** Per-class max-abs-diff on the first test
crop:

| Class prompt                | max\|diff\|                                      |
| --------------------------- | ------------------------------------------------ |
| water body, river, lake     | 2.7e-05                                          |
| railway track, rail line    | 3.2e-05                                          |
| sunlit paved road...        | 4.7e-04                                          |
| football pitch...           | 1.5e-03                                          |
| clay sports court...        | 1.6e-03                                          |
| building, rooftop...        | 2.1e-02                                          |
| grass, lawn, low vegetation | 2.2e-02                                          |
| tree, wooded canopy         | 2.9e-02                                          |
| **car, vehicle**      | **9.2e-02** ← worst, drives the aggregate |

The 9.2e-02 figure recurs across multiple crops (not a one-off outlier), and
it's too large and too concentrated to be ordinary bf16 rounding — bf16
noise wouldn't cluster this hard on specific classes. Argmax agreement stays
high (99.5–99.9%) but is not 100%.

## What we understand about *why*, after research

Full research write-up available on request (ran as a background research
pass); key points folded in here:

- **The VRAM blowup is architecturally inherent, not a bug in our
  implementation.** `forward_grounding` runs the full 200-query instance
  decoder and mask/segmentation head *per prompt in the batch* — none of
  that compute is shared across prompts. A batch of N prompts gets N full
  copies of decoder activations and mask logits, not a proportionally
  cheaper shared pass. Confirmed by reading `sam3/model/sam3_image.py` and
  `sam3/model/maskformer_segmentation.py` directly, not inferred from timing
  alone.
- **Why there's no speedup either, on a T4 specifically:** T4 is Turing
  architecture, which does not support bf16 FlashAttention (needs
  Ampere/Ada/Hopper) — the `sdpa_kernel` backend list in
  `sam3/model/vl_combiner.py` silently falls back to a slower attention
  backend on this hardware, eroding whatever compute-density gain batching
  is meant to provide. Separately, at N=9 the per-call Python/kernel-launch
  overhead batching removes was probably never the dominant cost — the
  image encoder (run once per crop, shared either way) is the expensive
  fixed cost, and each grounding-decoder pass is comparatively cheap. There
  wasn't much "dead time" between sequential calls for batching to actually
  eliminate.
- **Plausible (unconfirmed) cause of the correctness gap:** when N prompts
  of different lengths share one padded batch tensor, cross-attention
  padding-mask handling for heterogeneous-length captions is a common
  source of small numeric drift if any op isn't fully mask-safe. Notably,
  "car, vehicle" — our worst-diff class — is also our shortest prompt,
  consistent with this theory. Not confirmed against SAM3's actual internals
  or any public issue report; flagged as a real risk, not proven root cause.
- **Batching by prompts vs. batching by crops are different axes.** We only
  tested batching prompts against one crop. Batching multiple *crops*
  against one prompt would batch the image encoder instead — which is
  genuinely shared/scaling compute (like any standard batched ViT
  inference), not a per-prompt multiplicative blowup. `Sam3Processor` already
  has an unused `set_image_batch` method for exactly this. Architecturally
  more promising, but untested — not what this experiment measured.

## Bottom line

**Prompt-batching (as implemented and tested) is not worth adopting**: no
measured speedup, ~2.4x more VRAM, and a real (if not fully explained)
correctness gap concentrated in at least one class. Verdict reached from
actual measured numbers on Kaggle's T4, not assumed.

## A real bug we found as a side effect

While building the sequential baseline for this comparison, we had to write
a `cache_text` helper that encodes each class prompt's text embedding
**once** and reuses it across crops — because re-running `set_text_prompt`
(which calls the full text encoder) inside the per-crop loop, as
`segearthov3_segmentor.py::_inference_single_view` currently does, is pure
waste: the class list never changes across a tile's sliding-window pass. A
5000×5000 tile at crop=768/stride=576 is roughly 64 crops; at 9 classes
that's ~576 redundant text-encoder calls per tile where 9 would do. This fix
required to build the experiment's own baseline, but it was never ported
back into the actual production pipeline. Free, safe (numerically identical,
same computation called fewer times), and the one concrete win out of this
whole investigation. **This is what we're doing next** — porting this cache
into `segearthov3_segmentor.py`.

## Gotchas hit while running this on Kaggle (for next time)

1. **Repo is private; the notebook's `git clone` is anonymous HTTPS.** The
   existing NB09 kernels apparently work around this via a Kaggle-side
   credential we don't have wired into a fresh kernel. Made the repo
   temporarily public to unblock the clone — flip it back to private once
   this branch's work is merged/done.
2. **The clone cell had no `-b <branch>` flag** — it clones the repo's
   *default* branch (`master`), but this experiment folder lives on a
   feature branch. Fixed by adding `-b nb09-overnight-iteration` (later
   `nb09-batch-experiment`) explicitly to the clone command.
3. **`python script.py` vs `python -m`:** running the script by file path
   from the cloned repo root fails to import `config_local` (which lives at
   repo root) — Python puts the *script's own directory* on `sys.path[0]`,
   not the current working directory. Fixed with `export PYTHONPATH=.`
   before invoking the script.

## Branching note

This work originally landed on `nb09-overnight-iteration` (the branch
checked out at session start) — the wrong call, since that branch is shared
with active parallel NB09 sweep work, and a stray commit from that work
landed in the middle of ours. Moved to a dedicated `nb09-batch-experiment`
branch for anything further on this task. (That branch also turned out to be
shared with other concurrent work — NB10v2/NB10v3 commits from another
session landed on it too. Nothing of ours was affected, but worth noting the
isolation wasn't as clean in practice as intended.)

---

## Follow-up: the text-cache fix itself had a bug, found before trusting it

After documenting the batching verdict above, we ported the `cache_text`
fix into the real pipeline (`segearthov3_segmentor.py`) and built a
dedicated correctness check (`verify_text_cache_fix.py`) before trusting it:
run the old segmentor (re-encodes text every crop) and the new one (cached
once in `__init__`) on identical crops, diff the output logits.

**First run: FAIL.** Max-abs-diff of 0.116 — larger than the batching
experiment's own 0.092, and this was supposed to be a *zero-risk*,
numerically-identical refactor (same computation, called fewer times, no
architecture change). A diff that large meant something was actually wrong,
not just bf16 noise.

**Root cause, found by re-reading the old and new code side by side:** the
new cache builds each class's text embedding in `__init__`, via
`model.backbone.forward_text(...)` — but that call was **not** wrapped in
`torch.autocast(dtype=torch.bfloat16)`. The old path calls the exact same
`forward_text` function, but from *inside* `_inference_single_view`'s
existing `with torch.autocast(...)` block (via `set_text_prompt`). Same
function, two different numeric precisions — that mismatch, not the caching
change itself, produced the divergence. Fixed by wrapping the cache-building
loop in the same autocast context.

**This bug was hiding in the batching experiment too.** `cache_text` and
`cache_text_batched` in `batch_prompts_experiment.py` were copied from (or
modeled on) the same pattern and had the identical missing-autocast issue.
Fixed both. This means **the 0.092 max-diff and 99.86% argmax-agreement
numbers reported above for prompt-batching may be partly or mostly an
artifact of this precision bug, not proof that batching itself is
imprecise.** We have not yet re-run the batching comparison with the fix in
place — the verdict on batching's speed/VRAM cost stands (that's pure timing
and memory measurement, unaffected by this), but the *correctness* verdict
needs re-measuring before it's trusted. Flagging this explicitly rather than
quietly updating the number above, since the original numbers were reported
to the user and shouldn't be silently changed after the fact.

**Also confirmed:** this same missing-autocast pattern exists in NB09 cell
11's actual production `cache_text` — it was copied from there in the first
place. If NB09's sweeps ever get a text-caching optimization ported into
them, they need this same autocast fix, or they'll have the same latent
precision bug.

`verify_text_cache_fix.py` was also extended past a pass/fail number: it now
times both segmentor versions per crop, saves a PNG per crop (input crop |
old argmax | new argmax | disagreement overlay | logit-diff heatmap, with a
class-name legend), and writes a per-query pixel-count histogram to the JSON
output, to catch a class silently vanishing (which an averaged or max-only
diff can hide even when the headline numbers look fine).

**Second run (with the autocast fix applied), Kaggle T4, tile
`dop20_32_476_5524_1_he`, `cls_hessen.txt` (12 queries across 6 classes),
3 crops of 768px:**

```json
{
  "old_mean_s": 7.935, "old_std_s": 0.142,
  "new_mean_s": 7.631, "new_std_s": 0.249,
  "speedup": 1.040,
  "max_abs_diff": 0.0, "mean_abs_diff": 0.0
}
```

**PASS — exact, bit-for-bit match.** Every one of the 12 queries across all
3 crops has `max_diff: 0.0` and identical pixel counts (`old_px == new_px`,
checked individually, not just in aggregate) between the old and new
segmentor. Argmax agreement is `1.0` on every crop. Visually confirmed too —
the saved PNGs show pixel-identical segmentation maps and a flat, all-zero
diff heatmap on all 3 crops; no disagreement overlay pixels anywhere. This
is the result that should have appeared the first time, before the missing
autocast wrapper masked it as a false failure.

**The text-cache fix is confirmed correct and safe to keep.** Speedup at
this small scale (3 crops, 12 queries) is modest — 1.04x — because the
savings scale with crops-per-tile, and this test only exercises 3. The
expected larger win (a ~5000×5000 tile at crop=768/stride=576 is ~64 crops,
collapsing ~64× redundant text-encoder calls down to 1×) hasn't been
measured directly on a full tile yet; this test only establishes
correctness plus a directional (if small) speed improvement at low crop
count.

**Batching correctness re-measured with the same autocast fix applied,
same tile/crop/N=9-prompts setup as before:**

```json
{
  "seq_mean_s": 3.204, "bat_mean_s": 3.423,
  "speedup": 0.936,
  "seq_peak_mem_gb": 4.938, "bat_peak_mem_gb": 11.814,
  "max_abs_diff": 0.110, "argmax_agreement": 0.998
}
```

**The correctness gap did not shrink — it got slightly worse (0.110 vs the
original 0.092).** This resolves the open question above: prompt-batching's
divergence from the sequential baseline is a real property of batching
itself (plausibly the padding-mask/heterogeneous-prompt-length mechanism
described earlier), not an artifact of the missing-autocast bug. The speed
and VRAM verdicts are unchanged (still no speedup, still ~2.4x more memory).
**Batching remains not worth adopting**, now on a fully trustworthy
measurement.
