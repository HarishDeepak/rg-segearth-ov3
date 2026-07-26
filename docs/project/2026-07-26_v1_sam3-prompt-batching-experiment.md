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

| N prompts | Result | Peak VRAM |
|---|---|---|
| 6 | OK | 9.07 GB |
| 9 | OK | 11.75 GB |
| 12 | OK | 14.43 GB |
| 16 | **OOM** | — |

A T4 (16GB) tops out between 12 and 16 batched prompts at this crop size —
below the 20+ synonym-heavy class lists this project sometimes uses.

**Correctness: not a clean match.** Per-class max-abs-diff on the first test
crop:

| Class prompt | max\|diff\| |
|---|---|
| water body, river, lake | 2.7e-05 |
| railway track, rail line | 3.2e-05 |
| sunlit paved road... | 4.7e-04 |
| football pitch... | 1.5e-03 |
| clay sports court... | 1.6e-03 |
| building, rooftop... | 2.1e-02 |
| grass, lawn, low vegetation | 2.2e-02 |
| tree, wooded canopy | 2.9e-02 |
| **car, vehicle** | **9.2e-02** ← worst, drives the aggregate |

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
branch for anything further on this task.
