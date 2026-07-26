# NB10 v2 devlog — car-only vs. full-class confidence_threshold sweep

Live devlog, written step-by-step as work happens (not after the fact).
Companion to `notebooks/NB10v2_car_vs_fullclass_confthd.ipynb`. See also
v1's plan/result: `notebooks/NB10_car_confthd_demo.ipynb`, committed at
`4f495a5` on this branch, Kaggle run at
https://www.kaggle.com/code/harish77718/nb10-car-confthd-demo.

## Background: what v1 already showed

v1 swept `confidence_threshold=[0.05, 0.1, 0.2, 0.3, 0.5]` with `"car"` as
the sole prompt, on 2 tiles (`472_5525`, `473_5525`), full 5000×5000, no
crop, `prob_thd=0.1` and window params (`slide_crop=1024`,
`slide_stride=768`) fixed. Result: both tiles showed monotonically
decreasing car pixel count and car blob count as `confidence_threshold`
rose (tile 1: 594K→441K px, 1234→1019 blobs; tile 2: 1.23M→1.0M px,
1420→996 blobs). Confirmed via direct pixel counts on the saved `.npy`
label maps, not just visual inspection.

## v2 ask (from Harish/Dan's teammate chain)

1. Focus on tile `dop20_32_473_5525_1_he` only (drop tile 1 for this round).
2. Run **both** car-only AND a full multi-class prompt list, to compare
   "car in isolation" vs. "car competing with other classes for argmax."
3. Widen `confidence_threshold` range further in both directions.
4. Print legends (v1's car-only renders had none — single binary class).
5. Add clearer visualizations of what's happening at each threshold.
6. Explicit question: is `confidence_threshold` tunable per-class, or only
   globally for all classes in one run?
7. A specific observation: this tile has a train with **car-carrier
   wagons** (vehicles riding on a flatbed rail car) — a genuine
   car/train class ambiguity worth showing, not hiding.

## Step 0 — branch state

The working tree kept switching branches mid-session (parallel work
happening outside this conversation, with its own uncommitted edits and
stashes — confirmed via `git stash list` showing 3 unrelated stashes from
other work). Rather than force a switch to `nb09-overnight-iteration`
(where v1's NB10 lives, commit `4f495a5`) and risk disrupting that other
in-progress work, decided with Harish to build NB10v2 on whichever branch
is currently checked out (`nb09-batch-experiment`, HEAD `a7fe7ad` at time
of writing) — this notebook is a new, independent file, so it doesn't
need to be co-located with v1 to work correctly. Reconciling branches
(merge/rebase) can happen later if needed.

## Step 1 — resolving "is confidence_threshold per-class or global"

Re-confirmed from v1's own code (`make_processor(conf_thd) ->
Sam3Processor(model, confidence_threshold=conf_thd, ...)`): the value is
set once per `Sam3Processor` object and applies identically to every class
prompted through that processor in one run. There is no per-class
override anywhere in the pipeline (checked `cache_text`,
`collect_class_scores`, `run_sliding_window` — none of them take a
per-class threshold).

**Practical answer**: `confidence_threshold` can only be tuned per
*prompt-set*, not per individual class within a shared run. The only way
to get a "car-specific" tuning is to run car alone through its own
processor (exactly what v1 did, and what Run A in v2 repeats) — as soon as
car shares a run with other classes (Run B), one threshold value applies
to all 7 classes simultaneously. v2's Run A vs Run B numeric comparison
(car-alone curve vs. car-within-full-class curve) is designed to make the
practical cost of this limitation visible as numbers, not just stated as
a fact.

## Step 2 — precedent check on a "tracks" class (avoiding a repeated dead end)

Searched NB09's devlog + `IMAGE_CONFIGS` (via `git show
nb09-overnight-iteration:...` since that content lives in git history, not
necessarily the current working tree) for prior attempts at a standalone
`tracks`/`railway` class.

Found: on tile `475_5550` (Frankfurt Hbf) and tile `473_5524` (Darmstadt,
different tile ID than this one), a standalone tracks/railway/platform
class was attempted repeatedly across many wording variants (long
descriptive, short single-word, merged-concept) and **never once produced
a reliable detection** on either tile. Root cause identified in both
cases: tracks/railway kept getting confused with a **platform** structure
— same washed-out light-grey tonal range in strong sun. Both tiles
eventually dropped `tracks` entirely, leaving open trackbed as background,
keeping only a separate `train` class (the rolling stock itself), which
worked reliably at ~1% prevalence.

**Key distinction for this tile**: Harish confirmed `dop20_32_473_5525_1_he`
has **no platform** — just open trackbed and trains. Since the
platform-confusion mechanism that sank `tracks` twice before can't occur
here, the precedent doesn't fully transfer. Decision: include `tracks` in
v2's full-class list anyway, on tile-specific grounds, with wording that
deliberately avoids any platform-adjacent language ("railway tracks, rail
lines, ballast trackbed, gravel between rails" — no "platform," "raised
edge," "yellow tactile strip," etc. that sank it before). If it comes back
near 0% again despite no platform present, that's a genuine new finding
(a different failure mode), not a repeat of the old one — will be reported
as such either way.

## Step 3 — locating the car-carrier train wagon region

Harish's observation: "there are tracks and trains too but one of the
train has cars on it" — meaning one train on this tile is a flatbed
car-carrier wagon set, loaded with rows of small vehicles, visually
distinct from ordinary freight/tanker wagons elsewhere on the tracks.

First attempt at locating it: cropped `(1200, 1300)-(2000, 1850)` based on
an initial visual guess from the downsized full-tile preview — this turned
out to be a road/materials-yard crossing, NOT the wagon (wrong region).

Second attempt: cropped a wider band `(300, 2200)-(1900, 3600)` over the
main track corridor (lower-left quadrant, matching the area with the most
visible train wagons in the original full-tile view) to relocate it.

Third through sixth attempts: progressively narrowed crops chasing
distinctive-looking trains (repeating light-colored roof blocks) at
several locations along the track corridor, including one specifically
requested region ("bottom and right in front, towards down from
roundhouse"). Every one of these turned out on close zoom to be an
ordinary passenger/regional train — the repeating white blocks are
standard roof-mounted AC/vent units on German regional railcars, not
vehicles on a flatbed. None of the ~5 candidate crops matched the actual
car-carrier wagon.

**Decision: abandon manual visual search for this region.** It cost
several rounds of back-and-forth (including a coordinate-grid overlay
image, which the user correctly pushed back on — that was solving *my*
coordination problem, not a step the model or notebook actually needs) and
never converged. The car-carrier wagon exists somewhere in this tile per
Harish's direct observation, but its exact pixel location isn't needed as
a hardcoded input — the notebook can find any car/train ambiguity
algorithmically instead: compute, from the full-class run's raw per-class
logit maps (already produced by `run_sliding_window` before argmax),
where "car" and "train" logits are both simultaneously high, or where the
argmax label flips between the two as `confidence_threshold` sweeps. This
is strictly better than a manually-guessed bounding box anyway, since it's
derived from what the model actually computed rather than a human's
visual guess, and it can't miss the real ambiguous region or accidentally
include an irrelevant one.

**Lesson for future tiles**: when a specific visual feature needs to be
verified against model output, prefer computing it from the model's own
per-class scores rather than sinking time into manual pixel-hunting on a
5000×5000 tile — the model's disagreement between two class prompts *is*
the signal, and locating it doesn't require a human to find it by eye
first.

## Step 4 — final decisions before build (via AskUserQuestion)

- `confidence_threshold` range: `[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]`
  (8 values, widened both directions from v1's `[0.05, 0.1, 0.2, 0.3, 0.5]`).
- No `prob_thd` or stride/crop sweep this round — `confidence_threshold`
  stays the sole variable, matching v1's isolation principle.
- Verification approach: numeric plots are the primary deliverable, not a
  per-threshold rendered-image gallery — only one reference overlay image
  per run mode (car-only, full-class), at `confidence_threshold=0.1` for
  comparability with v1's baseline.
- Class list for the full-class run finalized: building, road, tracks,
  train, car, tree, grass (7 classes) — `tracks` included per the
  platform-absence reasoning above.

## Step 5 — branch handling

Working tree kept switching branches mid-session due to parallel work
happening outside this conversation (confirmed via `git stash list`
showing 3 unrelated stashes, plus real uncommitted edits to
`NB09_zeroshot_sensitivity.ipynb`, `batch_prompts_experiment.py`, and
`segearthov3_segmentor.py` that aren't part of this task). Decided with
Harish to build NB10v2 on whichever branch is currently checked out
(`nb09-batch-experiment`) rather than force a switch to
`nb09-overnight-iteration` and risk disrupting that other work — this
notebook is new/independent, so it doesn't need to live on the same
branch as v1 to function correctly.

## Step 6 — build

Built `notebooks/NB10v2_car_vs_fullclass_confthd.ipynb` (14 cells):
reused v1's env-setup (Miniconda + `segearth` conda env) and fork-clone
cells verbatim (cells 1-9, identical to `NB10_car_confthd_demo.ipynb`).

New inference cell (cell 11): loads `dop20_32_473_5525_1_he` once, then
for each of the 8 `confidence_threshold` values runs both Run A
(`words=["car"]`) and Run B (7-class full list) via the same
`run_sliding_window`/`finalize` functions v1 used unmodified — 16 full
sliding-window passes total. Saves per-run `.npy` label maps, and for
Run B specifically also saves the raw per-class logit maps (`float16`,
before argmax) needed for the car/train overlap analysis.

New rendering cell (cell 13, GPU-free): computes Run A's car
pixel/blob-count trend, Run B's per-class pixel-count trend (all 7
classes on one chart), the car-alone-vs-within-full-class comparison
chart, and the model-derived car/train overlap analysis (99th-percentile
logit threshold on both classes simultaneously, per `confidence_threshold`
value, with a padded crop rendered around the highest-overlap result if
any is found). Renders exactly 2 reference overlay images (one per run
mode) plus the overlap crop if applicable — not a 16-image gallery, per
the numeric-first decision above.

**Verification before considering this done**: both the inference cell's
embedded Python and the rendering cell were extracted from the assembled
notebook and run through `compile()` — both compile cleanly. (v1 caught a
real f-string bug this exact way before it would have failed on Kaggle;
worth repeating for every notebook built this way.) Not yet run on
Kaggle — that's the next step, pending confirmation to spend the GPU-hour
budget on a 16-rerun job (roughly 1.6x v1's per-tile cost).

