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

## Step 7 — v2 Kaggle run #1 (16 reruns, conf_thd up to 0.7)

Ran successfully: https://www.kaggle.com/code/harish77718/nb10v2-car-vs-fullclass-confthd
(kernel slug came out as `nb10v2-car-vs-fullclass-confthd`, not the
`nb10v2-car-fullclass-confthd` id originally specified in
`kernel-metadata.json` — Kaggle derives the slug from the title text, so
the id must match what Kaggle actually assigns or a second push 409s;
fixed by updating the `id` field after the first push).

Key results (full numbers/analysis relayed to Harish in-session):
- Run A (car-only) monotonically declines 1.23M→984K px across
  0.01→0.7, but **plateaus below 0.05** — 0.01/0.02/0.05 are nearly
  identical, meaning v1's original 0.05 floor wasn't cutting off much of
  the real curve.
- Run B (full-class): **`tracks` got real, substantial coverage this
  time** (~23% of the tile) — confirming the platform-absence hypothesis
  from Step 2/3. Its trend is inverted vs. every other class: increases
  as `confidence_threshold` rises, while building/road/car/tree/grass/
  train all decrease.
- Car pixel count is consistently ~1.5-2% *higher* in the full-class run
  than car-only at every threshold — no competition loss for car here.
- Model-derived car/train overlap analysis found real, growing overlap
  regions (353→1828 px) across the sweep, though the region turned out to
  be a broad multi-patch area (99th-percentile criterion), not a single
  pinpointed wagon — flagged as a caveat, not oversold as exact
  localization of the specific car-carrier wagon Harish spotted.

Follow-up analysis (no rerun needed, computed directly from the saved
`.npy` label maps): Dan/teammate asked why the large parking lot's car
count barely changes across the threshold sweep while the tile-wide total
clearly declines. Isolated the lot's bounding box (the single largest,
completely threshold-invariant blob, 71,723 px unchanged from
`conf_thd=0.01` to `0.7`) and diffed its logits directly: mean logit
change inside the lot between the threshold extremes was **0.00000** —
SAM3's grounding is saturated there regardless of threshold, while ~41%
of the tile's pixels *do* change. Cropping the lot out of the full label
maps confirmed it holds ~99.99% of its pixels across the whole sweep,
while "everything else" drops ~22%. Answered Dan's two hypotheses
directly: yes, it's one blob; and yes, the decline is driven by weaker,
more borderline detections elsewhere on the tile, not the lot. Proposed
(but did not build, per Harish's call — "skip both, this is enough
analysis for now") a follow-up `prob_thd` re-threshold of the already-
saved lot-region logits (free) and a bounded crop-only stride/crop test
(would need a new small experiment, real GPU cost) — both shelved as
diminishing returns for this demo.

## Step 8 — v2 Kaggle run #2 (18 reruns, conf_thd extended to 0.9)

Harish asked to extend the sweep ceiling to 0.9. Added one value —
`CONF_THD_SWEEP = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]` — not
a finer/more gradual step size throughout, just one more point at the top
(confirmed with Harish this reading was correct: "ya thats ok sounds
good"). Re-pushed; hit the same slug-vs-id 409 issue as before (metadata
file still had the stale `nb10v2-car-fullclass-confthd` id from before
the first fix — the fix from run #1 hadn't been committed/synced into the
push-folder copy). Fixed again, pushed successfully as kernel version 2,
ran to completion.

## Step 9 — course correction: car-only only, wrong visualization check

Two things Harish flagged as wrong after reviewing run #2's results:

1. **Full-class Run B should not have been included.** Re-reading the
   session: Harish's original v2 request *did* ask for both car-only and
   full-class ("i want both only car as prompt and also the full class
   for all test"), and this was confirmed again mid-session. But the
   follow-up correction reverses that — going forward, NB10 should be
   **car-only only**, matching v1's original approach. Not a
   contradiction so much as a genuine change of direction after seeing
   the full-class results in practice.

2. **The dummyirl visualization check was done on the wrong file.**
   Earlier (Step 6/end of v1 era) I checked `dummy-irl/SegEarth-OV-3`'s
   most recent commit (`55357cf`, "update sam3 visualization") and found
   it only touched `sam3/visualization_utils.py` — masklet/object-
   tracking rendering (bounding boxes, per-object IDs via `cv2.putText`),
   unrelated to our RGB|overlay-with-legend matplotlib style. Correctly
   concluded there was nothing to pull for our output template *from that
   file*. But this was incomplete: Harish redirected me to check NB09's
   **own most recent Kaggle run** instead
   (`harish77718/nb09-zero-shot-sensitivity`, run 2026-07-26 20:39),
   which revealed NB09 had *already* been re-synced (by other work
   happening in parallel outside this session) to match a *different*
   dummyirl file — `segment.py`, not `visualization_utils.py` — that I
   had never checked. NB09's `render_result` function's own docstring
   documents the exact re-sync and diffs from the old approximation:
   `figsize=(10,7)` `dpi=300` (not `(20,12)`/`200`), 4-column legend grid
   via `ncol=min(4,...)` (not single-row `ncol=min(7,...)`), `alpha=0.5`
   (not `0.6`), 2-line meta text with `\n` (not one crammed line), plain
   `savefig` with no `bbox_inches='tight'`.

   **Lesson**: "check dummyirl for updates" isn't fully answered by
   checking the upstream repo's commit log alone when a *sibling
   notebook in this same repo* may have already done that sync — always
   check for evidence of an already-completed sync (a docstring, a
   comment, a recent file diff) in the codebase itself before concluding
   "nothing relevant found" from the upstream repo directly.

## Step 10 — build NB10v3

Built `notebooks/NB10v3_car_confthd.ipynb` (14 cells): same env-setup/
clone cells as v1/v2. New inference cell: car-only only
(`words=["car"]`), 9 `confidence_threshold` values
(`0.01`→`0.9`), `prob_thd=0.1`/`slide_crop=1024`/`slide_stride=768`
unchanged — 9 reruns total (down from v2's 18).

New rendering cell: `to_rgb`/`render_result` copied **verbatim** (diffed
programmatically against NB09's cell 13 — identical except for the
docstring, confirmed via `difflib`) from
`notebooks/push/nb09/NB09_zeroshot_sensitivity.ipynb`, so the output
image format now genuinely matches the team's current template rather
than a reimplemented approximation. Kept the car pixel/blob-count trend
plot from v2 (Harish's own numeric-analysis addition, unrelated to the
template mismatch, still valuable). Dropped the car-vs-fullclass
comparison chart and the car/train overlap analysis, since both needed
Run B's full-class logits which no longer exist in v3.

Both cells verified via `compile()` on the assembled notebook before
pushing (same discipline as v1/v2). Committed as `5e51b29` on
`nb09-batch-experiment`, confirmed synced to `origin` via `git fetch`
before considering the push done — this branch has multiple concurrent
writers this session (other work landed commits `4615723` and `609bdb6`
in between my own commits without any action on my part), so verifying
against `origin` rather than trusting local state alone matters here.

Next: push to Kaggle and run.

