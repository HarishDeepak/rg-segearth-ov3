# NB09 overnight iteration log — 2026-07-18

Context: NB09 (zero-shot sensitivity demo) was pushed twice earlier with the
full 3-tile / small-crop sweep and hung past 3h with no completion signal
(see comment in `notebooks/NB09_zeroshot_sensitivity.ipynb` cell 11). Both
runs were cancelled. Switched to `TARGET_TILE` single-tile mode
(baseline `slide_crop=1024`, `slide_stride=768`) to keep iterations short
enough to actually converge in one night.

Plan: run one tile at a time, pull results, visually judge quality (clean
class boundaries, low speckle/noise, sensible confusions vs. what the scene
actually contains), edit prompts/stride/thresholds to fix the weakest part,
repush, repeat — 2 to 4 iterations, unattended overnight. No teammate
reference images exist for Darmstadt/Frankfurt tiles specifically, so
judgment is by eye against the source image, calibrated against the
teammate's general reference style below.

**Quality bar (`teammate/SegEarth-OV-3/resources/vis.png`):** RGB | prediction
side-by-side. Predictions read like a simplified map, not noisy pixel scatter —
building blocks are solid polygons following real footprints, roads are
continuous thin lines, fields/cropland are large clean blobs, near-zero
salt-and-pepper speckle. This is the bar each NB09 iteration is judged against:
does the tile look like a clean, confident segmentation map, or a noisy mess?

**Note on previous runs:** the user pointed out that scriptVersionId 332631621
(kernel version 2) was the only earlier attempt that ran to completion without
hanging, using "a different approach" — this is very likely the run whose
output is already pulled into `results/nb09/` (3 tiles, baseline preds +
partB/partC PNGs, ~6939s total). Treat that pulled data as the actual v2
output, not a separate unknown run.

## Reference set: teammate's actual Darmstadt/Frankfurt outputs

User shared `D:\Downloads\drive-download-20260718T135518Z-1-001\` — teammate's
real segmentation outputs on the *same* DOP20 tiles this project uses
(`468_5543`, `469_5521`, `472_5525` incl. a `windowsize_exp` sweep, `474_5546`,
`479_5550`, `525_5604`), each with an RGB|prediction panel, a class-color
legend, and a params footer (`img_size`, `prob_thd`, `conf_thd`,
`slide_stride`, `slide_crop`). This is the real quality bar — far more
directly useful than the generic non-German `teammate/SegEarth-OV-3/resources/vis.png`.

**Findings, consolidated across 6 tiles:**

1. **Short, scene-specific class names beat verbose multi-synonym lists.**
   Every clean-looking reference uses one or two words per class — `"grass"`,
   `"trees"`, `"solar panel"`, `"stadium"`, `"football court"`,
   `"automotive vehicles"`, `"dense tree canopy"`. NB09's current baseline
   prompts are long comma-joined synonym strings per class (e.g.
   `"airfield runway, taxiway, apron, light grey paved airfield surface,
   tarmac with faded painted markings"`) — the opposite of what's working
   here. User confirmed this pattern directly: short prompts work better
   than many synonyms at some places.
2. **The `windowsize_exp` sweep on `472_5525` (forest/suburb tile) is the
   clearest single proof point.** At the same crop=1500/stride=1200,
   swapping the tree-class prompt from `"tree"` to `"dense tree canopy"`
   (comparing `segmented0.png` → `segmented6.png`) eliminated large black
   background dropout patches inside the forest region and made the tree
   mask solid and uniform. Same crop/stride, only the prompt wording changed.
3. **Class lists are tailored per-scene, not a fixed universal list** —
   airport tile: `runways/planes/vehicle/parking ground`; farm tile:
   `crop fields/solar panel/industry`; stadium tile:
   `stadium/football court/swimming pool`. NB09 already does this via
   per-tile `IMAGE_CONFIGS` — keep that structure, just shorten the wording.
4. **`slide_crop`/`slide_stride` combos that produced clean results:**
   768/576, 1000/800, 1024/768, 1500/1200 (with the caveat above). Very small
   crops (400/350) were noisier/speckled; crop=1500 without a specific-enough
   prompt left visible black dropout holes. NB09's baseline (1024/768) already
   sits in the working zone — no change needed there for iteration 2.
5. **`prob_thd`/`conf_thd` varied 0.05–0.35 by scene** — lower (0.05) for
   permissive full-coverage scenes (rural fields), higher (0.35) for
   precision-sensitive classes (solar panel, to avoid over-triggering).
   NB09's baseline is 0.1/0.1 — reasonable middle ground, worth nudging per
   weak class rather than globally.

**Plan for iteration 2 based on this:** keep `slide_crop=1024`/`stride=768`
(already in the working zone), but rewrite the `468_5543` tile's `"multi"`
prompt list to be short and concrete per class (closer to the teammate's
style) instead of long synonym strings, and check whether any class shows
the black-hole dropout pattern — if so, apply the same fix as `"dense tree
canopy"`: make that one class's wording more specific/visually descriptive
rather than adding more synonyms.

## Iteration 1 result (v5) — judged against the reference set

Pulled `results` from v5 (single-tile `468_5543`, baseline `1024/768`,
`prob_thd=conf_thd=0.1`). NB09's own Part C panel (single-word vs
multi-synonym vs ambiguous prompts, all same tile/params) directly tested
the "short prompts win" hypothesis from the reference set — **result was the
opposite for this tile's runway/tarmac class**:

- multi-synonym (current baseline): cleanest of the three — runway surface
  reads correctly, planes are tight orange silhouettes, buildings solid blue.
- single-word (`"runway"` alone): runway detection **failed** — large
  white/undetected dropout patches across the tarmac. Same failure mode as
  the reference set's `windowsize_exp` black holes, just via prompt wording
  this time instead of crop size.
- ambiguous (`"paved surface"`): also broke down — runway partly fell back
  to background, tree class over-triggered into areas that aren't trees.

**Conclusion:** "short prompts beat long synonym lists" from the reference
set is not a universal rule — it held for concrete, visually distinctive
nouns (stadium, solar panel, water) but not for large, texturally ambiguous
surfaces like runway/tarmac, where multi-synonym wording that describes the
surface's actual appearance (faded markings, patched asphalt) is what keeps
detection working at stricter thresholds. Sensitivity sweep (prob_thd,
confidence_threshold, slide_stride) showed almost no visible difference
across all 6 variants for this tile — prompt wording is the dominant lever
here, not those three knobs.

## Output format overhaul (before iteration 2)

User clarified the actual ask: reproduce the teammate's **exact output
format** — one full-tile RGB | overlay (α=0.6) image *per config*, own
class-color legend row, own `img_size/prob_thd/conf_thd/slide_stride/
slide_crop` params footer — not the subplot-grid comparisons NB09 was
producing (3×3 grid for Part B, 1×3 grid for Part C). Also wanted a proper
window-size ablation (Image reference A), which NB09 didn't have at all
before — Part B only swept `slide_stride` at fixed `slide_crop=1024`.

Rewrote NB09 cell 11 (`notebooks/NB09_zeroshot_sensitivity.ipynb`):

- New `render_result()` replaces the two `plt.subplots(3,3)`/`plt.subplots(1,3)`
  grid functions — renders one full-size two-panel (RGB | overlay) figure per
  config, matching the teammate's exact layout, saved as its own PNG file.
- Added `WINDOW_SIZE_SWEEP = [(768,576), (1024,768), (1500,1200)]` — new
  **Image reference A** (window-size ablation), sourced directly from the
  crop/stride pairs that showed up as clean results in the teammate's own
  `windowsize_exp` folder.
- Renamed output groups to match the three requested proof panels:
  `{stem}_A_crop{c}_stride{s}.png` (window size), `{stem}_B_probthd{p}.png` /
  `{stem}_B_confthd{c}.png` (thresholds), `{stem}_C_prompt_{label}.png`
  (prompt wording) — each a standalone file, not a subplot.
- Kept the underlying sliding-window/logit-caching engine unchanged — only
  the rendering and sweep organization changed.

Pushed as kernel version 6 (iteration 2), still targeting `TARGET_TILE=
dop20_32_468_5543_1_he` (single tile, per the "one tile at a time" fix from
earlier tonight).

## Iteration 2 result (v6) — new format confirmed, one rendering bug found

v6 completed clean in ~2183s (~36 min), no errors, all 13 individual
full-tile images saved successfully (`results` pulled to
`notebooks/push/nb09/output_v6/output/`). Format overhaul worked as
intended — each config is its own RGB|overlay(α=0.6) panel with its own
legend row, matching the teammate's layout.

**Bug found:** the params footer text (`img_size/prob_thd/conf_thd/
slide_stride/slide_crop`) is missing from every rendered image — blank
space where it should be. Root cause: `fig.text(0.5, -0.02, ...)` places
the text just below the axes in figure-fraction coordinates, but
`savefig(..., bbox_inches="tight")` recomputes the bounding box from
artist extents and appears to be clipping this negative-y text before it's
captured. Fix for iteration 3: move the footer inside the figure's
positive coordinate space (e.g. reserve bottom margin via
`plt.subplots_adjust` and place text at a small positive y, same pattern
NB09's old Part B/C legend row already used successfully) instead of
placing it off-canvas and relying on `bbox_inches="tight"` to include it.

**Ablation findings, now visible at full resolution (confirms v5's
smaller-scale read):**

- **Window-size (Image A):** crop=768/stride=576 renders the solar-panel
  roof array as a clean, well-defined blue rectangle; crop=1500/stride=1200
  loses definition on the same roof (smaller, patchier blue region) and the
  plane cluster on the right edge shows less crisp boundaries. Real,
  visible crop-size effect — smaller crop = higher effective resolution on
  fine structure, consistent with the project's known DOP20 resolution-gap
  reasoning (CLAUDE.md rule 2).
- **Prompt wording (Image C):** single-word prompts (`"runway"` alone)
  cause the tarmac/runway area to render as a flat white/light-grey blob —
  wrong, and buildings show speckled color bleed (blue/green/cyan noise)
  that isn't present in the multi-synonym baseline. Confirms v5's grid-view
  finding, now unambiguous at full resolution.
- **Threshold (Image B):** `confidence_threshold=0.3` (vs. baseline 0.1)
  produces the same runway blob failure as single-word prompts, plus
  speckled building noise — conf_thd=0.3 is too strict for this tile.
  `prob_thd` sweep (0.05/0.1/0.3, free) showed comparatively little visible
  difference, consistent with v5.

**Plan for iteration 3:** fix the params-footer rendering bug (functional
fix, not a quality/prompt change), keep `confidence_threshold` at the
0.1 baseline (0.3 confirmed worse), and address the runway/tarmac
white-blob problem more directly — try tightening the multi-synonym
wording further with concrete surface-texture cues per the v5 finding
(what worked at prob_thd=0.3 in v1's original rework), rather than
touching crop/stride (already validated in the working zone).

## Iteration 3 (v7) + architecture change: decouple inference from rendering

While v7 (footer fix + strengthened runway prompt) was still running, user
compared v6's baseline output directly against the teammate's reference for
the *same tile* (`468_5543`) and flagged real problems beyond the footer bug:

1. **Runway/apron/taxiway class renders as fully transparent background**,
   not just weak — the whole tarmac area has zero color in our output where
   the reference shows a clean grey `runways` fill. Confirms this is a
   detection failure (losing to background), not a rendering bug.
2. **Image aspect ratio was squashed** — our output was 2893×1540 (very
   wide/short) against a near-square 5000×5000 source tile; caused by a
   fixed `figsize=(20,15)` that didn't account for the tile's actual aspect
   ratio, compounded by `subplots_adjust(bottom=...)`.
3. **Legend had dead entries** — `box, container, cargo` and `tree, forest`
   never appear in this scene at all, just cluttering the legend for zero
   pixels of coverage.
4. **Legend labels should be short**, matching the teammate's reference
   exactly (`runway`, `road`, `vehicles`, not the multi-synonym grounding
   strings) — user pointed at a second reference image
   (`dop20_32_479_5550_1_he` water/port scene) showing this explicitly:
   6-9 short class names per legend, clean 5-column swatch grid, no
   redundant α= text outside the panel title.

**Architecture change (user's suggestion, adopted):** decouple SAM3
inference from image rendering. Previously `render_result()` ran inside the
same GPU pass as `run_sliding_window()`, so every legend/layout/color tweak
required a full ~30-60 min Kaggle rerun. Now:

- Section 3 (`run_image()`) only calls SAM3 and saves raw prediction arrays
  (`output/preds/{stem}_{tag}.npy`, uint8 label maps) plus a
  `output/manifest.json` (one row per config: stem, tag, prob_thd, conf_thd,
  slide_stride, slide_crop) and per-tile `output/preds/{stem}_meta.json`
  (display labels, color map, img_size). No matplotlib, no plotting.
- New Section 4 is a **separate, GPU-free cell** that reads the manifest +
  npy files + meta and renders every image. This cell is now the only thing
  that needs to change for legend/layout/color fixes — rerun it alone in the
  Kaggle notebook UI, no repush needed for pure rendering changes.
- Also added a `display` field to each tile's config in `IMAGE_CONFIGS`:
  short 1-2 word legend labels, kept completely separate from `multi` (the
  long synonym strings sent to SAM3 for grounding) — a full multi-synonym
  string can now never leak into a legend again.
- Dropped `box, container, cargo` and `tree, forest` from the `468_5543`
  tile's class list (6 classes now, was 8) since neither appears in this
  scene.
- Fixed the aspect-ratio squash: `render_result()` now derives `figsize`
  from the actual tile's `w/h` ratio instead of a fixed 20×15.

**Still open going into iteration 4:** whether the strengthened runway
prompt (v7, still running as of this note) fixes the background-dropout
problem, or whether the class needs a different approach entirely (e.g.
lower `prob_thd` specifically for this class, or accepting the teammate's
offer to share their template/code directly rather than continuing to
reverse-engineer prompt wording from visual comparison alone — user raised
this as a fallback if the wording fixes don't converge).

## Iteration 3 result (v7) — runway fix did NOT work; root cause identified

v7 completed (~75 min — longer than v6's 36 min, but the strengthened
runway prompt just adds more words to encode per crop, not more crops, so
this is plausibly normal variance rather than a hang; no errors in the log).

**The strengthened runway wording did not fix anything** — visually
identical to v6, entire runway/tarmac area still fully uncolored. Checked
the actual per-class pixel histogram of the saved `.npy` prediction
(`class_idx=0` is the runway class, first in `multi`):

```
class_idx=0 (runway):  0 px        (0.0%)   <- completely absent
class_idx=1 (road):    5,457,843 px (21.8%)
class_idx=2 (aircraft):  330,427 px  (1.3%)
class_idx=3 (car):       216,755 px  (0.9%)
class_idx=5 (building):   99,684 px  (0.4%)
class_idx=7 (grass):   3,294,044 px (13.2%)
class_idx=255 (bg):   15,601,247 px (62.4%)
```

**Root cause identified: this is class confusion, not coverage/threshold.**
The runway class isn't losing to background at the margins (which more
descriptive wording could fix) — it's losing 100% of its pixels to the
`road` class outright. "Paved road, street, dark grey asphalt road" is
out-competing "airfield runway, taxiway, apron..." on every runway/tarmac
pixel, even though those surfaces are visually lighter grey than the road
prompt describes. Adding more runway synonyms (v6→v7) couldn't fix this
because the failure mode is the argmax picking road over runway, not
runway falling below `prob_thd`. This explains why the reference set's
"short prompts win" pattern didn't transfer to this class either — the
problem was never about the runway prompt's specificity, it was about the
**road prompt being too broad** and greedily claiming any grey paved
surface, runway included.

**Fix for iteration 4:** narrow the `road` class prompt so it no longer
describes generic "paved" surfaces — anchor it to road-specific cues that
don't apply to a runway/apron (lane markings, curbs/sidewalks alongside,
vehicle traffic) instead of "dark grey asphalt road" alone, which is
close enough to "light grey paved airfield surface" for SAM3's text
encoder to blur the two.

## Iteration 4 result (v8) — layout/legend fully fixed, runway still unsolved

v8 completed in ~39 min (back to normal v6-range timing). Pulled results
include, for the first time, the full `.npy` prediction cache + manifest +
meta — the inference/rendering split works exactly as designed.

**What's now fixed and matches the teammate's reference format:**
- Image aspect ratio: no longer squashed, panels are properly proportioned
  for the near-square 5000×5000 tile.
- Legend: 6 short display labels (`runway`, `road`, `aircraft`, `vehicles`,
  `building`, `grass`), no overflow, no dead entries, matches the clean
  swatch-grid style from the reference images.
- Params footer: renders correctly (`img_size`, `prob_thd`, `conf_thd`,
  `slide_stride`, `slide_crop`), no more clipping.
- Detection quality elsewhere in the tile (aircraft, buildings, vehicles,
  grass/low-veg) is clean and visually close to the reference bar.

**Runway is still unsolved.** Checked the new prediction's pixel histogram:
`class_idx=0` (runway) is still exactly 0 px. The road-prompt narrowing did
work as intended — road's share dropped from 21.8% (v7) to 3.7% (v8) — but
those freed pixels went to **background** (62.4%→80.3%), not to runway.
This means runway isn't losing to road anymore, but it's still losing the
argmax to background everywhere, i.e. the runway prompt itself is too weak
to win against *any* competing signal, not just against road specifically.
Narrowing road was a real, correct fix for the class-confusion problem it
was aimed at, but it wasn't sufficient on its own to make runway win.

**Not attempted tonight, worth trying next:** a runway-specific lower
`prob_thd` isn't supported by the current single-global-threshold
architecture; the more promising untried lever is dropping the runway
prompt back down to something shorter and more distinctive (undoing the
"more synonyms" approach entirely, in line with what actually worked for
the teammate's reference images — short, concrete nouns), or accepting
runway will need the teammate's actual code/config rather than continued
reverse-engineering, per the user's stated fallback plan.

## Session summary (4 iterations, v5-v8)

1. **v5** — single-tile scoping fix (already correctly configured before
   this session started), old subplot-grid rendering.
2. **v6** — output format overhaul to match teammate's per-config
   full-image style; introduced a params-footer rendering bug.
3. **v7** — fixed the footer bug; strengthened runway prompt wording,
   which had zero effect (confirmed via pixel histogram — the real problem
   was road, not runway, being too permissive).
4. **v8** — fixed the actual root cause (narrowed road prompt) + shipped
   the inference/rendering architecture split + fixed aspect ratio + fixed
   legend to short display-only labels. Layout/legend/footer now match the
   reference format. Runway detection itself remains an open problem.

All 4 iterations, plus the devlog and per-run params CSV, are committed
(`3e8479f`, `324a3ba`, `73812d7`). Nothing was left running or half-pushed.

## Iteration 4: inference/rendering split shipped (architecture, not a rerun)

Implemented the decouple-inference-from-rendering change from the previous
note. Section 3 (`run_image()`) now only calls SAM3 and saves
`output/preds/{stem}_{tag}.npy` + `output/manifest.json` +
`output/preds/{stem}_meta.json` — no plotting. New Section 4 is a
standalone GPU-free cell that reads those files and renders every image;
edit that cell alone and rerun it in the Kaggle UI (no repush, no SAM3,
no ~40-75 min wait) for any future legend/layout/color-only change.
Also: added `display` (short legend labels, kept separate from `multi`,
the long grounding strings — a full synonym string can no longer leak into
a legend) and dropped the two dead-weight classes (`box, container, cargo`,
`tree, forest`) that never appear in this tile.

## Tile 4 — `DOP20_32_525_5604_1_he` (Vogelsbergkreis-Lautertal solar farm)

Added as a 4th tile for direct comparability against a known-good reference
result on this exact tile. Uses per-tile baseline override
(`prob_thd=0.05, confidence_threshold=0.35, slide_crop=1500, slide_stride=1200`)
instead of the shared default, plus its own class list: building, grass,
road, vehicles, trees, solar panel.

**v10/v11 push mistakes (both non-fatal, caught before wasting significant
GPU time):**
- v10 was pushed with the push-folder copy still containing stale tile-2
  code (forgot to sync `notebooks/NB09_zeroshot_sensitivity.ipynb` →
  `notebooks/push/nb09/` before pushing) — caught via `grep -c` diff between
  the two copies, fixed by syncing and re-pushing as v11.
- v11 failed immediately with `NOT FOUND: DOP20_32_525_5604_1_he.jpg` —
  the `vogelsbergkreis-lautertal-dop20` dataset was never added to
  `kernel-metadata.json`'s `dataset_sources`, so Kaggle never mounted it.
  Fixed by adding the dataset slug, re-pushed as v12.
- v10 also turned into an orphaned session (stuck `RUNNING` on Kaggle's own
  dashboard well past normal duration) even though it had already been
  superseded — cancelled manually via the website once noticed, since no
  API-level stop exists.

**v12 result:** completed cleanly, all 13 configs saved. Baseline pixel
histogram — building 0.5%, grass 61.6%, road 20.1%, vehicles 0.0%, trees
5.4%, solar panel 12.1%, background 0.3%. No class-confusion problem like
tile 1's runway — every class present, background near-zero, visually
matches the reference's detection quality. The one gap: v12 ran before the
rendering-layout fix (exact match to the team's own plotting code, see
below) landed, so its output PNGs still have the old wide-margin layout.
Since inference and rendering are now decoupled, re-rendering these 13
predictions with the corrected layout doesn't require a Kaggle rerun — it
only needs the source tile image, which isn't available locally (only the
team's own pre-rendered output PNG is), so this is left as a known gap
rather than forcing another GPU run for a cosmetic-only fix on an already-
successful tile.

## Rendering layout — matched to the team's exact plotting code

The team shared their own exact matplotlib plotting code (verbatim, via
chat). Replaced NB09's `render_result()` with a line-for-line match:
`figsize=(20, 12)`, `subplots_adjust(left=0.01, right=0.99, top=0.95,
bottom=0.15, wspace=0.01)`, legend at `bbox_to_anchor=(0.5, 0.075)` with
bold weight, a light-gray separator line at y=0.055, the same `meta_text`
format/position, `dpi=200` with `bbox_inches='tight'`. This is a pure
rendering change against the split architecture — applies to any future
tile's already-saved predictions without new inference.

## Running two tiles concurrently

Kaggle allows up to 2 concurrent GPU sessions per account but only one
active version per kernel object — pushing a new version to a kernel that's
still running doesn't run in parallel, it errors with "Maximum batch GPU
session count of 2 reached" (if the account is already at the cap) or just
queues behind the current run. To get two tiles running at once, created a
second, separately-named kernel (`harish77718/nb09-tile2-475-5550`) as its
own push target — same notebook content, only `TARGET_TILE` differs. Both
kernels share the same underlying notebook code as it evolves (config,
rendering fixes) via manual copy + a scripted `TARGET_TILE` swap, so keep
both push folders (`notebooks/push/nb09/`, `notebooks/push/nb09-tile2/`) in
sync when the shared logic changes.

## Iteration 1 — v5 push, tile `dop20_32_468_5543_1_he` (Frankfurt airport)

- **Pushed:** 2026-07-18 ~22:48
- **Config:** `TARGET_TILE=dop20_32_468_5543_1_he`, baseline
  `prob_thd=0.1, confidence_threshold=0.1, slide_crop=1024, slide_stride=768`
- **Prompts (multi-synonym baseline):** runway/taxiway/apron, road, aircraft,
  car, box/cargo, building, tree, grass — see cell 11 `IMAGE_CONFIGS` in the
  notebook for full wording.
- **Status:** running — see below for outcome once pulled.

## Tile 2 — `dop20_32_475_5550_1_he` (Frankfurt Hbf) result

Ran via the second parallel kernel (`nb09-tile2-475-5550`, v2 after a v1
conda-network retry). Completed cleanly, exact-match rendering layout
confirmed correct (tight margins, gray separator, bold legend — matches
the reference format).

**Same class-confusion failure as tile 1's runway, on railway/platform/road
this time.** Baseline pixel histogram:

```
building:   25.8%
vehicles:    2.1%
trees:       0.8%
grass:      11.7%
background: 59.7%
railway:     0.0%   <- completely absent
platform:    0.0%   <- completely absent
road:        0.0%   <- completely absent
```

Visually, the rail-yard area (the dominant feature of this scene) is mostly
untouched background, while grass (cyan) has spread onto areas that are
clearly urban/paved in the source image, not grass — same over-permissive-
prompt pattern that swallowed tile 1's runway class, just with grass as the
culprit this time instead of road. Needs the same fix approach: narrow the
grass prompt to grass-specific cues (green color, blade texture) that don't
match paved/gravel rail-yard surfaces, rather than continuing to add more
railway/platform synonyms (already tried once for platform in the v1
rework, without success on this run).

**Update after inspecting v2's Part-C prompt-wording panel directly**
(user's own observation): the **single-word** variant's `"railway"` prompt
found 10.9% railway coverage (patchy, but real detection) vs. **0%** for
the elaborate multi-synonym prompt in the same baseline run. This is the
*opposite* lesson from tile 1's runway class (where multi-synonym beat
single-word) — confirms there is no universal "longer is better" or
"shorter is better" rule; it's genuinely per-class and has to be checked,
not assumed. Road was also visibly reasonable ("okayish") in the
single-word variant. Platform stayed at 0% across every prompt variant
tested (multi, single, ambiguous) in this scene — dropped from the class
list entirely rather than continue guessing at wording for a class that's
never once produced a detection here.

**Fix applied for the v3→v4 iteration:** switched tile 2's railway `multi`
prompt from the long synonym string down to short, concrete wording
(`"railway track, train tracks, rail line"`), and removed `platform` from
the class list (`multi`, `display`, `single`, `ambiguous`, `colors` all
updated — now 6 classes instead of 7). Grass narrowing (previous fix) is
also still in this run; v3 (grass-fix-only) was pushed and completed/is
completing separately before this second fix layers on top.

<!-- Next iterations appended below as they land -->
