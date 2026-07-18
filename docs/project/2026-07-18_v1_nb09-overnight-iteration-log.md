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

## Iteration 1 — v5 push, tile `dop20_32_468_5543_1_he` (Frankfurt airport)

- **Pushed:** 2026-07-18 ~22:48
- **Config:** `TARGET_TILE=dop20_32_468_5543_1_he`, baseline
  `prob_thd=0.1, confidence_threshold=0.1, slide_crop=1024, slide_stride=768`
- **Prompts (multi-synonym baseline):** runway/taxiway/apron, road, aircraft,
  car, box/cargo, building, tree, grass — see cell 11 `IMAGE_CONFIGS` in the
  notebook for full wording.
- **Status:** running — see below for outcome once pulled.

<!-- Next iterations appended below as they land -->
