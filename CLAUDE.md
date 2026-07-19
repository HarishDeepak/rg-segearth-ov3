# CLAUDE.md — rg-segearth-ov3

Read this first at session start. Ground truth for this project.

## This repo
`https://github.com/HarishDeepak/rg-segearth-ov3` (private)

## What this project is

Open-vocabulary remote sensing segmentation (Fraunhofer IGD Praktikum, SoSe 2026).
Inference-only pipeline using **SegEarth-OV-3** (SAM3 + open-vocab text prompts) on
Hessen DOP20 (20cm GSD) and Potsdam ISPRS (5cm GSD, for quantitative baseline).

No training — SAM3 is fully frozen. Improvements come from:
- Enriched multi-synonym class prompts (`cls_hessen.txt`, `cls_potsdam.txt`)
- Gaussian-weighted sliding window (reduces seam artifacts)
- Smaller `slide_crop=256` / `stride=128` for Hessen (compensates 4× resolution gap)
- Optional PAMR boundary refinement (available in pamr.py, not default)

The SegEarth-OV-3 code lives at `https://github.com/HarishDeepak/SegEarth-OV-3` (our fork of
`dummy-irl/SegEarth-OV-3`). Kaggle notebooks clone from there at runtime.

## Related project

`D:\VC\rg-geoprompt-peft` — the GeoPrompt/DINOv2 supervised training project.
That project trained on Potsdam (84.9% mIoU) and ran zero-shot on Darmstadt (mean F1 0.2059).
SegEarth-OV-3 is a separate method for comparison.

## Confirmed results

### Teammate's SegEarth-OV-3 on Darmstadt (qualitative, no metrics yet)
- 12-class open-vocab (single-name prompts): **7.9/10** visual quality
- Our GeoPrompt: 6.7/10 (weaker, especially on cars)

### Quantitative Potsdam zero-shot (from literature, NOT our run yet)
- SegEarth-OV: 48.5% mIoU (zero-shot)
- SegEarth-OV-3: 57.8% mIoU (zero-shot, SAM3 backbone)

**TODO:** Run NB01 on Kaggle to confirm label format (indexed vs RGB) before NB02.
**TODO:** Run NB02 to get our own Potsdam mIoU number with enriched prompts.
**TODO:** Run NB03 for Hessen — visual comparison only (no OSM F1 for now).

## Non-negotiable rules

1. **Inference only** — SAM3 is frozen. Never try to backprop through SAM3 unless
   explicitly designing a PEFT experiment.
2. **Resolution gap** — Hessen DOP20 is 20cm GSD, Potsdam is 5cm (4× gap).
   Use `slide_crop=256` / `stride=128` for Hessen (smaller crops = higher effective
   resolution when SAM3 resizes to 1008×1008 internally).
3. **Potsdam eval split** — val tile = `6_15`. Never use random split (patch overlap
   → leakage). Same constraint as rg-geoprompt.
4. **Darmstadt eval is patch-based** — Never stitch predictions for metrics.
   OSM pseudo-GT: rasterize → erode 3×3 (borders→255) → F1.
5. **Our fork is the source of truth** — all config/prompt changes go to
   `HarishDeepak/SegEarth-OV-3`, pushed to GitHub before running on Kaggle.
6. **DOP20 is RGBI** — always take only first 3 bands for SAM3.
7. **SAM3 checkpoint** — Kaggle path: `/kaggle/input/sam3-weights/sam3.pt`
   (dataset `dummyirl/sam3-weights`). Never hardcode elsewhere.

## Code layout

- `src/segearth_utils/` — our thin utilities (constants, osm_eval). No SAM3 code here.
- `notebooks/` — 4 notebooks (see below). Logic in modules, notebooks orchestrate.
- `results/` — output PNGs, CSVs, logits. Not committed (gitignored).

## Notebooks

| Notebook | Runs on | Datasets | Purpose |
|---|---|---|---|
| NB01_verify_data.ipynb | Kaggle CPU | rskt-potsdam-test-data, darmstadt-dop20 | Verify datasets, confirm label format (indexed vs RGB), preview patches |
| NB02_potsdam_eval.ipynb | Kaggle T4 | sam3-weights, rskt-potsdam-test-data | Quantitative eval on Potsdam val tile `6_15` → mIoU / mAcc |
| NB03_hessen_infer.ipynb | Kaggle T4 | sam3-weights, darmstadt-dop20 | Hessen inference → visual comparison (no F1 for now) |
| NB04_demo.ipynb | Kaggle T4 | sam3-weights, darmstadt-dop20 | Live presentation demo (open-vocab, audience picks vocab) |

## Dataset structures (confirmed)

**Potsdam** — `dummyirl/6isprs`
- Kaggle path: `/kaggle/input/datasets/dummyirl/6isprs/`
- Files: `top_potsdam_{tile}_RGB.tif` + `top_potsdam_{tile}_label_noBoundary.tif`
- Tiles: 5_14, 5_15, 6_13, 6_14, **6_15** (val), 7_13
- Label format: **TBD — run NB01 cell 1.3 to confirm indexed vs RGB**

**DOP20 (Hessen/Darmstadt)** — `harish77718/darmstadt-dop20-presliced`
- Kaggle path: `/kaggle/input/datasets/harish77718/darmstadt-dop20-presliced/darmstadt_dop20/images/`
- Files: `dop20_32_474_5532_1_he_y{Y}_x{X}.png` — pre-sliced 256×256 PNG patches, ~1267 files
- Channel format: **TBD — likely RGB (3ch); NB01 will confirm**
- NB03 implication: patches are already 256px — run each patch directly, no sliding window needed

## Kaggle dataset slugs

| Dataset | Slug | Kaggle mount path |
|---|---|---|
| SAM3 weights | `dummyirl/sam3-weights` | `/kaggle/input/sam3-weights/` |
| Hessen DOP20 | `harish77718/darmstadt-dop20-presliced` | `/kaggle/input/datasets/harish77718/darmstadt-dop20-presliced/darmstadt_dop20/images/` |
| Potsdam ISPRS | `dummyirl/6isprs` | `/kaggle/input/datasets/dummyirl/6isprs/` |

## NB09 — sensitivity/ablation work (active, `nb09-overnight-iteration` branch)

Separate from the NB01-04 pipeline above. `notebooks/NB09_zeroshot_sensitivity.ipynb`
sweeps window size, `prob_thd`/`confidence_threshold`, and prompt wording on
4 full-tile images, producing per-config full-tile RGB|overlay images
matching the team's own reference plotting style (exact layout code, not
approximated).

**Architecture:** inference (SAM3 sliding-window passes) and rendering are
decoupled. The inference cell only saves raw `.npy` label maps +
`manifest.json` + per-tile `meta.json` (display labels, colors, img_size) —
no plotting. A separate GPU-free cell renders images from those files, so
legend/layout/color-only fixes never require a Kaggle rerun, only a local
edit + a rerun of that one cell (or a cheap repush).

**Two Kaggle kernels run in parallel** (Kaggle caps concurrent GPU sessions
at 2, but only one active version per kernel object — a genuine 2nd kernel
is needed for real concurrency):
- `harish77718/nb09-zero-shot-sensitivity` — main kernel, push folder
  `notebooks/push/nb09/`
- `harish77718/nb09-tile2-475-5550` — second kernel, push folder
  `notebooks/push/nb09-tile2/`

Both copies must be manually kept in sync with `notebooks/NB09_zeroshot_sensitivity.ipynb`
(`cp` after any shared-code edit) — a stale push-folder copy silently reruns
old code; always `grep` for a fresh-edit marker in the push-folder file
before pushing to confirm the sync worked.

**4 tiles:**

| # | Tile | Dataset | Notes |
|---|---|---|---|
| 1 | `dop20_32_468_5543_1_he` | `dummyirl/frankfurt-dot20` | Airport/runway. Runway class detection unresolved — road/generic prompts keep winning the argmax over it even after narrowing. |
| 2 | `dop20_32_475_5550_1_he` | `dummyirl/frankfurt-dot20` | Frankfurt Hbf, railway/platform. Same class-confusion pattern as tile 1 (grass swallowed railway/platform/road); grass prompt narrowed, retry in progress. |
| 3 | `dop20_32_476_5524_1_he` | `dummyirl/darmstadt-dop20` | Water/sports/roads. Road+grass prompts pre-narrowed before first run based on tiles 1-2's pattern. |
| 4 | `DOP20_32_525_5604_1_he` | `dummyirl/vogelsbergkreis-lautertal-dop20` | Solar farm — added for direct comparability with a known-good team reference result on this exact tile (their exact params/class list used as this tile's baseline override). Completed successfully, no class-confusion issue. |

**Recurring failure mode:** a generic class prompt (bare "road", "grass",
etc.) tends to win the pixel-wise argmax across broad grey/mixed-texture or
green areas, starving a more specific but narrower class (runway, railway,
platform) down to 0 px — check via `np.unique(pred, return_counts=True)` on
the saved `.npy`, not just visually. Fix is narrowing the *permissive*
class's wording with concrete visual cues (lane markings/curbs for road,
"bright green grass blades" for grass), not strengthening the starved
class's wording (tried repeatedly on tile 1's runway, never worked).

**Devlog:** `docs/project/2026-07-18_v1_nb09-overnight-iteration-log.md` —
full iteration-by-iteration history, findings, and root causes. Read this
before making further NB09 changes; it has the reasoning behind every prompt
tweak so far.

## Working style

Teach-as-you-go: section-by-section with confirmation checkpoints.
Flag irreversible actions. Honest assessment over optimism. Do not invent results.
