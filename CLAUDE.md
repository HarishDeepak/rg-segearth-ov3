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

**Potsdam** — `rachanaamraghuthama/rskt-potsdam-test-data`
- Kaggle path: `/kaggle/input/rskt-potsdam-test-data/6ISPRS/`
- Files: `top_potsdam_{tile}_RGB.tif` + `top_potsdam_{tile}_label_noBoundary.tif`
- Tiles: 5_14, 5_15, 6_13, 6_14, **6_15** (val), 7_13
- Label format: **TBD — run NB01 cell 1.3 to confirm indexed vs RGB**

**DOP20 (Hessen/Darmstadt)** — `dummyirl/darmstadt-dop20`
- Kaggle path: `/kaggle/input/darmstadt-dop20/Darmstadt_dop20_presliced/darmstadt_dop20/images/`
- Files: `dop20_32_474_5532_1_he_y{Y}_x{X}.png` — pre-sliced 256×256 PNG patches, ~1267 files
- Channel format: **TBD — likely RGB (3ch) not RGBI (4ch) since PNG; NB01 will confirm**
- NB03 implication: patches are already 256px — run each patch directly, no sliding window needed

**Kaggle path rule:** `/kaggle/input/{slug-name}/` — only the part after the `/` in the slug.
`dummyirl/darmstadt-dop20` → `/kaggle/input/darmstadt-dop20/` (not `/kaggle/input/datasets/dummyirl/...`)

## Kaggle dataset slugs

| Dataset | Slug | Contents |
|---|---|---|
| SAM3 weights | `dummyirl/sam3-weights` | `sam3.pt` checkpoint |
| Darmstadt DOP20 | `dummyirl/darmstadt-dop20` | Hessen DOP20 patches (RGBI) |
| Potsdam tiles | `rachanaamraghuthama/rskt-potsdam-test-data` | Potsdam GeoTIFF + labels |

## Working style

Teach-as-you-go: section-by-section with confirmation checkpoints.
Flag irreversible actions. Honest assessment over optimism. Do not invent results.
