# CLAUDE.md — rg-segearth-ov3

Read this first at session start. Ground truth for this project.

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

**TODO:** Run our NB02 to get our own Potsdam mIoU number with enriched prompts.
**TODO:** Run NB03 to get quantitative F1 on Darmstadt via OSM pseudo-GT.

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
- `data/` — local sample patches for NB01 verification. Not committed.

## Notebooks

| Notebook | Runs on | Purpose |
|---|---|---|
| NB01_verify_data.ipynb | Local (no GPU) | Preview DOP20 patches, confirm RGBI, check class lists |
| NB02_potsdam_eval.ipynb | Kaggle T4 | Quantitative eval on Potsdam val tile → mIoU / mAcc |
| NB03_hessen_infer.ipynb | Kaggle T4 | Hessen inference + OSM pseudo-GT F1 |
| NB04_demo.ipynb | Kaggle T4 | Live presentation demo (open-vocab, audience picks vocab) |

## Kaggle dataset slugs

| Dataset | Slug | Contents |
|---|---|---|
| SAM3 weights | `dummyirl/sam3-weights` | `sam3.pt` checkpoint |
| Darmstadt DOP20 | `dummyirl/darmstadt-dop20` | Hessen DOP20 patches (RGBI) |
| Potsdam tiles | `rachanaamraghuthama/rskt-potsdam-test-data` | Potsdam GeoTIFF + labels |

## Working style

Teach-as-you-go: section-by-section with confirmation checkpoints.
Flag irreversible actions. Honest assessment over optimism. Do not invent results.
