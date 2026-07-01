# plan1.md — Handoff for Next Session
_Created 2026-06-27. Read this at session start before touching anything._

---

## State as of this handoff

### What we are doing (one-line)
Running SegEarth-OV-3 (SAM3 + open-vocab text prompts, fully frozen, no training) on two datasets:
- **Potsdam ISPRS** (5cm GSD) — for quantitative zero-shot mIoU baseline
- **Hessen DOP20 / Darmstadt** (20cm GSD) — for qualitative visual comparison vs GeoPrompt

Goal: Praktikum SoSe 2026 — show competitive or better zero-shot performance vs supervised GeoPrompt (84.9% mIoU Potsdam, 0.2059 F1 Darmstadt).

---

## Auth and Infrastructure — Already Working

### Kaggle CLI authentication
- CLI 2.2.2 uses `api.kaggle.com` (NOT `www.kaggle.com/api/v1`). Legacy API key auth does NOT work with this endpoint.
- OAuth credentials live in `C:\Users\haris\.kaggle\credentials.json` (373 bytes, created 2026-06-27 20:50 via `kaggle auth login --force`)
- `C:\Users\haris\.kaggle\kaggle.json` has `{"username": "harish77718"}` — NO `key` field. Keeping `key` out forces the CLI to use OAuth.
- Push command: `kaggle kernels push -p notebooks/push/nb02` or `notebooks/push/nb03`

### T4 GPU
- `"machine_shape": "NvidiaTeslaT4"` present in BOTH `notebooks/push/nb02/kernel-metadata.json` and `notebooks/push/nb03/kernel-metadata.json`
- This was NOT there before — it defaulted to P100. Now fixed.

### GitHub fork (source of truth)
- `https://github.com/HarishDeepak/SegEarth-OV-3` — our fork
- `https://github.com/HarishDeepak/rg-segearth-ov3` — this repo (config, notebooks, utils)
- Kaggle notebooks clone `HarishDeepak/SegEarth-OV-3` at runtime to `/tmp/SegEarth-OV-3`
- ANY config/prompt changes MUST be committed + pushed to `rg-segearth-ov3` (and `SegEarth-OV-3` if code changes) BEFORE running on Kaggle

---

## Critical Bug Fixed — MUST KNOW

### NB02 label indexing bug (fixed in version 16, result pending)

**What was broken:** `RGB_TO_IDX` in NB02 used 0-indexed class labels (0–5). `PotsdamDataset` in mmseg uses `reduce_zero_label=True` which maps label 0 → 255 (ignored, excluded from all metrics). This caused ~40% of pixels (impervious surface, the most common class) to be completely invisible to the evaluator. mIoU was catastrophically wrong: **3.69%** vs teammate's 62.26%.

**Why visual preds looked fine:** The model detected real structures spatially. The visualization just shows colored predictions — it doesn't compare against GT inline. Only the metric computation was broken.

**What was fixed (cell `adb25d60` in NB02):**
```python
# CORRECT — 1-indexed so reduce_zero_label maps 1→0, 2→1, ..., 6→5 (all valid)
RGB_TO_IDX = {
    (255, 255, 255): 1,  # impervious surface
    (  0,   0, 255): 2,  # building
    (  0, 255, 255): 3,  # low vegetation
    (  0, 255,   0): 4,  # tree
    (255, 255,   0): 5,  # car
    (255,   0,   0): 6,  # clutter
}
```

**Rule:** `_noBoundary` label files have no class-0 boundary pixels so it is safe to use 0 as "nothing present". 1-indexed means class 0 stays 255 (ignored) after reduce_zero_label — exactly right.

**NB02 version 16 was pushed with this fix and was running when this session ended. Check results first thing next session.**

---

## Current Results

### Teammate's Potsdam result (ALL 6 tiles as val, not held-out)
```
impervious_surface: 75.97
building:           86.13
low_vegetation:     64.92
tree:               46.02
car:                79.98
clutter:            20.53
mIoU: 62.26   aAcc: 81.20   mAcc: 75.74
```
Used single-word prompts. Used PNG labels (already indexed). GPU: T4 ×2.
**Important:** They used ALL tiles as val. Our eval uses only tile 6_15 (proper split, no leakage).

### Our NB02 result BEFORE fix (do not use — wrong)
mIoU: 3.69% — caused by label indexing bug (see above). Ignore this number entirely.

### NB03 Hessen visual result
Running at end of session. User confirmed visual improvements are visible.
No quantitative metric yet (no GT). OSM pseudo-GT F1 planned but not implemented.

### Literature baselines (for report)
- SegEarth-OV: 48.5% mIoU Potsdam zero-shot
- SegEarth-OV-3: 57.8% mIoU Potsdam zero-shot (published)
- If our tile-6_15 run shows ~55-65%, we're in the right range
- If teammate-equivalent all-tiles run shows ~62%, consistent with published numbers

---

## Config Files — Current State

### configs/cls_potsdam.txt (DO NOT REVERT)
```
impervious surface, road, pavement, paved ground
building, rooftop, structure
low vegetation, grass, shrub, lawn, meadow
tree, forest, canopy
car, vehicle
clutter, background
```
Previously was single-word: `road`, `building`, `grass`, `tree`, `car`, `clutter`. The old prompts hurt all classes that have multiple visual forms (especially class 1: "road" was too narrow for all impervious surfaces).

### configs/cfg_potsdam.py (key settings)
```python
slide_stride=1024, slide_crop=1024   # full tile inference
bg_idx=5                              # clutter is background
prob_thd=0.1                          # presence threshold
img_suffix='_RGB.tif'
seg_map_suffix='_label_noBoundary.tif'
data_root='/kaggle/working/PotsdamEval'
```
Val tile: `6_15` only (NEVER use random split — patch overlap → leakage).

### configs/cls_hessen.txt
NOT YET UPDATED. Current content has 7 Hessen/Darmstadt classes.
**Next step:** Add viewpoint descriptors (see improvement plan below).

---

## Improvement Plan (Priority Order)

### DONE
- [x] Fix label indexing bug in NB02 → pushed version 16
- [x] Improve cls_potsdam.txt with multi-synonym prompts
- [x] T4 GPU fix in both kernel-metadata.json files

### NEXT (in order)

#### 1. Check NB02 version 16 result
Open `https://www.kaggle.com/code/harish77718/nb02-potsdam-eval` and check if it ran to completion. The mIoU output should be much higher than 3.69%. Expected range: 40–65%.

If still wrong: check whether `PotsdamDataset` was properly importing our local label files (the notebook prepares them in `/kaggle/working/PotsdamEval/ann_dir_indexed/val/`). The cell that saves `.tif` files must complete before eval runs.

#### 2. Add nadir viewpoint descriptors to cls_hessen.txt
Change from:
```
impervious surface, road, pavement, paved ground
building, rooftop, structure
...
```
To format like:
```
nadir aerial view of road, paved surface, impervious ground, asphalt
overhead view of building rooftop, flat roof casting shadow
aerial top-down view of tree canopy, forest patch
overhead view of vehicle, parked car
nadir view of low vegetation, grassland, lawn
aerial view of railway track, rail line
overhead view of paved urban street
```
Push commit, then re-push NB03 (`kaggle kernels push -p notebooks/push/nb03`).

Why this matters: CLIP text encoder is biased toward ground-level photography. "building" activates facade/window features which are invisible from above. "overhead view of building rooftop" shifts the text embedding toward aerial top-down features. Confirmed in LAE-80C study: viewpoint descriptors give biggest zero-shot improvement for aerial data.

#### 3. Per-patch presence filtering in NB03
In the stitch cell, add:
```python
PRESENCE_THRESHOLD = 0.2

for patch, presence_scores in zip(patches, all_presence_scores):
    for class_idx, score in enumerate(presence_scores):
        if score < PRESENCE_THRESHOLD:
            patch.logits[class_idx] = float('-inf')  # won't win argmax
```
Keep ALL class indices fixed (same N classes every patch, same palette) — only suppress absent classes by setting logits to -inf. This ensures consistent blending indices at seams.

Also produce dual output after stitch:
- **Fine output:** full N-class prediction with distinct colors
- **Coarse output:** merge via COARSE_MAP to 6-7 canonical groups (built, vegetation, impervious, water, bare, vehicle, clutter)

#### 4. OSM pseudo-GT F1 for Hessen evaluation
Port `osm_eval.py` from `D:\VC\rg-geoprompt-peft` to this project.
Steps:
1. Query Overpass API for Darmstadt bounding box: `(49.8°N, 8.6°E, 49.9°N, 8.8°E)` approximately
2. Rasterize OSM polygons to 256×256 masks per class
3. Erode 3×3 pixels (set boundary pixels to 255 = ignored)
4. Load NB03 `.npy` prediction outputs
5. Compute F1 per class and mean

This gives the number needed for the Praktikum report. Compare against GeoPrompt's 0.2059 mean F1 on Darmstadt.

#### 5. PTSAM decoder tuning (future, if time)
Feasibility: Yes, possible with ~2,048 trainable params (soft prompt tokens injected into SAM3 mask decoder cross-attention only). PE-L+ encoder stays 100% frozen.
- Train on 5 Potsdam tiles with resolution bridge augmentation (30% chance: downsample ×4, then upsample back = simulates 20cm GSD from 5cm training data)
- No overfitting risk at 2,048 params on 5 images
- Transfers to any German city — tokens encode "aerial imagery style" not "Potsdam specific"
No implementation code exists yet. Would need to write training loop.

#### 6. OSM auto-prompting (future, novel contribution)
Query Overpass API for each tile → get unique OSM tags present → map to class names → send to LLM for aerial description generation → run inference with tile-specific prompts.
This automatically adapts to any German region without manual prompt engineering.

---

## What NOT To Do

1. **Never use random tile split for Potsdam eval.** Val tile is always `6_15`. Patch-level random split has overlap → leakage.
2. **Never try to backprop through PE-L+ (image encoder).** It's 307M params, frozen. Would catastrophically overfit on 5 tiles.
3. **Never pre-upsample patches (SOPSeg method) for our Hessen patches.** SAM3 already upsamples internally to 1008×1008. Pre-upsampling adds fake bilinear texture that confuses the encoder.
4. **Never commit NB02 label files as 0-indexed again.** The correct range is 1–6 (or 1–N). Zero maps to 255 (ignored) after reduce_zero_label.
5. **Never push notebooks to Kaggle without pushing the rg-segearth-ov3 config changes to GitHub first.** The notebook clones from GitHub at runtime — if config isn't on GitHub, the notebook uses stale config.
6. **DOP20 is RGBI (4 channels).** Always take only first 3 bands before feeding to SAM3. (Already handled in inference code but keep this in mind if adding new data loading.)

---

## File Map

```
D:\VC\rg-segearth-ov3\
├── configs/
│   ├── base_config.py         ← mmseg base settings (eval type, hooks, etc.)
│   ├── cfg_potsdam.py         ← Potsdam eval config (stride, crop, paths)
│   ├── cls_potsdam.txt        ← UPDATED: 6 class prompts with synonyms ✓
│   └── cls_hessen.txt         ← TODO: add nadir viewpoint descriptors
├── notebooks/
│   ├── NB01_verify_data.ipynb ← Kaggle CPU: verify label format, preview patches
│   ├── NB02_potsdam_eval.ipynb← Kaggle T4: Potsdam mIoU eval (label bug FIXED)
│   ├── NB03_hessen_infer.ipynb← Kaggle T4: Hessen inference, visual output
│   ├── NB04_demo.ipynb        ← Kaggle T4: live demo with audience prompt input
│   └── push/
│       ├── nb02/
│       │   ├── NB02_potsdam_eval.ipynb  (copy for push)
│       │   └── kernel-metadata.json     (T4 GPU fixed ✓, version 16 pushed)
│       └── nb03/
│           ├── NB03_hessen_infer.ipynb  (copy for push)
│           └── kernel-metadata.json     (T4 GPU fixed ✓)
├── src/segearth_utils/
│   ├── constants.py           ← class name lists, palette colors
│   └── osm_eval.py            ← OSM pseudo-GT F1 (port from GeoPrompt, not done)
├── extra/
│   ├── infoplan.md            ← full session technical notes (read this for context)
│   └── plan1.md               ← this file
├── archive/
│   ├── lastclaudechat.md      ← full prior session (GeoPrompt vs OV-3 comparison, PEFT analysis)
│   └── Remote Sensing Segmentation Adaptation.md ← Gemini academic survey
└── teammate/
    └── segearth03 (4).ipynb   ← teammate's run, 62.26% mIoU, all-tiles eval
```

---

## Kaggle Notebook URLs

| Notebook | URL |
|---|---|
| NB01 | https://www.kaggle.com/code/harish77718/nb01-verify-data |
| NB02 | https://www.kaggle.com/code/harish77718/nb02-potsdam-eval |
| NB03 | https://www.kaggle.com/code/harish77718/nb03-hessen-infer |

---

## Dataset Slugs and Kaggle Paths

| Dataset | Slug | Kaggle path |
|---|---|---|
| SAM3 weights | `dummyirl/sam3-weights` | `/kaggle/input/sam3-weights/sam3.pt` |
| Potsdam ISPRS | `dummyirl/6isprs` | `/kaggle/input/datasets/dummyirl/6isprs/6ISPRS/` |
| Hessen DOP20 | `harish77718/darmstadt-dop20-presliced` | `/kaggle/input/datasets/harish77718/darmstadt-dop20-presliced/darmstadt_dop20/images/` |

DOP20 file pattern: `dop20_32_474_5532_1_he_y{Y}_x{X}.png` — pre-sliced 256×256 PNG patches, ~1267 files.
Potsdam file pattern: `top_potsdam_{tile}_RGB.tif` + `top_potsdam_{tile}_label_noBoundary.tif`

---

## Commit Before Pushing to Kaggle Checklist

```
1. git add configs/cls_potsdam.txt configs/cls_hessen.txt  (whichever changed)
2. git add notebooks/NB02_potsdam_eval.ipynb notebooks/NB03_hessen_infer.ipynb
3. git add notebooks/push/nb02/ notebooks/push/nb03/
4. git commit -m "..."
5. git push origin master
6. kaggle kernels push -p notebooks/push/nb02
   kaggle kernels push -p notebooks/push/nb03
```

Do NOT run the kaggle push before the git push. The notebook clones from GitHub at runtime.

---

## Understanding SegEarth-OV-3 Architecture (Quick Reference)

```
Input image (any size)
    ↓
PE-L+ Image Encoder (~307M params) — FROZEN ALWAYS
    Resizes internally to 1008×1008
    Outputs dense feature map
    ↓
Text class names (from cls_*.txt)
    ↓
CLIP-style text encoder — FROZEN
    Encodes each class line as embedding vector
    ↓
SAM3 Mask Decoder (lightweight, cross-attention)
    Attends image features to class embeddings
    Outputs per-class masks
    ↓
Presence Head (per-class confidence 0–1)
    Suppresses false positives (classes not in image)
    ↓
Final segmentation output
```

Key difference vs SegEarth-OV (v1): OV-1 uses CLIP ViT-B/16 + SimFeatUp. OV-3 uses SAM3 (PE-L+ encoder) which is much stronger for spatial structure. OV-3 gets 57.8% mIoU vs OV-1's 48.5% on Potsdam zero-shot.

---

## Related Project Context

`D:\VC\rg-geoprompt-peft` — GeoPrompt/DINOv2 supervised baseline.
- Trained on Potsdam: **84.9% mIoU** (supervised upper bound)
- Zero-shot on Darmstadt: **0.2059 mean F1** (weak — confirms domain gap for supervised methods)
- SegEarth-OV-3 aim: beat 0.2059 F1 on Darmstadt zero-shot; get competitive Potsdam mIoU ~57%
- Teammate's qualitative: OV-3 (7.9/10) > GeoPrompt (6.7/10) on Darmstadt already
