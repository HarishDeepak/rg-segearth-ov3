# Project Progress Log — rg-segearth-ov3
> Last updated: 2026-06-28
> Fraunhofer IGD Praktikum SoSe 2026 — Open-vocab RS segmentation

---

## What this project is

Zero-shot aerial image segmentation using **SegEarth-OV-3** (SAM3 + open-vocab text prompts).
- **SAM3 is fully frozen** — no backprop through it, ever
- Improvements come from: better prompts, sliding window overlap, PAMR post-processing, PTSAM soft prompt tuning
- Two datasets: **Potsdam ISPRS** (5cm GSD, quantitative) and **Darmstadt DOP20** (20cm GSD, qualitative)
- Comparison baseline: GeoPrompt/DINOv2 (in `rg-geoprompt-peft` repo), which got 84.9% mIoU on Potsdam supervised, 6.7/10 qualitative on Darmstadt

---

## Results so far

### Potsdam val tile 6_15 (5-class mIoU, excl. clutter — matches literature protocol)

| Method | mIoU | Notes |
|---|---|---|
| Literature ZS (SegEarth-OV-3) | 57.8% | Paper number |
| Our ZS v1 (broken) | 7.07% | Sigmoid missing, wrong scoring |
| Our ZS v8 (fixed) | **54.6%** | Sam3Processor used correctly |
| Our PTSAM v8 | **54.9%** | +0.3% over ZS — marginal |
| Our ZS v9 (clutter removed, 5-cls) | TBD | NB05b still running |

### Darmstadt (qualitative, no GT)
- Teammate's SegEarth-OV-3 (single-word prompts): **7.9/10**
- Our GeoPrompt: 6.7/10 (weaker on cars)
- Our runs: visual comparison pending NB07

---

## Notebooks built

| NB | Kaggle kernel | Purpose | Status |
|---|---|---|---|
| NB01 | nb01-verify-data | Verify Potsdam label format (indexed vs RGB) | Done |
| NB02 | nb02-potsdam-eval | Original Potsdam eval (old broken pipeline) | Superseded by NB05b |
| NB03 | nb03-hessen-infer | Hessen visual inference | Done |
| NB04 | nb04-demo | Live demo | Done |
| NB05 | nb05-ptsam-train / nb05v2 | PTSAM soft prompt training | **Done** — 50 epochs, ~3 hrs |
| NB05b | nb05b-ptsam-eval | Zero-shot vs PTSAM comparison on Potsdam val | Active — v9 running |
| NB06 | nb06-method-ablation | 4-method ablation: ZS-single / ZS-multi / +PAMR / PTSAM+PAMR | Active — v2 running |
| NB07 | nb07-multi-tile-darmstadt | 3 Potsdam tiles + 5 random Darmstadt patches + full Darmstadt tile | Active — v1 running |

---

## PTSAM training (NB05)

- Trained soft prompts: shape `[8, 1, 256]` — appended to language features per text query
- Only 2048 parameters trained (everything else frozen)
- 50 epochs completed in ~3 hours (faster than expected because backbone is frozen)
- Saved to Kaggle dataset: `harish77718/ptsam-soft-prompts` as `soft_prompts.pt`
- How it works at eval: `language_features [32,1,256]` + `soft_prompts [8,1,256]` → `[40,1,256]`
- `language_mask` gets `zeros [1,8]` appended (zeros = valid/unmasked tokens)

---

## Key bugs found and fixed

### Bug 1 — Missing sigmoid on semantic_seg (root cause of 7.07% mIoU)
- `Sam3Processor._forward_grounding` stores `outputs["semantic_seg"].sigmoid()` as `semantic_mask_logits`
- Old code called raw `forward_grounding` directly, bypassing `Sam3Processor` entirely — no sigmoid, unbounded logits
- Fix: rewrite to use `Sam3Processor.set_image()` + `processor._forward_grounding(state)`

### Bug 2 — Missing instance mask contribution
- Old code only used `semantic_mask_logits`, ignored `masks_logits * object_score`
- Correct formula (from `segearthov3_segmentor._inference_single_view`):
  `max(inst_logit * object_score, semantic_mask_logits) * presence_score`

### Bug 3 — `language_mask` dtype crash (`~` operator on float)
- `{k: v.float().cpu()}` converted `language_mask` from bool → float32
- `model_misc.py:59`: `is_valid = (~prompt_mask).float()` — `~float` is not supported
- Fix: `{k: v.cpu()}` — preserve original dtype

### Bug 4 — `soft_m = ones` (PTSAM tokens marked as masked/ignored)
- `torch.ones(...)` for the PTSAM mask extension meant soft prompt tokens were masked out
- Training used `zeros` (valid/unmasked); eval must match
- Fix: `torch.zeros(..., dtype=te["language_mask"].dtype)`

### Bug 5 — `dummy_geom` mutation
- `model._get_dummy_prompt()` called once, reused across all class iterations
- `_forward_grounding` mutates the geometric prompt in-place
- Fix: call `model._get_dummy_prompt()` fresh inside each per-class loop iteration

### Bug 6 — PAMR OOM on full tile
- PAMR on 6000×6000 tile needs 21 GiB; T4 has 14.56 GiB
- Fix: run PAMR per-crop (1024×1024) inside the sliding window loop → ~0.3 GiB per call

---

## Prompt engineering decisions

### cls_potsdam.txt (current)
```
impervious surface, road
building, rooftop
low vegetation, grass
tree
car, vehicle
```
- Clutter removed from queries — only appears as `prob_thd` fallback
- Reasoning: "clutter, background" query caused scattered red noise at object edges
- Teammate feedback: simple short prompts beat verbose ones (original repo style)

### cls_hessen.txt (current)
```
impervious surface, road, pavement
building, rooftop
tree, forest
car, vehicle
low vegetation, grass
railway track
```
- Simplified from verbose "nadir aerial view of..." phrases
- Teammate tested simple prompts on Darmstadt → 7.9/10

### What we tried and reverted
- Multi-paragraph descriptions per class → worse (CLIP text encoder prefers short noun phrases)
- "canopy", "shrub", "meadow" as tree synonyms → caused over-prediction of tree class

---

## Eval methodology

- **Potsdam mIoU**: 5-class average (impervious, building, low_veg, tree, car) — **excludes clutter** — matches literature 57.8% protocol
- **Val tile**: always `6_15`, never random split (patch overlap → leakage)
- **GT file**: `*_label_noBoundary.tif` — boundary pixels = 255 = IGNORE_IDX, excluded from metrics
- **Darmstadt**: visual only, no GT (OSM pseudo-GT planned but not implemented yet)

---

## Sliding window parameters

| Param | Potsdam (NB05b) | Potsdam (NB06/07) | Hessen |
|---|---|---|---|
| `slide_crop` | 1024 | 1024 | 256 (patches pre-sliced) |
| `slide_stride` | 512 | 1024 | 256 |
| crops per tile | ~121 | 36 | 1 per patch |
| Gaussian blend | yes | yes | N/A |
| PAMR | per-crop | per-crop | per-crop |

**Tuning guide:**
- `confidence_threshold` (0.1): min `presence_score` inside Sam3Processor. Raise to suppress false positives.
- `prob_thd` (0.1): pixels with winning score < this → `BG_IDX`. Raise to suppress noisy low-conf pixels.
- `stride < crop`: enables Gaussian blending, removes seam artifacts. Use `stride = crop/2` as default.

---

## Methods in ablation (NB06/NB07)

| Method | Description |
|---|---|
| **A ZS-single** | One word per class. Baseline — closest to original repo |
| **B ZS-multi** | Multi-synonym `cls_potsdam.txt`, max over synonyms per class |
| **C ZS-multi + PAMR** | B + PAMR boundary refinement (image-guided edge sharpening, 10 iter) |
| **D PTSAM + PAMR** | C + soft prompts injected into language features |

**Reading the ablation:**
- B vs A → value of synonym prompts
- C vs B → value of PAMR boundary refinement
- D vs C → value of soft prompt tuning

---

## Pending work

- [ ] **NB05b v9 results** — check mIoU after removing clutter query + 5-class protocol
- [ ] **NB06 v2 results** — ablation table: how much does each method contribute?
- [ ] **NB07 v1 results** — multi-tile Potsdam + Darmstadt visual comparison
- [ ] **OSM class-gating (Method E)** — use OSM bbox query to suppress irrelevant class queries per tile (e.g. skip "railway" if OSM has no railway in bbox). Short prompts stay short — just fewer of them. Implement after seeing NB06/07 results.
- [ ] **Retrain PTSAM** — soft prompts were trained with clutter in vocab. Now that clutter is removed, retrain for 50 epochs to see if PTSAM delta improves beyond +0.3%.
- [ ] **Darmstadt F1 vs OSM** — rasterize OSM pseudo-GT → erode 3×3 (borders→255) → F1 per class

---

## Key files

| File | Purpose |
|---|---|
| `configs/cls_potsdam.txt` | Potsdam class prompts (5 classes, multi-synonym) |
| `configs/cls_hessen.txt` | Hessen class prompts (6 classes incl. railway) |
| `configs/cfg_potsdam.py` | Potsdam inference config |
| `pamr.py` | PAMR boundary refinement module |
| `src/segearth_utils/` | Thin utilities (constants, osm_eval) |
| `notebooks/push/nb05b/` | PTSAM eval notebook |
| `notebooks/push/nb06/` | 4-method ablation notebook |
| `notebooks/push/nb07/` | Multi-tile + Darmstadt notebook |

---

## Kaggle datasets

| Dataset | Slug | Mount path |
|---|---|---|
| SAM3 weights | `dummyirl/sam3-weights` | `/kaggle/input/sam3-weights/sam3.pt` |
| Potsdam ISPRS | `dummyirl/6isprs` | `/kaggle/input/datasets/dummyirl/6isprs/` |
| Darmstadt DOP20 | `harish77718/darmstadt-dop20-presliced` | `/kaggle/input/datasets/harish77718/darmstadt-dop20-presliced/darmstadt_dop20/images/` |
| PTSAM soft prompts | `harish77718/ptsam-soft-prompts` | `/kaggle/input/` (glob for `soft_prompts.pt`) |

---

## Related project

`D:\VC\rg-geoprompt-peft` — GeoPrompt/DINOv2 supervised training.
- Trained on Potsdam: **84.9% mIoU**
- Zero-shot on Darmstadt: mean F1 **0.2059** (weaker than SegEarth-OV-3 visually)
- SegEarth-OV-3 is the zero-shot comparison method for the Praktikum report
