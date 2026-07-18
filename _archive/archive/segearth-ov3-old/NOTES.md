# SegEarth-OV-3 — Architecture Notes

Source: https://github.com/dummy-irl/SegEarth-OV-3  
Kaggle notebook: `segearth03-bfd0b0.ipynb`

---

## What it is

SegEarth-OV-3 is an **open-vocabulary segmentation** pipeline for remote sensing imagery. It replaces class-specific decoders with natural language prompts — you describe classes in text and the model segments them without retraining. The backbone is **SAM3** (Segment Anything Model 3), a vision-language foundation model.

---

## Core components

### SAM3 (`sam3/`)
- The foundation model — a vision-language model that takes an image + text prompt and returns binary instance masks plus a semantic mask.
- Built with `build_sam3_image_model(bpe_path, checkpoint_path, device)`.
- Wrapped in a `Sam3Processor` that exposes `set_image()`, `set_text_prompt()`, `reset_all_prompts()`.
- Three outputs per query: `masks_logits` (instance-level), `semantic_mask_logits` (holistic), `presence_score` (confidence the class exists), `object_score` (per-instance confidence).

### `segearthov3_segmentor.py` — the main model class
Wraps SAM3 inside an MMSegmentation `BaseSegmentor`.

**Inference flow for one image:**
```
image (PIL RGB)
  └─► Sam3Processor.set_image()          # encode image once
  └─► for each class text prompt:
        set_text_prompt(prompt)
        → masks_logits [N_inst, H, W]     # transformer decoder heads
        → semantic_mask_logits [H, W]     # dense semantic head
        → presence_score (scalar)         # does class exist?
        → object_score [N_inst] (scalar)  # per instance confidence
        combine: seg_logits[query] = max(instance_logit * object_score, semantic_logit)
                                   * presence_score
  └─► argmax over queries → seg_pred [H, W]
  └─► threshold: pixels with max_logit < prob_thd → bg_idx
```

**Three toggle flags** (all default True):
- `use_transformer_decoder` — include instance-level masks
- `use_sem_seg` — include semantic mask
- `use_presence_score` — scale by presence confidence

**Sliding window** (`slide_inference`): activates when `slide_crop < image.width or slide_crop < image.height`. Crops the image, infers each crop, accumulates logits with count averaging. Good for large tiles.

**Multi-label handling**: class list file can have multiple names per line (comma-separated synonyms) that all map to the same class index. `get_cls_idx()` handles this.

### `demo.py` — inference runner
- Reads images from `INPUT_FOLDER` (from `config_local.py`)
- Optional single-image mode (`RUN_SINGLE_IMAGE`, `TARGET_IMAGE`)
- Multi-GPU sharding via env vars `TOTAL_GPUS` / `LOCAL_GPU_ID`
- Current prompts (10-class): building, road, construction area, parking lot, grassland, car, tree, water, container, clutter
- Saves PNG visualisations (RGB | overlay) to `output/`
- Has Potsdam GT loading path (`_RGB` suffix + `_label_noBoundary.tif`) — visual only, no metrics

### `config_local.py` — **in the repo**
Imported with `from config_local import *` in every file. Handles Kaggle vs local via `KAGGLE_KERNEL_RUN_TYPE` env var:

```python
KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

SAM3_CHECKPOINT = "/kaggle/input/datasets/dummyirl/sam3-weights/sam3.pt"  # Kaggle
                  "weights/sam3/sam3.pt"                                   # local

INPUT_FOLDER    = "/kaggle/input/datasets/dummyirl/frankfurt-dot20/"  # Kaggle
                  "resources/"                                          # local

RUN_SINGLE_IMAGE = True
TARGET_IMAGE     = "dop20_32_484_5550_1_he.jpg"   # Frankfurt DOP20 sample
```

**Current target dataset: Frankfurt DOP20** (not Potsdam). The team is running open-vocab inference on Frankfurt aerial imagery.

### `eval.py`
Uses MMSeg `Runner` for formal evaluation. Records per-dataset metrics (aAcc, mIoU, mAcc) to an Excel file. **Not called in the Kaggle notebook** — it exists but the team's workflow is currently demo.py only (visual outputs, no numbers).

### `configs/cfg_potsdam.py` + `configs/cls_potsdam.txt`
A **complete Potsdam eval config already exists** in the repo. Key settings:
- Prompts (6-class): `road, building, grass, tree, car, clutter`
- `slide_stride=448, slide_crop=512` (tighter than demo.py's 768/1024 — right for 512px patches)
- `bg_idx=5` (clutter)
- `data_root='/kaggle/working/PotsdamEval'` with structure `img_dir/val` + `ann_dir_indexed/val`

This config is designed to run via `eval.py`, not `demo.py`. The Potsdam dataset would need to be staged at that path on Kaggle. **The infrastructure to get quantitative Potsdam numbers is all there — it just hasn't been wired into the notebook.**

Note: prompt "road" for ISPRS class 0 (impervious surface) is narrower than the actual class — impervious includes pavements, parking lots, etc. May underperform on non-road impervious areas.

### `pamr.py`
Pixel-Adaptive Mask Refinement — a local affinity refinement module from TU Darmstadt (Apache 2.0). Iteratively propagates mask predictions using local pixel similarity. **Not currently used** in the segmentor — available but dormant.

### `custom_datasets.py`
25 MMSeg dataset class registrations. Includes OpenEarthMap, GID5, UAVid, WHU, and others. Relevant here: dataset infra if the team switches from demo.py to eval.py.

### `custom_transforms.py`
`LoadCDImagesFromFile` — loads image pairs for change detection. Not relevant for segmentation.

---

## Kaggle notebook setup (`segearth03`)

The env setup is non-trivial because MMSeg/MMCV don't work in Kaggle's default Python:

1. Download + install Miniconda to `/kaggle/working/miniconda`
2. `conda create -n segearth python=3.10`
3. `conda run -n segearth pip install torch==2.4.0 torchvision==0.19.0`
4. `pip install openmim` → `mim install mmcv==2.2.0`
5. `pip install mmsegmentation==1.2.2`
6. **Manual patch**: change `MMCV_MAX = '2.2.0'` → `'2.3.0'` in mmseg's `__init__.py`
7. `pip install numpy==1.26.4`
8. Clone `dummy-irl/SegEarth-OV-3`
9. `pip install -r requirements.txt`
10. Run `demo.py`

---

## Data flow summary

```
Input: PIL RGB image (any size)
  ↓  slide_inference if large
SAM3 encoder (image once per slide crop)
  ↓  per class text prompt
SAM3 decoder → instance masks + semantic mask + scores
  ↓  combine with max + presence scaling
seg_logits [num_classes, H, W]
  ↓  argmax + threshold
seg_pred [H, W]   (class index map)
  ↓  COLOR_MAP lookup
RGB overlay PNG saved to output/
```
