# DEVLOG — rg-segearth-ov3
> Chronological narrative of everything we tried, broke, and fixed.
> Written so future-you can understand the reasoning, not just the outcome.

---

## Session 1 — Project setup + first inference attempt

### What we had
- SegEarth-OV-3 fork at `HarishDeepak/SegEarth-OV-3`
- Potsdam ISPRS tiles on Kaggle (`dummyirl/6isprs`)
- Darmstadt DOP20 presliced patches on Kaggle (`harish77718/darmstadt-dop20-presliced`)
- SAM3 checkpoint on Kaggle (`dummyirl/sam3-weights`)

### What we built
- NB01: verified Potsdam label format (RGB labels, not indexed)
- NB02: first attempt at Potsdam eval
- NB03: Hessen visual inference
- NB04: live demo notebook
- Wrote `cls_potsdam.txt` and `cls_hessen.txt` with verbose multi-synonym prompts

### Early prompt style (later simplified)
We initially wrote prompts like:
```
nadir aerial view of impervious surface, road, pavement, tarmac, asphalt
nadir aerial view of building, rooftop, structure, house, warehouse
```
The idea: more synonyms = better coverage. Turned out to be wrong (see Session 3).

---

## Session 2 — PTSAM soft prompt training (NB05)

### The idea
SAM3 is frozen. But its text encoder takes `[language_features, language_mask]` as input.
We can append 8 extra learnable vectors to `language_features` and train only those.
That's 8 × 1 × 256 = 2048 parameters — tiny. Everything else stays frozen.

### How injection works at eval time
```
language_features [32, 1, 256]   ← SAM3's text encoding of the query word
soft_prompts      [ 8, 1, 256]   ← our trained vectors
cat → [40, 1, 256]               ← what the grounding head sees

language_mask [1, 32]            ← original (True = masked/ignored)
zeros         [1,  8]            ← our tokens are valid, NOT masked
cat → [1, 40]
```

### Training result
- Expected: 12–22 hours (our initial estimate was wrong)
- Actual: ~3 hours
- Why so fast: backbone is completely frozen, only 2048 params update per step
- 50 epochs completed, `soft_prompts.pt` saved to `harish77718/ptsam-soft-prompts`

### First eval result: 7.07% mIoU
Both zero-shot AND PTSAM gave 7.07%. The predictions were pure noise — random colors everywhere, nothing resembling buildings or roads.

This told us the problem wasn't with the soft prompts. The **inference pipeline itself was broken**.

---

## Session 3 — Finding and fixing the broken inference pipeline

### What was broken (NB05b rewrites v1–v6)

**Root cause 1 — No sigmoid on semantic_seg**
The correct inference path is through `Sam3Processor`:
```python
processor.set_image(crop_pil)           # backbone forward
processor._forward_grounding(state)     # grounding head
# internally does: semantic_mask_logits = outputs["semantic_seg"].sigmoid()
```
Our old code called `model.forward_grounding(...)` directly, bypassing `Sam3Processor`.
Result: raw unbounded logits (range -∞ to +∞) instead of [0,1] probabilities.
Every class scored the same garbage values → noise predictions.

**Root cause 2 — Missing instance mask contribution**
The correct scoring formula (from `segearthov3_segmentor._inference_single_view`):
```
score = max(inst_logit * object_score, semantic_mask_logits) * presence_score
```
Old code only used `semantic_mask_logits`, completely ignored instance masks.

**Root cause 3 — Single word per class instead of per synonym**
We were making one grounding call per class. Should be one per synonym, then take max.

**Root cause 4 — `dummy_geom` mutation bug**
We called `model._get_dummy_prompt()` once and reused it across 6 class iterations.
`_forward_grounding` mutates the geometric prompt in-place → class 2+ got corrupted geometry.
Fix: call `_get_dummy_prompt()` fresh inside each per-class loop iteration.

### The fix: complete rewrite using Sam3Processor correctly
```python
state = processor.set_image(crop_pil)           # backbone once per crop
for word, cls_idx in zip(query_words, cls_indices):
    processor.reset_all_prompts(state)
    state["backbone_out"]["language_features"] = text_embed.to(DEVICE)
    state["backbone_out"]["language_mask"]     = text_mask.to(DEVICE)
    state["geometric_prompt"] = model._get_dummy_prompt()  # fresh each time
    processor._forward_grounding(state)
    # now state["semantic_mask_logits"] is correctly sigmoid'd
    scores = max(inst_logit * object_score, semantic_mask_logits) * presence_score
```

### v8 result: 54.6% zero-shot mIoU
Pipeline now works. Gap from literature (57.8%) is ~3.2% — explainable by our prompt differences.

---

## Session 4 — PTSAM dtype crash (v7 failure)

### The error
```
TypeError: ~ (operator.invert) is only implemented on integer and Boolean-type tensors
```
at `model_misc.py:59`: `is_valid = (~prompt_mask).float()`

### Root cause 1 — `.float()` on language_mask
We were caching text embeddings as:
```python
{k: v.float().cpu() for k, v in te.items()}
```
`language_mask` is `bool` dtype. `.float()` converts it to `float32`.
Then `~float32_tensor` crashes — Python's `~` operator only works on bool/int.

Fix: `{k: v.cpu() for k, v in te.items()}` — preserve original dtype.

### Root cause 2 — `torch.ones` for soft prompt mask
```python
soft_m = torch.ones(1, 8, ...)   # WRONG — marks soft tokens as masked/ignored
```
`language_mask`: `True` = masked/ignored, `False`/`0` = valid.
Training used `zeros` (valid). Eval used `ones` (ignored) — soft prompts were being thrown away.

Fix: `torch.zeros(..., dtype=te["language_mask"].dtype)`

Also had to cast soft_prompts to match language_features dtype:
```python
soft = soft_prompts.to(dtype=lang_f.dtype)
```

---

## Session 5 — Prompt simplification (teammate feedback)

### What the teammate said
> "For tree is because of your prompt. I think you should use more simple not too broad prompt.
> Like if you read the cls_dataset.txt file you can see the way how the owner do the prompt"

### What changed

**Before (too verbose):**
```
nadir aerial view of tree, forest canopy, shrub, woodland, meadow
```

**After (simple, like original repo):**
```
impervious surface, road
building, rooftop
low vegetation, grass
tree
car, vehicle
```

And for Hessen:
```
impervious surface, road, pavement
building, rooftop
tree, forest
car, vehicle
low vegetation, grass
railway track
```

### Why verbose prompts hurt
SAM3's text encoder is CLIP-based. CLIP was trained on short image captions, not long descriptions.
Long prompts dilute the embedding — the class vector becomes an average of too many concepts.
"nadir aerial view of tree, forest canopy, shrub, woodland, meadow" → fuzzy embedding that matches many things weakly.
"tree" → sharp, specific embedding that matches tree strongly.

### Also fixed: clutter scatter noise
Querying "clutter, background" caused scattered red noise at every object edge because
shadow/ambiguous regions scored high for "background."
Fix: remove clutter from queries entirely. Clutter only appears as the `prob_thd` fallback
(pixels where no class scores above 0.1 get assigned BG_IDX=5).

---

## Session 6 — Seam artifacts + proper mIoU protocol

### Boxy patch seams
With `STRIDE=1024` = `CROP_SIZE=1024`, every pixel comes from exactly one crop.
The Gaussian kernel we apply has nothing to blend against → seam artifacts at every crop border.
Fix: `STRIDE=512` → 50% overlap → Gaussian blending now actually works.
Crop count: 36 → 121 for a 6000×6000 tile. Worth it for clean boundaries.

### mIoU protocol fix
We were computing mIoU over all 6 classes including clutter.
Literature 57.8% is computed over 5 classes (impervious, building, low_veg, tree, car) — clutter excluded.
Fix: `EVAL_CLASSES = [0, 1, 2, 3, 4]` only.

### Why roads don't have their own class
Potsdam ISPRS benchmark groups all paved non-building ground into "impervious surface."
Roads, parking lots, sidewalks, driveways all look identical from 5cm aerial imagery (same grey asphalt).
That's why `cls_potsdam.txt` uses `"impervious surface, road"` — road is a synonym for class 0, not its own class.

---

## Session 7 — NB06 ablation + PAMR

### The 4 methods (plain language)

**A — ZS-single**
One word per class. Closest to original SegEarth-OV-3 out of the box.
`"building"`, `"tree"`, `"car"` etc. Baseline.

**B — ZS-multi**
Multiple synonyms per class, take max score across them.
`"building"` OR `"rooftop"` → whichever matches better wins for each pixel.
More chances to activate SAM3's recognition.

**C — ZS-multi + PAMR**
Take B's soft logits, run PAMR (Pixel-Adaptive Mask Refinement):
- PAMR looks at the RGB image and finds where real edges are
- It propagates class scores along smooth regions and stops at edges
- Result: segmentation boundaries snap to actual object boundaries in the image
- 10 iterations, dilations [1,2,4], no training, ~0.3 GiB per 1024×1024 crop

**D — PTSAM + PAMR**
Same as C but text queries have 8 extra trained vectors appended.
These vectors were optimized to improve SAM3's grounding on Potsdam aerial imagery.
Cross-domain: using Potsdam-trained soft prompts on Darmstadt is an experiment —
might generalize (aerial imagery is similar) or might hurt (different classes, GSD).

### PAMR OOM bug
First attempt ran PAMR on the full stitched 6000×6000 tile after all crops.
`LocalStDev` layer allocates `[6 classes × batch × 48 channels × 6000 × 6000]` → 21 GiB.
T4 only has 14.56 GiB → crash.

Fix: PAMR inside the sliding window loop, per 1024×1024 crop → ~0.3 GiB.

---

## Session 8 — NB07 multi-tile + Darmstadt

### What NB07 tests
**Potsdam:**
- Tiles 6_15 (official val with mIoU), 5_14, 7_13 (visual + mIoU for reference)
- All 4 methods on each
- STRIDE=1024 (36 crops, fast) for cross-tile comparison

**Darmstadt:**
- 5 randomly sampled 256×256 patches from `darmstadt-dop20-presliced`
- 1 full reconstructed tile: parsed `y{Y}_x{X}` from filenames, stitched all patches of one scene, ran sliding window
- Uses `cls_hessen.txt` (6 classes including railway track)
- All 4 methods — PTSAM is cross-domain here (trained on Potsdam classes)

### OSM+LLM idea (deferred)
We discussed per-tile prompt generation using OSM metadata + LLM.
Decision: **don't add more synonyms** (teammate's feedback), instead use OSM for **class presence gating**:
- Query OSM bbox for each tile
- If OSM says "no railway in this area" → skip "railway track" query entirely
- Short prompts stay short, just fewer of them
- Implement as Method E after seeing NB06/07 results

---

## What's still pending

1. **NB05b v9 result** — clutter removed from queries, 5-class mIoU, stride=512. Should be cleaner visually and higher mIoU number.
2. **NB06 v2 result** — ablation table showing A vs B vs C vs D. Will tell us whether PAMR or synonyms contribute more.
3. **NB07 v1 result** — 5 outputs: 3 Potsdam tile comparisons + Darmstadt patches + Darmstadt full tile.
4. **Retrain PTSAM** — soft prompts were trained with "clutter, background" in the vocab. Now clutter is removed. Retrain with 5-class `cls_potsdam.txt` to see if the +0.3% delta improves.
5. **OSM class-gating (Method E)** — implement after seeing NB06/07 to know if it's worth it.
6. **Darmstadt F1** — rasterize OSM road/building vectors as pseudo-GT, compute F1 per class.

---

## Numbers to remember

| Thing | Number |
|---|---|
| Literature ZS mIoU (Potsdam) | 57.8% |
| Our ZS mIoU before fix | 7.07% |
| Our ZS mIoU after fix (v8) | 54.6% |
| Our PTSAM mIoU (v8) | 54.9% |
| PTSAM parameters trained | 2048 |
| PTSAM training time | ~3 hours |
| T4 VRAM | 14.56 GiB |
| PAMR on full tile needed | 21 GiB → OOM |
| PAMR per crop needs | ~0.3 GiB → fine |
| Crops at stride=1024 (6000×6000 tile) | 36 |
| Crops at stride=512 (6000×6000 tile) | 121 |
| Teammate Darmstadt score | 7.9/10 |
| Our GeoPrompt Darmstadt score | 6.7/10 |
| GeoPrompt Potsdam supervised mIoU | 84.9% |
