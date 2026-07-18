# SegEarth-OV-3 Session Reference

_Full explanations preserved — not summarised_

---

## PART 1 — KAGGLE AUTH & T4 GPU FIXES

---

### Kaggle CLI 2.2.2 Authentication — Full Root Cause

**Problem:** `kaggle kernels push` kept returning "Authentication required to call the Kaggle API" even after placing `kaggle.json` with a valid key and even after running `kaggle auth login`.

**What we tried and why each failed:**

1. **Legacy API key in `kaggle.json`** — file was correct (`{"username":"harish77718","key":"..."}`, 32-char hex key). But CLI 2.2.2 uses the new `api.kaggle.com` endpoint. When it reads `kaggle.json`, `_authenticate_with_legacy_apikey()` returns True *locally* (just checks that username+key exist in config), but the actual HTTP call to `api.kaggle.com` returns 401 because that endpoint does not accept HTTP Basic auth with legacy keys.
2. **Writing key to `~/.kaggle/access_token`** — made things worse. `get_access_token_from_env()` in kagglesdk reads this file and sends its contents as a Bearer token (`Authorization: Bearer <key>`). A 32-char API key is not an OAuth Bearer token, so `api.kaggle.com` rejects it with 401. Meanwhile this path is checked BEFORE the legacy key path, so the legacy key never even gets tried properly.
3. **KAGGLE_USERNAME + KAGGLE_KEY env vars** — same result. The new `api.kaggle.com` endpoint just doesn't support legacy key auth at all regardless of how it's supplied.

**Why `kaggle auth login` (first attempt) showed "already logged in" but still failed:**
There was already a stale OAuth token cached in `~/.kaggle/credentials.json` from a previous login. The CLI saw this and said "already logged in" — but the token was expired. `kaggle auth login --force` re-ran the OAuth flow, got a fresh token, and auth worked.

**Why push still failed after OAuth login:**
`kaggle.json` still had the `key` field. `_authenticate_with_legacy_apikey()` returned True because username+key were present, and the code returned early — never reaching `_authenticate_with_oauth_creds()` which would have used the fresh OAuth credentials from `credentials.json`. The legacy key path "wins" and then fails at the server.

**Final fix:**

1. Run `kaggle auth login --force` → fresh OAuth token → saved to `~/.kaggle/credentials.json`
2. Delete or remove the `key` field from `~/.kaggle/kaggle.json` (keep only `username`) so `_authenticate_with_legacy_apikey()` returns False, and the code falls through to OAuth credentials

**Auth flow in CLI 2.2.2 source (confirmed by reading kaggle_api_extended.py):**

```python
def authenticate(self):
    self._load_config()  # reads kaggle.json
    if self._authenticate_with_access_token():  # reads access_token file as Bearer
        return
    if self._authenticate_with_legacy_apikey():  # checks if username+key in config
        return                                    # returns True even if key is revoked!
    if self._authenticate_with_oauth_creds():    # reads credentials.json
        return
    print_auth_help()
    exit(1)
```

**Key insight:** The legacy key path does NOT validate the key against the server — it just checks the key exists in the config file. Validation only happens when an actual API call is made, at which point the server returns 401.

**Permanent setup:** OAuth credentials in `~/.kaggle/credentials.json` are the only auth that works with CLI 2.2.2 on `api.kaggle.com`. Legacy key only works with the old `www.kaggle.com/api/v1` endpoints.

---

### T4 GPU Fix

**Problem:** Notebooks were getting P100 GPU despite `"enable_gpu": true` in `kernel-metadata.json`.

**Root cause:** `enable_gpu: true` alone defaults to P100. Kaggle's API does not infer GPU type from any other field. Must explicitly specify the accelerator.

**Fix:** Add `"machine_shape": "NvidiaTeslaT4"` to `kernel-metadata.json`:

```json
{
  "id": "harish77718/nb02-potsdam-eval",
  "title": "NB02 Potsdam Eval",
  "code_file": "NB02_potsdam_eval.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["dummyirl/sam3-weights", "dummyirl/6isprs"],
  "competition_sources": [],
  "kernel_sources": []
}
```

Known `machine_shape` values:

- `NvidiaTeslaT4` — T4 x2 (what we want)
- `NvidiaTeslaP100` — P100 (the default when only `enable_gpu: true`)

Applied to both `notebooks/push/nb02/kernel-metadata.json` and `notebooks/push/nb03/kernel-metadata.json`. Both pushed successfully.

---

## PART 2 — NB02 POTSDAM EVAL: THE LABEL INDEXING BUG

---

### Teammate's Results (for comparison baseline)

From `teammate/segearth03 (4).ipynb` — this ran on ALL 6 Potsdam tiles as validation (not a proper held-out split):

```
+--------------------+-------+-------+--------+-----------+--------+
|       Class        |  IoU  |  Acc  | Fscore | Precision | Recall |
+--------------------+-------+-------+--------+-----------+--------+
| impervious_surface | 75.97 | 84.82 | 86.34  |   87.92   | 84.82  |
|      building      | 86.13 | 92.44 
| 92.55  |   92.66   | 92.44  |
|   low_vegetation   | 64.92 | 92.27 | 78.73  |   68.66   | 92.27  |
|        tree        | 46.02 | 47.44 | 63.04  |   93.92   | 47.44  |
|        car         | 79.98 | 98.47 | 88.88  |   80.99   | 98.47  |
|      clutter       | 20.53 | 38.99 | 34.07  |   30.25   | 38.99  |
+--------------------+-------+-------+--------+-----------+--------+
mIoU: 62.26   aAcc: 81.20   mAcc: 75.74
```

**Important caveat:** They used all 6 tiles as val — the full dataset with no held-out split. Our NB02 uses only `6_15` (proper tile-level split, no patch overlap leakage). So our number will differ and is more honest.

**Their prompts:** Single class names (`building`, `road`, etc.) — no synonyms. Their labels were already in PNG format (no RGB→indexed conversion needed).

---

### Our NB02 Result BEFORE the fix: mIoU 3.69%

```
+--------------------+-------+
| impervious_surface |  0.19 |
|      building      |  0.04 |
|   low_vegetation   | 20.88 |  ← only class with any IoU
|        tree        |  0.0  |
|        car         |  1.01 |
|      clutter       |  0.0  |
+--------------------+-------+
mIoU: 3.69   aAcc: 10.51   mAcc: 8.56
```

This is catastrophically bad. The model IS detecting real structures spatially — but the evaluation is completely broken.

---

### Root Cause: The `reduce_zero_label=True` Trap

`PotsdamDataset` in mmseg is defined with `reduce_zero_label=True`. This means:

```
Ground truth label 0  →  255  (IGNORED — excluded from all metrics)
Ground truth label 1  →  0    (becomes "class 0" in evaluation)
Ground truth label 2  →  1    (becomes "class 1" in evaluation)
Ground truth label 3  →  2
Ground truth label 4  →  3
Ground truth label 5  →  4
```

Our NB02 label conversion cell used **0-indexed mapping (0–5)**:

```python
RGB_TO_IDX = {
    (255, 255, 255): 0,  # impervious surface  ← THIS MAPS TO 255 (IGNORED)
    (  0,   0, 255): 1,  # building            ← becomes GT class 0
    (  0, 255, 255): 2,  # low vegetation      ← becomes GT class 1
    (  0, 255,   0): 3,  # tree                ← becomes GT class 2
    (255, 255,   0): 4,  # car                 ← becomes GT class 3
    (255,   0,   0): 5,  # clutter             ← becomes GT class 4
}
```

**What this means after `reduce_zero_label` is applied:**

- **Impervious surface** (class 0, about 40% of all pixels in Potsdam) → **255 = IGNORED**. Gone. Not evaluated at all.
- **Building** (originally class 1) → becomes GT class 0 → gets compared against model's class 0 ("road" in our cls_potsdam.txt)
- **Low vegetation** (originally class 2) → becomes GT class 1 → compared against model's class 1 ("building")
- **Tree** (originally class 3) → becomes GT class 2 → compared against model's class 2 ("grass")
- **Car** (originally class 4) → becomes GT class 3 → compared against model's class 3 ("tree")
- **Clutter** (originally class 5) → becomes GT class 4 → compared against model's class 4 ("car")

And there IS no GT class 5 in this setup — so model class 5 ("clutter") never matches anything → 0 IoU always.

**Complete disaster:** ~40% of pixels are ignored, and the remaining 60% are evaluated against the WRONG model class at every position.

**Why low_vegetation got 20.88% IoU despite this:**
"Building" text prompt (what GT class 0 "building" maps to in model space after shift) and "low vegetation" (what GT class 1 maps to) might have some spatial overlap in ambiguous regions. Also "building" is a very common dominant class, and the model predicts it a lot, so some accidentally overlaps with whatever GT class ended up in position 1. Pure coincidence of scale.

---

### Why It LOOKED Good Visually Despite 3.69% mIoU

The visualization (NB02 cell 113c5d9f) shows:

- The model's prediction output, colored by its class palette
- The GT, colored by the PALETTE array

The model IS detecting real buildings and roads spatially — the shapes and boundaries are correct. The visualization doesn't compare against GT inline — it just shows each as a separate panel colored by class index. So:

- You see a panel that looks like a building-shaped prediction colored blue → looks correct
- You see another panel with the GT buildings colored blue → also looks correct
- But the MODEL thinks those blue regions are one class and the GT has them as a different class index

The visual looks fine because the model is doing the right thing geographically. The mIoU is catastrophic because the label indices are misaligned.

---

### The Fix: 1-Indexed Labels (1–6)

The `_label_noBoundary.tif` files use RGB colors. After converting, we need **1-indexed** labels (1–6) so that `reduce_zero_label=True` maps them correctly:

```
Our label 1 (impervious surface) → reduce → GT class 0 → model class 0 ✓
Our label 2 (building)           → reduce → GT class 1 → model class 1 ✓
Our label 3 (low vegetation)     → reduce → GT class 2 → model class 2 ✓
Our label 4 (tree)               → reduce → GT class 3 → model class 3 ✓
Our label 5 (car)                → reduce → GT class 4 → model class 4 ✓
Our label 6 (clutter)            → reduce → GT class 5 → model class 5 ✓
Our label 0 (boundary/none)      → reduce → 255 (IGNORED) — _noBoundary files have NONE of these
```

**Fixed conversion cell (NB02, cell `adb25d60`):**

```python
# 1-indexed: PotsdamDataset uses reduce_zero_label=True which maps 0→255(ignored),
# 1→0, 2→1, ..., 6→5. _noBoundary labels have no class-0 boundary pixels.
RGB_TO_IDX = {
    (255, 255, 255): 1,  # impervious surface
    (  0,   0, 255): 2,  # building
    (  0, 255, 255): 3,  # low vegetation
    (  0, 255,   0): 4,  # tree
    (255, 255,   0): 5,  # car
    (255,   0,   0): 6,  # clutter
}
```

Added safety check: if the label already has correct 1-indexed values (e.g. re-running the cell), skip conversion instead of double-shifting.

**Pushed as NB02 version 16.**

---

### Also Fixed: cls_potsdam.txt Prompts

**Old (wrong/too narrow):**

```
road          ← impervious surface includes parking lots, paths, not just roads
building
grass         ← low vegetation includes shrubs, not just grass
tree
car
clutter
```

**New (better aerial alignment):**

```
impervious surface, road, pavement, paved ground
building, rooftop, structure
low vegetation, grass, shrub, lawn, meadow
tree, forest, canopy
car, vehicle
clutter, background
```

The first class mismatch ("road" vs "impervious surface") was causing the model to miss large areas of pavement and parking. "Grass" was too narrow for the low_vegetation class which includes all sub-1m vegetation.

---

## PART 3 — NB03 HESSEN INFERENCE STATUS

---

NB03 does **not** have the label indexing bug — it is inference-only with no ground truth comparison. `reduce_zero_label` is irrelevant. It just runs SAM3 and saves colored prediction PNGs.

**Current config:**

- `STITCH_REGION = True` → runs all ~289 non-overlapping patches from the 17×17 grid
- `N_SAMPLE = 100` → ignored when `STITCH_REGION=True`
- `slide_crop=512`, patch size = 256×256 PNG inputs
- 7 urban classes matching teammate vocabulary: impervious surface, building, tree canopy, vehicle, vegetation, railway, road

**Estimated time on T4:** ~15 min env setup + ~15 min inference + stitch = ~35 min total

**Per-patch log format:** `[1/289] dop20_32_474_5532_1_he_y0_x0.png → 3.2s (ETA 15.1 min)`

User confirmed visual improvements visible in NB03 output.

---

## PART 4 — IMPROVEMENT STRATEGIES: FULL RESEARCH

---

### What the Archive Already Established (lastclaudechat.md)

From a prior session, detailed research was done. Key conclusions:

**On fine-tuning SegEarth-OV-3 (SAM3):**
PE-L+ encoder alone is ~307M params and needs ~12–16GB+ VRAM just for inference at 1008×1008. Fine-tuning it directly would need a full redesign — there's no training code, no adapter path in the SegEarth-OV-3 repo. Not feasible without major engineering.

**On using SegEarth-OV-1 (CLIP-based) instead:**
OV-1 uses CLIP ViT-B/16 + SimFeatUp upsampler + cosine similarity. This IS fine-tunable by injecting LoRA into CLIP's attention layers. The resolution bridge augmentation (your GeoPrompt `augment.py`, 30% probability downsample ×4) is directly portable. But:

**The teammate's result changes the calculus:**
Teammate got 7.9/10 visual quality on Darmstadt using OV-3 with 12 single-name classes. Our GeoPrompt supervised method got 6.7/10. This means training-free OV-3 already beats a supervised method on the visual quality metric. The highest-leverage path is therefore **better prompts**, not PEFT.

**5 tuning knobs already identified for OV-3 (all training-free):**

1. Text aliases — biggest lever, teammate used single names only
2. `prob_thd` sweep (default 0.0, try 0.2, 0.4, 0.5)
3. `use_presence_score=False` for rare classes (car, water, clutter)
4. Semantic vs instance head fusion toggle
5. Sliding window stride reduction (50% overlap = 4× compute but cleaner seams)

---

### Region-Adaptive Magnification (SOPSeg) — SKIP THIS

**What it is:** Pre-upsample images from 256×256 → 1008×1008 before feeding to SAM3, to simulate higher-resolution input.

**Why it sounds appealing:** The SOPSeg paper shows that upsampling small objects before inference helps SAM3 engage with them properly since SAM3's attention operates at a spatial scale expecting full-resolution inputs.

**Why we should skip it for our case:**
SAM3 already upsamples internally to 1008×1008 regardless of input size — it does this before the backbone processes the image. So if we pre-upsample 256→1008 externally, we're doing redundant bilinear interpolation before SAM3's own internal resize. The result:

- Creates a regular bilinear interpolation grid — SAM3's attention sees fake periodic texture patterns, not real pixels
- Large uniform areas (crop fields, roads) get blurry halos at upsampled edges that the model misinterprets as class boundaries
- No extra information is created — bilinear cannot invent missing 20cm GSD detail

The SOPSeg paper's gain comes from cases where a model crops a small sub-region from a large image and loses small objects in the crop. Our patches are already the full 256×256 scene (not sub-crops), so that specific problem doesn't apply.

**Bottom line: skip this. SAM3 does it internally already. Risk > reward.**

---

### Viewpoint Prompts — HIGHEST GAIN, LOWEST EFFORT

**The problem:** CLIP's text encoder (which SAM3 uses) was trained on internet-scale data which is overwhelmingly ground-level photography. When you type "building", the text embedding activates latent features associated with facades, windows, doors, brick texture — things visible from street level. In a top-down aerial image, NONE of these features exist. You see rooftops, shadows, footprints.

**How viewpoint descriptors fix this:**
Adding "nadir aerial view of" or "overhead top-down perspective of" to the prompt shifts the text embedding toward the aerial image sub-space of CLIP's joint latent space. The cross-attention in SAM3 then looks for geometric footprints and roof topologies rather than facades.

**Confirmed in LAE-80C study (arxiv 2601.22164):**
Evaluated 80-class open-vocabulary detection on aerial data. Adding explicit spatial and viewpoint descriptors to class names gave the biggest zero-shot improvement — exceeding synonym expansion alone.

**What to change in cls_hessen.txt:**

Current:

```
impervious surface, road, pavement, paved ground
building, rooftop, structure
...
```

With viewpoint descriptors:

```
nadir aerial view of road, paved surface, impervious ground, asphalt
overhead view of building rooftop, flat roof, residential roof casting shadow
aerial top-down view of tree canopy, forest patch, treetop
overhead low-resolution view of vehicle, parked car, automobile
nadir view of low vegetation, grassland, lawn, green surface
aerial view of railway track, rail line
overhead view of paved road, urban street
```

Also: adding "low-resolution" or "20cm GSD" or "blurred, low-contrast" to the prompt description acknowledges the resolution degradation. CLIP has seen varying quality imagery during web-scale pre-training, so its latent space does encode resolution/blur concepts. Telling it to expect blurry tokens reduces the penalty for missing sharp internal textures.

**LLM-Driven Prompt Expansion (TMPA approach, CVPR 2026):**
Instead of manually writing descriptors, use an LLM with a task-specific system prompt:

```
System: You are an expert in remote sensing aerial imagery at 20cm ground sample 
distance (20cm GSD). Generate 3 distinct single-sentence visual descriptions for 
each class as seen from directly above (nadir view), emphasizing shape, 
roof/ground texture, shadow direction, and colour as they appear in blurry 
low-resolution overhead imagery. Do not mention ground-level features.

User: Class: building
→ "rectangular artificial structure with flat or gabled roof casting directional 
   shadow on adjacent surfaces in overhead imagery"
→ "dense cluster of impervious geometry with uniform roof texture distinct from 
   surrounding natural vegetation"
→ "man-made block structure with sharp rectilinear footprint and low spectral 
   variance in satellite top-down view"
```

Average these 3 text embeddings → much broader, more robust semantic manifold. The model now has multiple angles of attack for the same class.

---

### PTSAM — What It Is, In Full Detail

**Paper:** "Prompt-Tuning SAM: From Generalist to Specialist with only 2,048 Parameters and 16 Training Images" — CVPRW 2025

**What SAM3's architecture looks like:**

```
Input image
    ↓
┌─────────────────────────────────┐
│  PE-L+ Image Encoder (~307M)    │  ← MASSIVE. Stays 100% frozen always.
│  (Perception Encoder Large+)    │  Processes 1008×1008 → dense feature map
└────────────────┬────────────────┘
                 │ image features
┌────────────────▼────────────────┐
│  Prompt Encoder                 │  ← encodes text class names as embeddings
│  (text → embeddings)            │  Also stays frozen.
└────────────────┬────────────────┘
                 │ prompt embeddings
┌────────────────▼────────────────┐
│  Mask Decoder                   │  ← LIGHTWEIGHT. Cross-attention between
│  (cross-attention layers)       │    image features and prompt embeddings.
│                                 │    THIS is where PTSAM adds soft tokens.
└─────────────────────────────────┘
                 │
           segmentation mask
```

**What "soft prompt tokens" are:**
They are NOT text. They are learnable floating-point vectors (same dimensionality as the prompt embeddings) that get concatenated with the real text prompt embeddings before the mask decoder processes them. They have no human-readable meaning — they're just numbers that gradient descent adjusts.

Think of it like this: you're adding a few extra "hint" vectors alongside the text description. During training on Potsdam, those hint vectors learn to encode "how aerial imagery at 20cm GSD looks when you decode it into masks" — knowledge that the original SAM3 didn't have because it was trained on natural images.

**Why decoder-only tuning:**

- The image encoder (PE-L+, 307M params) learned to extract general visual features from 1 billion training images. This is incredibly valuable and you don't want to corrupt it with 5 Potsdam tiles.
- If you touch the encoder, it will very quickly overfit to Potsdam's specific textures (red rooftops, grey cobblestones, specific German urban layouts) and FORGET everything else → catastrophic forgetting → terrible on any other German city
- The mask decoder is lightweight and only needs to learn "given these features from PE-L+, how do I draw accurate boundaries for aerial imagery". This is a much simpler adaptation that 2,048 parameters can handle on 5 images without overfitting

**Why 2,048 parameters is sufficient:**
This is roughly equivalent to a single 64×32 weight matrix, or a few small embedding vectors. With only 2,048 floating-point numbers adjusting during training, there is simply not enough capacity to memorize the idiosyncratic details of 5 specific tiles. The optimization is forced to find a genuinely generalizable signal.

For comparison: a single attention head in a transformer typically has 3 weight matrices of ~(768×768) = ~1.8M parameters each. PTSAM's entire trainable budget is ~0.001× that.

**What training looks like:**

```python
# Freeze everything
for param in sam3.image_encoder.parameters():
    param.requires_grad = False
for param in sam3.prompt_encoder.parameters():
    param.requires_grad = False

# Only the soft prompt tokens are trainable
soft_prompts = nn.Parameter(torch.randn(N_TOKENS, EMBED_DIM))  # ~2048 values total

# Training loop on 5 Potsdam tiles (with resolution bridge aug)
for epoch in range(N_EPOCHS):
    for img, label in potsdam_dataloader:
        # Random resolution bridge: simulate 20cm GSD
        if random.random() < 0.30:
            img = F.interpolate(img, scale_factor=0.25, mode='bilinear')
            img = F.interpolate(img, size=original_size, mode='bilinear')
      
        # Forward pass — soft_prompts injected into cross-attention
        pred = sam3.forward_with_soft_prompts(img, soft_prompts)
      
        loss = cross_entropy(pred, label, ignore_index=255)
        loss.backward()
        optimizer.step()  # only updates soft_prompts
```

**Transfer to Germany:**
Once trained, the soft prompt tokens encode "how to interpret blurry low-resolution top-down aerial imagery" — not anything specific to Potsdam city. Run on Frankfurt, Munich, Stuttgart, rural Bavaria — same tokens, same inference. The text class names still change per region (next section), but the soft prompts are universal.

**NOT the same as "PEFT on Potsdam":**
PEFT usually refers to fine-tuning the backbone (LoRA on attention). PTSAM specifically avoids the backbone entirely. This is important because LoRA on PE-L+ (307M params) is risky even with regularization on 5 images. PTSAM's decoder-only approach is much safer.

---

### Per-Patch Presence Filtering + Dual Output (Planned for NB03)

**The problem with different classes per patch:**
If patch A uses classes `[building, road, tree]` (indices 0,1,2) and patch B uses `[building, road, water]` (indices 0,1,2), then index 2 means "tree" in patch A and "water" in patch B. When you Gaussian-blend overlapping regions at the seam, you're averaging "tree confidence" with "water confidence" — completely meaningless. The stitch breaks.

**The correct approach: fixed superset + per-patch filtering**

Use the same N classes (same indices) for every patch. Use SAM3's presence head to suppress classes that aren't in a given patch:

```
All N classes, same indices everywhere
         ↓
For each patch, SAM3 scores each class with a presence probability (0 to 1)
         ↓
If presence score < threshold (e.g. 0.2):
    set that class's logit to -inf for this patch only
    (it will never "win" argmax, but indices stay consistent)
         ↓
Argmax still produces valid class indices
Gaussian blend works correctly at seams (same class means same thing everywhere)
         ↓
At stitch time: produce TWO outputs from the same prediction array:
    Output A — fine-grained: all N classes, distinct color per class
    Output B — coarse: merge via COARSE_MAP to 6-7 canonical categories
```

**COARSE_MAP (sketch for Hessen):**

```python
COARSE_MAP = {
    "road":              "impervious surface",
    "pavement":          "impervious surface",
    "railway":           "impervious surface",
    "building":          "built structure",
    "greenhouse":        "built structure",
    "tree":              "vegetation",
    "tree canopy":       "vegetation",
    "grassland":         "vegetation",
    "low vegetation":    "vegetation",
    "crop field":        "vegetation",
    "water":             "water",
    "bare soil":         "bare ground",
    "car":               "vehicle",
    "vehicle":           "vehicle",
    "clutter":           "clutter",
}
```

**Result:** One inference run, two visualization outputs. Toggle between 12-class detailed view and 6-class summary view. Both are consistent across the full stitched map.

---

### OSM-Grounded Auto-Prompting (Novel Idea)

**Motivation:** Different German regions have completely different land-use compositions. A forest-heavy region near Stuttgart needs different classes than industrial Ruhr. Manually writing prompts for every German tile is not scalable.

**The full pipeline:**

```
GPS bounding box of target tile
    ↓
OpenStreetMap Overpass API query:
    [out:json][bbox:lat1,lon1,lat2,lon2];
    (
      way["landuse"];
      way["natural"];
      way["highway"];
      relation["landuse"];
    );
    out body;
    ↓
Parse response → unique tags present in this tile:
    landuse: forest, farmland, residential, industrial
    natural: water, scrub
    highway: motorway, primary, secondary
    ↓
Map OSM tags to segmentation class names:
    landuse=forest → "tree canopy, forest"
    landuse=farmland → "crop field, farmland, agricultural land"
    highway=motorway → "motorway, road, paved surface"
    etc.
    ↓
Feed to LLM with context:
    "German region, 20cm aerial imagery, summer season.
     Classes present: forest, farmland, residential, motorway, water.
     Generate 3 nadir-aware aerial descriptions per class."
    ↓
LLM generates context-aware nadir prompts automatically
    ↓
Run SegEarth-OV-3 inference with these prompts
    ↓
SAM3 presence head: if presence < 0.2 across whole tile → drop that class entirely
    ↓
Final prediction: only classes that are actually there, with accurate prompts
```

**Why this is novel:**
TMPA paper (CVPR 2026) uses LLM-generated prompts but with no geospatial grounding — it just generates generic descriptions from class names. Combining OSM grounding (know WHICH classes are in THIS tile) + LLM (generate aerial descriptions for only those classes) + presence head filtering (auto-remove false positives) is a new approach not published yet.

**What you'd need to implement it:**

1. `overpy` or `requests` library + Overpass API query (free, no auth needed)
2. OSM tag → class name mapping dict
3. Any LLM API call (Claude, GPT-4, Gemini) with the right system prompt
4. Modify NB03 to accept dynamic class list instead of fixed `cls_hessen.txt`

This scales to the whole of Germany with zero manual prompt engineering per tile.

---

## PART 5 — EVALUATION WITHOUT GROUND TRUTH

---

**The problem:** For Hessen/Darmstadt, there are no official pixel-level labels. Visual scoring (like 7.9/10) is subjective and not publishable.

**Three options that give numbers without GT:**

**Option 1 — OSM Pseudo-GT F1 (best option)**
Rasterize OpenStreetMap polygons → binary masks per class → erode 3×3 pixels (removes ambiguous boundaries) → set boundary pixels to 255 (ignored) → compute F1 against predictions.

Already implemented in `src/segearth_utils/osm_eval.py` (from GeoPrompt project). Needs to be ported and run on NB03's prediction .npy files.

Gives a real metric directly comparable to the teammate's runs. This is what the Praktikum report needs.

**Option 2 — Prediction Self-Consistency**
At seam boundaries between overlapping patches, compute the entropy of the prediction. If two overlapping patches strongly disagree on the class of a seam pixel, entropy is high → model is uncertain. Lower mean entropy at seams = better model (more consistent across patches). No GT needed, computable purely from prediction arrays.

**Option 3 — Presence Score Distribution**
SAM3's presence head already computes per-class confidence for each patch. Plot the distribution across all 289 patches. If most classes have bimodal distribution (confidently present OR confidently absent), the model is making crisp decisions. If everything clusters around 0.4–0.6, the model is confused. Zero additional compute — these scores are available during inference already.

---

## PART 6 — PRIORITY QUEUE

---

1. **Wait for NB02 version 16 result** → confirm label fix → get our actual mIoU on tile 6_15
2. **Update cls_hessen.txt with viewpoint/nadir descriptors** → push NB03 → compare visual output
3. **Add presence filtering + dual coarse/fine output to NB03 stitch cell**
4. **Port osm_eval.py to Hessen** → run on NB03 predictions → get F1 number
5. **PTSAM decoder tuning** → only if time before Praktikum deadline; needs training loop
6. **OSM auto-prompting** → longer-term, novel contribution if time allows
