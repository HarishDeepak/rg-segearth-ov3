Potsdam does not affect Darmstadt — they're independent

Two separate inference runs, no shared state. Nothing from Potsdam carries into Darmstadt. SAM3 is frozen, so there's no learning or memory between images.

The resolution gap is a different problem entirely: it's about what SAM3's frozen weights "expect" objects to look like in pixel space, based on

---

The actua

SAM3 was  images).During training, buildings, roads, etc. appeared at certain pixel sizes. The ViTDet backbone
learned fse scales.That's baked into the frozen weights permanently.

Every image fed to SAM3 gets resized to 1008×1008 (Sam3Processor, resolution=1008, confirmed in
the code) does anobject appear after that resize?

![1782342752958](image/1/1782342752958.png)



With a 512px Hessen crop, the building is ~4× smaller than SAM3's feature detectors were calibrated for → CLIP text-image matching for "building" is weak → bad segmentation.

With a 256px crop, the same building fills a larger fraction of the frame. After SAM3 upsamples it to 1008px, the building appears at ~195px — much closer to the scale SAM3 was trained on → better feature response → better predictions.

This is not upscaling the image. The actual pixels are unchanged. You're just feeding SAM3 a smaller geographic area per crop, so the objects in it fill more of the frame.

---

Why not bilinear upscale Hessen 4× first?

Bilinear interpolation invents smooth blurry pixels. SAM3's ViTDet features fire on real texture and edge information — they don't improve on interpolated pixels. The feature response is the same or worse. The extra/understanding.md explicitly notes this: "Don't upscale the image — bilinear interpolation just invents blurry detail."

The extra/plan.md mentions SimFeatUp (feature-level upsampling, not pixel-level) as a Tier 1 future improvement if 256px crops aren't sufficient — but that's a more complex addition for later.

---

What the extra/ files add

- understanding.md: confirms 256px crop is the chosen fix, lists all 5 improvements with rationale
- plan.md: full improvement roadmap — resolution bridge (SimFeatUp), enriched prompts, Gaussian sliding window, PAMR, LoRA on ViTDet
- SAM3 Remote Sensing Segmentation Research.md: German-language literature review confirming scale mismatch between RS imagery and SAM training distribution is a known problem; tile sizing is a standard mitigation
- note.md: project strategy (local dev vs Kaggle push) — not relevant to this question





The Miniconda installer is appearing in your Output Data because wget downloads to /kaggle/working/ by default, which Kaggle treats as output. The actual env installs to /tmp/miniconda — the .sh script just sits there wasting 163MB of output quota.

Why conda is needed at all:

SegEarth-OV-3 requires mmcv==2.2.0 with compiled CUDA ops + mmsegmentation==1.2.2 + torch==2.4.0. Kaggle's base env ships a newer PyTorch (2.5+), and mmcv's CUDA ops are compiled against a specific torch ABI — if versions don't match exactly, you get import errors. So a fresh isolated env with a pinned torch 2.4.0 is needed. GeoPrompt only needed HuggingFace + standard PyTorch, which Kaggle provides natively.

You could replace conda with venv (same idea, lighter) but you'd still need to pip-install torch 2.4.0 into it. Conda isn't strictly necessary — it's just the path we took.

Fix the 163MB output problem — just download to /tmp:

  Read 1 file

Here's the full picture:

Why conda is needed: mmcv==2.2.0 requires compiled CUDA ops that must match the torch version exactly. Kaggle's base environment ships PyTorch 2.5+, which is ABI-incompatible with mmcv 2.2.0's prebuilt wheels. So you need a fresh env with torch==2.4.0 pinned. GeoPrompt only needed HuggingFace + standard PyTorch — both already in Kaggle's base — so no isolation was needed.

Why 163MB shows in Output Data: wget downloads to the current directory, which on Kaggle is /kaggle/working/. Anything in /kaggle/working/ becomes output. The env itself installs to /tmp/miniconda (correct), but the installer .sh file just sits in working and gets swept up as output.

Fix — one-liner change in all three notebooks: download the installer to /tmp/ instead:

# before

!wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!bash Miniconda3-latest-Linux-x86_64.sh -b -p /tmp/miniconda

# after

!wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda_installer.sh
!bash /tmp/miniconda_installer.sh -b -p /tmp/miniconda






 /btw why was that prob
    The OOM happened because of how multi-synonym prompts interact with the slide_inference accumulator.

    cls_potsdam.txt has 41 total synonym words across 6 classes (e.g. "impervious surface, road, asphalt, paved road..." = 8 synonyms for class 0). The model runs one forward pass per synonym, so:

    self.num_queries = len(self.query_idx)  # = 41, not 6
    preds = torch.zeros((self.num_queries, h_img, w_img), device=self.device)
    # = [41, 6000, 6000] float32 = 5.9 GB already on GPU


Two things here — NB02 ran for 2 hours and failed at the very last step (OOM during final aggregation), and NB03 needs to be scoped to one source image with a stitched mosaic output.

Let me look at both files to plan the fixes:

  Read 1 file

Found the OOM cause — num_queries = 41 (total synonym count across all 6 classes). So preds is [41, 6000, 6000] float32 = 5.9 GB already allocated, and line 157 tries to allocate another 5.9 GB for the output. With SAM3 weights (~3.5 GB) still loaded, that exceeds 16 GB.

Fix is one line — in-place division. Also fixing NB03 to run one source image + stitch mosaic:

25/6 after potstadm results



The run succeeded but the results are catastrophic — 6.68% mIoU vs teammate's 74.99%. Something is fundamentally broken. Let me diagnose.

● Read(D:\VC\rg-segearth-ov3\configs\cls_potsdam.txt)
Read 7 lines

● Read(D:\VC\rg-segearth-ov3\results\nb02_gt_vs_pred.png)

● Search(pattern: "results/**/*.png")
  ⎿  Found 4 files
     results\nb01_dop20_preview.png
     results\nb01_potsdam_preview.png
     results\nb02\nb02_gt_vs_pred.png
     results\nb02\show_dir\vis_data\vis_image\test_top_potsdam_6_15_RGB.tif_0.png

● Bash(cd "D:\VC\rg-segearth-ov3" && git show HEAD~5:configs/cls_potsdam.txt 2>/dev/null || echo "not found")
impervious surface, parking area
rooftop, building
low vegetation, grass
tree, forest canopy
car, vehicle
clutter, background

● Read(D:\VC\rg-segearth-ov3\results\nb02\nb02_gt_vs_pred.png)
Read image (1.1MB)

● Read(D:\VC\rg-segearth-ov3\results\nb02\show_dir\vis_data\vis_image\test_top_potsdam_6_15_RGB.tif_0.png)
Read image (72.2MB)

❯ What the teammate's 74.99% actually means: It's on tiles 6_13 + 6_14, which are not the held-out val tile. Our eval on 6_15 will give the honest number. Literature claims 57.8% — if our enriched prompts help, we might beat that. but how does 613 pic and 615 be diff>? anyway its zeroshotting both
Interrupted · What should Claude do instead?

● Search(pattern: "results/**/*.png")
  ⎿  Found 4 files
     results\nb01_dop20_preview.png
     results\nb01_potsdam_preview.png
     results\nb02\nb02_gt_vs_pred.png
     results\nb02\show_dir\vis_data\vis_image\test_top_potsdam_6_15_RGB.tif_0.png

● Bash(cd "D:\VC\rg-segearth-ov3" && git show HEAD~5:configs/cls_potsdam.txt 2>/dev/null || echo "not found")
impervious surface, parking area
rooftop, building
low vegetation, grass
tree, forest canopy
car, vehicle
clutter, background

● Read(D:\VC\rg-segearth-ov3\results\nb02\nb02_gt_vs_pred.png)
Read image (1.1MB)

● Read(D:\VC\rg-segearth-ov3\results\nb02\show_dir\vis_data\vis_image\test_top_potsdam_6_15_RGB.tif_0.png)
Read image (72.2MB)

❯ What the teammate's 74.99% actually means: It's on tiles 6_13 + 6_14, which are not the held-out val tile. Our eval on 6_15 will give the honest number.
  Literature claims 57.8% — if our enriched prompts help, we migh3 pic and 615 be diff>? anyway its zeroshotting both
Interrupted · What should Claude do instead?

❯ /memory 57.8% — if our enriched prompts help, we might beat that. but how does 613 pic and 615 be diff>? anyway its zeroshotting both
  Interrupted · What should Claude do instead?

You're right — zero-shot means the model has never been tuned on ANY Potsdam tile, so 6_13 vs 6_15 shouldn't matter in principle. The 67× gap (74.99% vs 6.68%) is not a data split issue — something is actually broken or the tiles look genuinely different to SAM3.

Looking at the prediction image from this run:

  Read 1 file

That prediction image is diagnostic — the right half (prediction) shows large blue and cyan blocks. But the metrics say building=0.04%. That contradiction needs resolving. Let me check the sliding window code.

  Read 1 file, ran 1 shell command

I see it. Looking at cls_potsdam.txt — there's an empty line 7. readlines() returns 7 elements, creating a 7th "class" with an empty string prompt. That ghost class steals most predictions.

Let me verify this is actually the bug:

  Ran 1 shell command

OK the file is clean — 6 lines, no ghost class. Let me check what the stock SegEarth-OV-3 uses vs ours, and look at the archive.

  Ran 2 shell commands
Interrupted · What should Claude do instead?

❯ wait a sec, when u analyse can u also see @"segearth03 (2).ipynb" @"segearth03 (1).ipynb"  and also https://github.com/dummy-irl/SegEarth-OV-3 pull this teamate set up repo and see for the differences that might cause the problems in detail, i know that the project structure is bit diff from that one but we have to do these to solve ... continue /effort low
Read segearth03 (2).ipynb (23 cells)
Read segearth03 (1).ipynb (22 cells)

  Ran 2 shell commands

● Fetch(https://raw.githubusercontent.com/dummy-irl/SegEarth-OV-3/main/configs/cls_potsdam.txt)
Received 37 bytes (200 OK)

● Fetch(https://raw.githubusercontent.com/dummy-irl/SegEarth-OV-3/main/segearthov3_segmentor.py)
Received 9.2KB (200 OK)

● Fetch(https://raw.githubusercontent.com/dummy-irl/SegEarth-OV-3/main/segearthov3_segmentor.py)
Received 9.2KB (200 OK)

● Fetch(https://raw.githubusercontent.com/dummy-irl/SegEarth-OV-3/main/configs/cfg_potsdam.py)
Received 774 bytes (200 OK)
Interrupted · What should Claude do instead?

❯ its better if u clone and compare and try to reduce token usage and also see carefull what this github and those 2 nbs does and what and where we differ and stuff /compact  and continue and also try to make the inf time less coz for them its less somehow

  Searched for 2 patterns, read 2 files, listed 2 directories, ran 6 shell commands
Referenced file segearth03 (2).ipynb
Read config_local.py (54 lines)
Referenced file segearth03 (1).ipynb
Referenced file notebooks\NB02_potsdam_eval.ipynb
Read configs\cls_potsdam.txt (7 lines)

Now I have the full picture. Let me read our segmentor's inference logic to spot any bugs.

  Searched for 2 patterns, read 1 file, ran 2 shell commands

Now I have a clear picture of the root cause. Let me write the plan.

● Updated plan
/plan to preview

● User approved Claude's plan
Plan saved to: C:\Users\haris\.claude\plans\lovely-riding-hennessy.md · /plan to edit
Fix NB02: Get from 6.68% → ~57.8% mIoU on Potsdam

Context

NB02 ran SegEarth-OV-3 on Potsdam val tile 6_15 and got 6.68% mIoU — catastrophically below the 57.8% literature baseline and the teammate's 68-75%. Per-class breakdown:

┌────────────────────┬────────┐
│       Class        │  IoU   │
├────────────────────┼────────┤
│ impervious_surface │ 8.64%  │
├────────────────────┼────────┤
│ building           │ 0.04%  │
├────────────────────┼────────┤
│ low_vegetation     │ 28.45% │
├────────────────────┼────────┤
│ tree               │ 0.0%   │
├────────────────────┼────────┤
│ car                │ 2.95%  │
├────────────────────┼────────┤
│ clutter            │ 0.0%   │
└────────────────────┴────────┘

Pattern: model overwhelmingly predicts low_vegetation/class ro. Inference took 2034 seconds vs ~212s for the original.

Root cause

Primary: slide_crop=512, slide_stride=256 is wrong for Potsdam.

- Our config: 529 crops × 12 queries = 6348 SAM3 forward passes, 2034 seconds
- Original: 36 crops × 6 queries = 216 SAM3 forward passes, ~212 seconds
- 512 px crops at 5 cm GSD cover only 25.6 m × 25.6 m — too
- SAM3 upsamples 512→1008 internally (2× vs baseline 1×), changing spatial statistics

Secondary: synonym prompts are risky untested changes.

- "rooftop" instead of "building" — SAM3/CLIP sees rooftops otos, not from above
- "background" as clutter synonym — activates for ANY area that isn't foreground, causes class 5 to hijack pixels or inflate class 2 scores indirectly
- "parking area" instead of "road" — narrower concept, missees
- "forest canopy" instead of "tree" — more ambiguous concept

The Gaussian accumulation code is mathematically correct and is NOT the bug. With slide_crop=1024, slide_stride=1024 (no overlap), each pixel is covered by exactly one crop — Gaussian vs simple average gives identical results after normalization.

Fix (two files to change)

1. configs/cfg_potsdam.py — change slide settings

# BEFORE (ours, broken):

slide_stride=256,
slide_crop=512,

# AFTER (matches original):

slide_stride=1024,
slide_crop=1024,

2. configs/cls_potsdam.txt — revert to original prompts

# AFTER (original 6 single-word prompts):

road
building
grass
tree
car
clutter

These are the exact prompts from dummy-irl/SegEarth-OV-3 used to achieve 57.8% in the paper. Get baseline first; add synonyms as an experiment after.

NB02 notebook changes needed

The notebook already handles:

- RGB→indexed label conversion (correct, uses standard Potsdam palette)
- Cloning from HarishDeepak/rg-segearth-ov3
- SAM3 checkpoint via config_local.py (symlink fix already in place)

No notebook cell changes needed — the config file changes propagate automatically.

Steps to execute

1. Edit configs/cfg_potsdam.py: slide_crop=512→1024, slide_stride=256→1024
2. Edit configs/cls_potsdam.txt: 6 single-word prompts
3. git add configs/cfg_potsdam.py configs/cls_potsdam.txt && git commit && git push
4. Re-run NB02 on Kaggle T4

Expected outcome

- mIoU ≈ 57–68% (matches literature / teammate baseline)
- Inference time ≈ 200–250 seconds (10× speedup from 2034s)

After baseline confirmed

Once 57.8%+ is confirmed, can test synonyms incrementally:

- Try "building" + "rooftop" (but keep "building" as first synonym, not "rooftop")
- Test "road" + "impervious surface" (keep "road" first)
- Avoid "background" as a synonym — too generic
- With slide_crop=1024, slide_stride=1024 (no overlap), synonyms only cost extra SAM3 passes per crop — manageable

Two edits, then push.

● Update(configs\cfg_potsdam.py)





What changed and why:

┌──────────────────┬──────────────────┬─────────────────────────┐
│                  │ Before (broken)  │       After (fix)       │
├──────────────────┼──────────────────┼─────────────────────────┤
│ slide_crop       │ 512 px           │ 1024 px                 │
├──────────────────┼──────────────────┼─────────────────────────┤
│ slide_stride     │ 256 px           │ 1024 px                 │
├──────────────────┼──────────────────┼─────────────────────────┤
│ Total crops      │ 529              │ 36                      │
├──────────────────┼──────────────────┼─────────────────────────┤
│ Queries per crop │ 12               │ 6                       │
├──────────────────┼──────────────────┼─────────────────────────┤
│ SAM3 passes      │ 6348             │ 216                     │
├──────────────────┼──────────────────┼─────────────────────────┤
│ Expected time    │ 2034 s           │ ~210 s                  │
├──────────────────┼──────────────────┼─────────────────────────┤
│ Prompts          │ 2 synonyms/class │ single words (original) │
└──────────────────┴──────────────────┴─────────────────────────┘

The diagnosis: 512-px crops at 5 cm GSD = 25.6 m patches with no surrounding context, plus SAM3 internally upsampling 2× — this caused "grass" to dominate nearly every small crop, pushing class 2 everywhere. With 1024-px crops and no overlap, each pixel is seen once at the resolution SAM3 was designed for, matching the conditions that produce 57.8% in the literature.

Once you have the baseline confirmed (~57.8% mIoU), we can start adding synonyms back — but only after seeing which classes underperform and tuning carefully.
