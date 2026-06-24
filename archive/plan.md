│ Plan: SegEarth-OV-3 — Improvement Analysis + Gemini Research Prompt                                                                                     │
│                                                                                                                                                         │
│ Context                                                                                                                                                 │
│                                                                                                                                                         │
│ The Praktikum goal is open-vocab RS segmentation on Hessen DOP20 (20cm GSD). The team's                                                                 │
│ SegEarth-OV-3 runs SAM3 (Meta's vision-language model) with text prompts — no training, pure                                                            │
│ zero-shot. We need to: (a) understand what the team has/hasn't addressed, (b) identify                                                                  │
│ concrete improvements, (c) generate a research prompt for Gemini to fill the literature gaps                                                            │
│ without burning Claude tokens, and (d) plan what to actually implement.                                                                                 │
│                                                                                                                                                         │
│ ---                                                                                                                                                     │
│ What the team has done (from reading the repo)                                                                                                          │
│                                                                                                                                                         │
│ ┌──────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐                                                                                                                                                    │
│ │       Area       │                                                            What they did                                                             │                                                                                                                                                     │
│ ├──────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                                                                                                                                    │
│ │ Resolution gap   │ Nothing explicit. slide_crop=512, stride=448 for Potsdam; 1024/768 for Frankfurt. SAM3 resizes everything to 1008×1008 internally    │                                                                                                                                                     │
│ │                  │ regardless.                                                                                                                          │                                                                                                                                                     │
│ ├────────────
│ ├──────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                                                                                                                                    │
│ │ Quantitative     │ eval.py + cfg_potsdam.py exist but are never called from the notebook. No mIoU numbers yet.                                          │                                                                                                                                                     │
│ │ eval             │                                                                                                                                      │                                                                                                                                                     │
│ ├──────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                                                                                                                                    │
│ │ PAMR             │ In repo (pamr.py) but unused.                                                                                                        │                                                                                                                                                     │
│ ├──────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                                                                                                                                    │
│ │ PEFT             │ None. SAM3 is fully frozen; all inference under @torch.inference_mode().                                                             │                                                                                                                                                     │
│ └──────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘                                                                                                                                                    │
│                                                                                                                                                         │
│ ---
│                                                                                                                                                         │
│ ---                                                                                                                                                     │
│ Improvement tiers (ranked by effort vs expected gain)                                                                                                   │
│                                                                                                                                                         │
│ Tier 1 — No training, can implement now                                                                                                                 │
│                                                                                                                                                         │
│ A. Resolution bridge at inference (Hessen)                                                                                                              │
│ Hessen DOP20 is 20cm GSD; Potsdam is 5cm → 4× resolution gap. SAM3 sees 1008×1008                                                                       │
│ regardless of input size. Upsample Hessen patches 4× (bilinear/Lanczos) before passing to                                                               │
│ SAM3 → objects appear at the right scale. Same idea as our RESOLUTION_FACTOR in rg_geoprompt.                                                           │
│ File to change: demo.py or a new demo_hessen.py — one image = image.resize(...) line.                                                                   │
│                                                                                                                                                         │
│ B. Better prompts for Potsdam/Hessen classes                                                                                                            │
│ SAM3 supports multi-word synonyms (comma-separated in cls file → same class index).                                                                     │
│ cls_potsdam.txt currently has "road" for impervious surface — too narrow.                                                                               │
│ Replace with enriched prompts: "impervious surface, road, asphalt, concrete pavement, sidewalk".                                                        │
│ Affects configs/cls_potsdam.txt. Zero risk.                                                                                                             │
│                                                                                                                                                         │
│ C. Gaussian sliding window                                                                                                                              │
│ Current overlap averaging (preds / count_mat) weights edge pixels equally to center pixels,                                                             │
│ causing seam
│ Apply LoRA r=8 or r=16 to Q,V projections in the ViTDet attention layers (same as our                                                                   │
│ DINOv2+LoRA). Train on Potsdam 5-tile split (2420 patches). Requires:                                                                                   │
│ 1. Remove @torch.inference_mode() from Sam3Processor.set_image()                                                                                        │
│ 2. Freeze all SAM3 params                                                                                                                               │
│ 3. Inject LoRA adapters via peft library                                                                                                                │
│ 4. Train with cross-entropy loss on Potsdam GT                                                                                                          │
│ Estimated gain: our DINOv2+LoRA went from ~80% to 85.26% with r=16.                                                                                     │
│                                                                                                                                                         │
│ F. Soft-prompt tuning (CoCoOp-style)                                                                                                                    │
│ Learn K soft tokens prepended to each class text (instead of hard-coded strings).                                                                       │
│ Only the prompt tokens are trained; SAM3 text encoder stays frozen.                                                                                     │
│ Train on Potsdam → test on Hessen. Much cheaper than LoRA (few thousand params).                                                                        │
│                                                                                                                                                         │
│ Tier 3 — Test-time adaptation (no labels)                                                                                                               │
│                                                                                                                                                         │
│ G. Test-time prompt adaptation (TTPA equivalent)                                                                                                        │
│ At test time, adapt text prompts via entropy minimization on Hessen patches — same idea as our                                                          │
│ ttpa.py but applied to SAM3's text embeddings. No GT labels needed.                                                                                     │
│ Risk: SAM3's text path is forward_text() → needs gradient flow enabled.                                                                                 │
│                                                                                                                                                         │
│ ---                                                                                                                                                     │
│ Gemini research prompt (copy-paste ready)                                                                                                               │
│                                                                                                                                                         │
│ I am working on a Praktikum project in remote sensing segmentation. My context:                                                                         │
│                                                                                                                                                         │
│ SYSTEM SETUP:                                                                                                                                           │
│ - Model: SegEarth-OV-3, which wraps Meta's SAM3 (Segment Anything 3) with an MMSegmentation                                                             │
│   interface for open-vocabulary segmentation.                                                                                                           │
│ - SAM3 architecture: ViTDet vision backbone (RoPE attention) + VETextEncoder (language) +                                                               │
│   transformer decoder that cross-attends text queries to visual features. Outputs instance masks,                                                       │
│   semantic masks, and presence scores per text prompt.                                                                                                  │
│ - Inference only (no training): currently everything under torch.inference_mode(), fully frozen.                                                        │
│ - Kaggle T4 GPU (16GB). No multi-GPU. Training budget: ~1-2 hours max.                                                                                  │
│                                                                                                                                                         │
│ DATASETS:                                                                                                                                               │
│ - Training available: ISPRS Potsdam, 5cm GSD, 6 classes (impervious surface, building, low                                                              │
│   vegetation, tree, car, clutter). Tile-level split: 5 tiles train (2420 patches 512×512),                                                              │
│   1 tile val (484 patches).                                                                                                                             │
│ - Test target: Hessen DOP20, 20cm GSD — 4× resolution gap vs Potsdam. No labels.                                                                        │
│ - Also Frankfurt DOP20 (20cm GSD, no labels) — same resolution as Hessen.                                                                               │
│                                                                                                                                                         │
│ WHAT WE HAVE DONE ALREADY (do not suggest these):                                                                                                       │
│ - DINOv2+LoRA r=16 on Q,V (85.26% mIoU supervised on Potsdam) — different model, already done.                                                          │
│ - Resolution bridge augmentation during DINOv2 training (downsampled 30% of patches 4× to                                                               │
│   simulate 20
│    on ISPRS Potsdam (6-class)? What mIoU do SegEarth-OV and TPOVSeg report on Potsdam                                                                   │
│    specifically? Are those numbers zero-shot or fine-tuned?                                                                                             │
│                                                                                                                                                         │
│ 7. Sliding window improvements: Is Gaussian-weighted overlap averaging (vs uniform averaging)                                                           │
│    for sliding window inference documented to help in RS segmentation? By how much typically?                                                           │
│                                                                                                                                                         │
│ Please be specific, cite papers, and flag if something is speculative vs. established.                                                                  │
│                                                                                                                                                         │
│ ---                                                                                                                                                     │
│ What to implement (after Gemini answers)                                                                                                                │
│                                                                                                                                                         │
│ Immediate (before Gemini, no research needed)                                                                                                           │
│                                                                                                                                                         │
│ 1. configs/cls_potsdam.txt → enriched prompts (30 min)                                                                                                  │
│ 2. demo_hessen.py → 4× upsample for 20cm input (30 min)                                                                                                 │
│ 3. Gaussian sliding window in segearthov3_segmentor.py (1 hr)                                                                                           │
│ 4. PAMR activation flag (30 min)                                                                                                                        │
│                                                                                                                                                         │
│ After Gemini                                                                                                                                            │
│                                                                                                                                                         │
│ 5. Decide on LoRA vs soft-prompt vs TTA based on literature findings                                                                                    │
│ 6. Implement whichever PEFT tier is realistic on T4 in <2hr training                                                                                    │
│                                                                                                                                                         │
│ Files to modify (all in segearth-ov3/repo/ fork)                                                                                                        │
│                                                                                                                                                         │
│ - configs/cls_potsdam.txt — better prompts                                                                                                              │
│ - segearthov3_segmentor.py — Gaussian slide, PAMR flag                                                                                                  │
│ - New: demo_hessen.py — Hessen-specific config + upsampling                                                                                             │
│ - New: configs/cfg_hessen.py — Hessen config (based on cfg_potsdam.py)                                                                                  │
│ - New: configs/cls_hessen.txt — Hessen-tuned prompts                                                                                                    │
│                                                                                                                                                         │
│ ---                                                                                                                                                     │
│ Verification                                                                                                                                            │
│                                                                                                                                                         │
│ 1. Run python eval.py configs/cfg_potsdam.py on val tile → baseline mIoU                                                                                │
│ 2. Run with enriched prompts → compare mIoU                                                                                                             │
│ 3. Run with Gaussian sliding window → compare seam artifacts visually                                                                                   │
│ 4. Run demo_hessen.py on sample Hessen patch with and without 4× upsample → visual comparison                                                           │
╰────────────────────────────────────────────────────────────────────────────────────────────────
