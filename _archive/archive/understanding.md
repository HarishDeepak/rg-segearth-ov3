The Technical Improvements

1. Enriched prompts — single words like road are ambiguous in the shared CLIP embedding space. We replace them with synonym lists (comma-separated, already supported by the codebase) that get averaged into one stable embedding. Example: impervious surface, road, asphalt, paved path, concrete pavement, sidewalk. Zero code change, just edit cls_potsdam.txt. Expected +1–3% mIoU.
2. Gaussian sliding window — the current code uses preds / count_mat (uniform). The problem: ViTDet's attention is truncated at tile edges, making edge predictions unreliable. Two tiles' bad edges average together → visible seams. A 2D Gaussian kernel weights each tile's center contribution high and edges near-zero. When tiles overlap, the confident center dominates automatically. Documented ~+2% mIoU, prevents up to 47% precision loss on large objects (forests, fields). We also increase overlap from 12.5% → 50% for this to work properly.
3. Hessen config with smaller tiles — the 4× resolution gap (5cm Potsdam vs 20cm Hessen) means a 512px Hessen crop covers 4× the geographic area of a 512px Potsdam crop. SAM3's learned scale intuition breaks. The fix: use 256px tiles instead of 512px. SAM3 internally resizes to 1008×1008 anyway, so feeding it a smaller crop effectively zooms in — the model "sees" Hessen structures at the spatial density it was calibrated for. Don't upscale the image — bilinear interpolation just invents blurry detail.
4. PAMR — wire it in behind a flag as post-processing. Refines jagged boundaries using local pixel affinity. Good for building edges and field boundaries.

---

The Presentation Demo — My Confident Take

Your instinct is right, and here's exactly why it works:

The setup: Display a grid of 8–10 pre-selected Hessen tiles. Audience picks 2 by index.

Run A (default): Model runs with Potsdam vocabulary — road, building, grass, tree, car. It partially fails on Hessen. That's intentional — it shows the domain gap viscerally. The audience sees their chosen image being misclassified.

Run B (user-guided): Audience looks at the RGB image and says what they see — "fields, forest, a village, dirt tracks, maybe a pond." Those descriptions become the class list. Model runs again. The visual features don't change at all (frozen ViTDet) — only the text queries to the cross-attention decoder change.

Why this is the right demo: It teaches exactly what "open vocabulary" means better than any slide could. The audience co-authors the experiment. They see the model fail, understand why it fails (wrong vocabulary, not wrong model), and then fix it themselves. And critically — it's honest. We're not faking a perfect result.

Why NOT point/click prompts: That's interactive instance segmentation (SAM's original use case). It would be impressive but it doesn't demonstrate open-vocabulary transfer, which is what this whole project is about. Text prompts from the audience make the concept land directly.


Implementation Order

┌──────┬─────────────────────────────────────────────────────────┬────────┐
│ Step │                          What                           │  Time  │
├──────┼─────────────────────────────────────────────────────────┼────────┤
│ 1    │ Enriched prompts (cls_potsdam.txt)                      │ 30 min │
├──────┼─────────────────────────────────────────────────────────┼────────┤
│ 2    │ Gaussian sliding window (segearthov3_segmentor.py)      │ 1.5 hr │
├──────┼─────────────────────────────────────────────────────────┼────────┤
│ 3    │ Hessen config + classes (cfg_hessen.py, cls_hessen.txt) │ 30 min │
├──────┼─────────────────────────────────────────────────────────┼────────┤
│ 4    │ Fix demo.py inconsistencies                             │ 30 min │
├──────┼─────────────────────────────────────────────────────────┼────────┤
│ 5    │ PAMR flag                                               │ 30 min │
├──────┼─────────────────────────────────────────────────────────┼────────┤
│ 6    │ Presentation notebook (demo_presentation.ipynb)         │ 2 hr   │
└──────┴─────────────────────────────────────────────────────────┴────────┘
