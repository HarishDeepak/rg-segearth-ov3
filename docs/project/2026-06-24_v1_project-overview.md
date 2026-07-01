# PROJECT.md — rg-segearth-ov3

## Goal

Run SegEarth-OV-3 (SAM3-based open-vocabulary segmentation) on Hessen DOP20 (20cm GSD) and
compare against GeoPrompt (rg-geoprompt-peft project). Improve zero-shot results through:
1. Enriched multi-synonym class prompts
2. Gaussian sliding window (no seam artifacts)
3. Resolution-aware `slide_crop=256` for 20cm imagery
4. PAMR boundary refinement (optional)

## Experiment log

| Date | Notebook | Result | Notes |
|------|----------|--------|-------|
| 2026-06-22 | segearth03 (teammate) | Visual 7.9/10 | 12-class, single-name prompts, qualitative only |
| — | NB02 | **TODO** | Potsdam mIoU with enriched prompts |
| — | NB03 | **TODO** | Hessen F1 with Gaussian window + enriched prompts |

## Comparison context

| Method | Potsdam mIoU | Darmstadt F1 | Notes |
|---|---|---|---|
| GeoPrompt (ours) | 84.9% (supervised) | 0.2059 mean | rg-geoprompt-peft project |
| SegEarth-OV ZS | 48.5% | — | Literature (zero-shot, not our run) |
| SegEarth-OV-3 ZS | 57.8% | — | Literature (zero-shot, not our run) |
| **Our OV-3 run** | **TODO** | **TODO** | Will fill in after NB02/NB03 |

## Architecture decisions

- No training — SAM3 fully frozen
- Our fork: `HarishDeepak/SegEarth-OV-3`
- Upstream: `dummy-irl/SegEarth-OV-3` (teammate's repo)
- slide_crop=256/stride=128 for Hessen (resolution bridge via smaller crops)
- bg_idx=5 for Hessen (background = last class), bg_idx=5 for Potsdam (clutter)
