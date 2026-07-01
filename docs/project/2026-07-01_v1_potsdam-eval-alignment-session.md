# Session log + plan: Potsdam eval alignment with teammate + repo cleanup

## Part 1 — Context: teammate's eval methodology vs ours (already done)

Teammate (dummy-irl) shared their exact Potsdam evaluation methodology by email plus a results
screenshot (aAcc 81.2, mIoU 62.26, mIoU_NoClutter 70.6). Investigated and acted on this across the
session — summarized below for the record, since this plan file is being moved into the project as
a permanent doc, not just a throwaway plan-mode artifact.

### What was found

**Teammate's pipeline** (read from his repo `dummy-irl/SegEarth-OV-3`, local clone at
`D:\VC\rg-segearth-ov3\teammate\SegEarth-OV-3`):
- `configs/cls_potsdam.txt`: 6 single-word classes incl. clutter (`road, building, grass, tree,
  car, clutter`), one shared file for all 6 Potsdam tiles (no per-tile prompts).
- `configs/cfg_potsdam.py`: `slide_crop=1024`, `slide_stride=1024`, `prob_thd=0.1`, `bg_idx=5`.
- `configs/base_config.py`: `test_evaluator = dict(type='IoUMetricNoClutter', ...)` — a custom
  subclass (in `custom_metrics.py`) that wraps mmseg's standard `intersect_and_union`/
  `total_area_to_metrics` math and additionally reports a clutter-excluded mean via
  `np.delete(ret_metric_value, 5)`.
- `eval.py` → `mmengine.Runner.from_cfg(cfg).test()`, all 6 tiles pooled into one confusion matrix
  (his "Iter(test) [6/6]" = 6 tiles, one pooled score, not 6 averaged separate scores).
- RGB GT → label index conversion happens via mmseg's **built-in** `PotsdamDataset`
  (`reduce_zero_label=True`), not custom code in his repo.
- Sliding window overlap blending: **uniform average** — `preds += crop_logit; count += 1`, then
  `preds / count`. Same grid math as ours (`h_grids`/`w_grids` formula identical), but overlap
  regions are averaged with equal weight everywhere.

**Our pipeline before this session:**
- `configs/cls_potsdam.txt`: 5 classes, multi-synonym, **no clutter prompt** — clutter only
  emerged via `prob_thd`/`bg_idx=5` fallback thresholding, never explicitly queried.
- `configs/base_config.py`: used plain `IoUMetric` (no NoClutter variant).
- NB02 (meant to mirror his mmseg `eval.py` approach) never produced results historically —
  crashed on a matplotlib backend bug (`MPLBACKEND` env leak from Kaggle's Jupyter kernel into
  mmengine's visualizer init), marked superseded in `PROGRESS.md`.
- Actual working numbers (NB05b → NB06 → NB07) bypass mmseg entirely: hand-rolled
  `Sam3Processor`-based inference + custom 5-class mIoU (`EVAL_CLASSES=[0,1,2,3,4]`, clutter
  baked out, not reported as a variant). No per-class Fscore/Precision/Recall table, no aAcc.
- `segearthov3_segmentor.py` (root-level, our fork) carries a **local modification not in his
  repo**: Gaussian-weighted sliding-window overlap blending (`_make_gaussian_kernel` — center
  pixels of each crop get near-full weight, edges down-weighted, vs. his uniform average). Same
  crop positions/grid, different overlap math. This is a real, intentional improvement
  (documented in CLAUDE.md as "reduces seam artifacts") but means our sliding-window inference is
  **not a byte-for-byte match** to his baseline.

### Changes made this session (already committed + pushed to `HarishDeepak/rg-segearth-ov3`)

Commit `38afff5` — "Align Potsdam eval with teammate's exact protocol":
- `configs/cls_potsdam.txt` → replaced with his exact 6-class single-word list.
- `custom_metrics.py` (new, repo root) → `IoUMetricNoClutter`, copied/adapted from his design.
- `configs/base_config.py` → evaluator `IoUMetric` → `IoUMetricNoClutter`.
- `eval.py` → added `import custom_metrics` (registers the metric); forced
  `matplotlib.use("Agg", force=True)` before mmengine builds its visualizer, fixing the crash that
  killed the last historical NB02 run before any metrics were produced.
- `configs/cfg_potsdam.py` needed **no changes** — already matched his config.

Commit `8838abe` — "Add NB02b: all-6-tiles Potsdam eval with per-tile + combined metrics":
- New notebook `notebooks/NB02b_potsdam_eval_all_tiles.ipynb`: stages + evaluates each of the 6
  Potsdam tiles (`5_14, 5_15, 6_13, 6_14, 6_15, 7_13`) **separately** (per-tile metrics), then
  stages all 6 together and runs once more for the **pooled/combined** metric matching his "6/6"
  protocol exactly. Outputs a summary CSV with both.

### Kaggle run in progress

Pushed kernel `harish77718/nb02b-potsdam-all-tiles` (T4 GPU) via Kaggle API and it is running
(started ~12:09, expected 45–90 min for 7 total eval passes: 6 per-tile + 1 combined). Not yet
polled to completion as of this plan being written — check `kaggle kernels status
harish77718/nb02b-potsdam-all-tiles` and pull output via `kaggle kernels output
harish77718/nb02b-potsdam-all-tiles -p <dir>` when ready. Compare resulting `mIoU`/`mIoU_NoClutter`
against teammate's screenshot numbers (62.26 / 70.6).

### Known remaining methodology gap (flagged, not yet resolved)

Our sliding-window inference uses Gaussian-weighted overlap blending; his uses uniform averaging.
Same window positions, different blend math in overlap zones (~50%+ of every tile, since
stride == crop size). This means the NB02b run is aligned on prompts/config/evaluator but **not**
on this one axis. Two options discussed, not yet decided:
- (A) Add a config flag / temporary revert to plain uniform averaging for a true 1:1 comparison
  run, keeping the Gaussian version as a separate "our improvement" experiment.
- (B) Leave Gaussian version as-is and treat any number difference as expected (attributable to a
  known, intentional improvement, not a bug).

**Next step when resuming this thread:** decide (A) vs (B), and report the NB02b results once the
Kaggle run completes.

## Part 2 — Repo cleanup / reorganization (this session's new request)

### Problem

The project directory has accumulated messy, duplicate, badly-named files across two sessions of
work. Confirmed duplicates (byte-identical, via `diff`):
- `archive/note.md` == `extra/note.md`
- `archive/plan.md` == `extra/plan.md`
- `archive/understanding.md` == `extra/understanding.md`

Near-duplicate (same research content, different line-wrapping/reflow — not byte-identical but
same substance):
- `archive/SAM3 Remote Sensing Segmentation Research.md` vs `extra/SAM3 Remote Sensing
  Segmentation Research.md`

Root-level files with no dates/versions in their names, ambiguous purpose from filename alone:
`CLAUDE.md`, `DEVLOG.md`, `PROGRESS.md`, `PROJECT.md`, `SETUP.md`, `wf.md`.

Other structural issues:
- `sam3/` and root-level `*.py` files (`eval.py`, `segearthov3_segmentor.py`, `custom_metrics.py`,
  `custom_datasets.py`, `custom_transforms.py`, `demo.py`, `pamr.py`,
  `segearthov3_change_detector.py`, `config_local.py`) are actually **our fork's SegEarth-OV-3
  code**, sitting unglazed at the repo root, mixed in with our own project docs and `configs/`.
- `teammate/` (his repo clone + his Kaggle notebook exports) is already gitignored, correctly kept
  out of git, but still physically sits inside our project folder next to our own stuff.
- `archive/segearth-ov3-old/` contains an older fork snapshot (`repo/`) plus its own
  `NOTES.md`/`REVIEW.md` — a second, separate "old copy of the fork" story.

**Constraint from user: do not delete anything.** Every file must end up somewhere findable, not
removed. Duplicates get consolidated by keeping one canonical copy in an organized location and
noting the duplicate in a manifest, not by silently deleting the other copy — actually, since user
said no deletions, exact duplicates should be **left alone or consolidated only with explicit
confirmation**, not auto-deleted even if identical.

### Proposed structure (for review before any moves happen)

```
rg-segearth-ov3/
├── CLAUDE.md                          # stays at root — read by Claude Code at session start
├── docs/
│   ├── project/                       # OUR project's own docs, dated
│   │   ├── 2026-06-24_v1_project-overview.md      (was PROJECT.md)
│   │   ├── 2026-06-28_v1_setup-guide.md            (was SETUP.md)
│   │   ├── 2026-06-28_v2_progress-log.md           (was PROGRESS.md, most recent edit date)
│   │   ├── 2026-06-28_v2_devlog.md                 (was DEVLOG.md)
│   │   └── 2026-06-25_v1_workflow-notes.md         (was wf.md)
│   └── research/                      # background research, not day-to-day project state
│       ├── 2026-06-22_v1_sam3-remote-sensing-research.md
│       ├── 2026-06-27_v2_remote-sensing-segmentation-adaptation.md
│       ├── 2026-06-24_v1_understanding.md
│       ├── 2026-06-24_v1_note.md
│       ├── 2026-06-22_v1_plan.md
│       ├── 2026-06-16_v1_lastclaudechat.md
│       ├── 2026-06-27_v1_plan1-handoff.md          (was extra/plan1.md)
│       ├── 2026-06-25_v1_infoplan.md               (was extra/infoplan.md)
│       └── DUPLICATES.md              # manifest noting which files in archive/ vs extra/
│                                       # were byte-identical, so nothing is silently lost
├── fork/                              # OUR fork's SegEarth-OV-3 code, currently loose at root
│   ├── eval.py
│   ├── segearthov3_segmentor.py
│   ├── custom_metrics.py
│   ├── custom_datasets.py
│   ├── custom_transforms.py
│   ├── demo.py
│   ├── pamr.py
│   ├── segearthov3_change_detector.py
│   ├── config_local.py
│   ├── configs/                       # stays logically grouped with fork code
│   └── sam3/
├── teammate/                          # already gitignored, left as-is, untouched
├── archive/                           # kept as-is for historical old-fork snapshot
│   └── segearth-ov3-old/
├── notebooks/
├── src/
├── results/
└── data/
```

Naming scheme: `YYYY-MM-DD_vN_short-slug.md`, date = file's last-modified date (from `ls -la`,
already gathered this session), version = 1 unless the same topic has multiple revisions.

### Open questions before executing (must be answered by user, not assumed)

1. Should `configs/` and the root-level fork `.py` files actually move under a new `fork/`
   subdirectory? This changes relative import paths used by notebooks (`classname_path='./configs/
   cls_potsdam.txt'` in `cfg_potsdam.py`, `import custom_metrics` in `eval.py`) — every notebook
   cell that does `cd /tmp/SegEarth-OV-3` and runs `eval.py configs/cfg_potsdam.py` would need its
   path updated, and the Kaggle clone step clones the **whole repo** as `/tmp/SegEarth-OV-3` — so
   fork code needs to stay at a predictable relative path from repo root, or all notebooks need
   editing. **This is the highest-risk move — needs explicit sign-off.**
2. Is it OK to consolidate the 3 byte-identical duplicate pairs (`note.md`, `plan.md`,
   `understanding.md`) into single files in `docs/research/`, with the old `archive/` and `extra/`
   copies **left in place untouched** (not deleted, just no longer the "canonical" copy)? This
   satisfies "don't delete anything" while still de-cluttering the active/canonical set.
3. Should `PROGRESS.md`/`DEVLOG.md` keep being updated at their *current* filenames (since other
   tooling/habits may reference them by name), or is a full rename to dated filenames acceptable
   even though future updates would need to go to the new name?

### Verification

- After moves: run `notebooks/NB02_potsdam_eval.ipynb`'s clone+eval cell path logic mentally (or on
  Kaggle) to confirm `configs/cfg_potsdam.py` and `eval.py` are still found at the paths the
  notebook expects, if fork code is relocated.
- Confirm `git status` shows renames (`R`) not delete+add where possible, to preserve history.
- Confirm no file present before the reorg is missing afterward — every original path should still
  resolve to *some* location (`git log --follow` should still find history for moved files).
