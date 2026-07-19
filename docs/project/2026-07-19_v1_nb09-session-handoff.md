# NB09 session handoff — 2026-07-19

Read this first before continuing NB09 work in a new session/terminal. This
captures everything from a long prior session so nothing has to be
re-derived or re-discovered.

## 1. Branch/repo state

- Work happened on **`nb09-overnight-iteration`** branch (cleanly rebased
  off `master`, **not** `cyclist-painpoints` — an early mix-up where 4
  commits accidentally landed on `cyclist-painpoints` was caught and fixed:
  those commits were moved to a new branch off `master`, and
  `cyclist-painpoints` was reset back to `origin/cyclist-painpoints` with
  its own pending edits (README rewrite, archive reorg) restored via
  stash-pop).
- All commits on `nb09-overnight-iteration` are **local only, never
  pushed** to origin. Push when ready, not automatically.
- Commit history (chronological): output-format overhaul → footer-clipping
  fix + failed runway-prompt-strengthen attempt → runway/road
  class-confusion fix + inference/render architecture split → session
  summary doc → CLAUDE.md doc update → tile-4 addition (solar farm,
  direct comparison tile) → dataset-attachment fix for tile 4 →
  tile-2-second-kernel setup → tile-2 grass-prompt fix → main-kernel
  switch to tile 3 with pre-emptive road/grass fix → CLAUDE.md NB09
  section → devlog corrections (platform-not-out-of-bounds, deliverable
  framing) → **most recent local edit, uncommitted**: tile 2's railway
  shortened, platform re-added with position-based wording, tree→canopy
  fix applied to tiles 2 and 3 (see §6).

## 2. Architecture decisions (still valid — do not re-decide)

- **Inference/rendering split.** `run_image()` in the notebook's main code
  cell (cell index 11 in `notebooks/NB09_zeroshot_sensitivity.ipynb`) only
  runs SAM3 and saves `output/preds/{stem}_{tag}.npy` +
  `output/manifest.json` + `output/preds/{stem}_meta.json` (display
  labels, colors, img_size) — **no plotting**. The next cell (index 13,
  notebook id `722e5f9f`) is a separate GPU-free rendering cell that reads
  those files and produces images. Edit that cell alone and rerun it for
  any legend/layout/color-only fix — **never requires a Kaggle rerun** for
  cosmetic-only changes, only for changes to what SAM3 actually detects
  (prompts, thresholds, window size).
- **Two Kaggle kernels run in parallel** (Kaggle allows up to 2 concurrent
  GPU sessions per account, but only 1 active version per kernel object —
  a second, differently-named kernel is required for real concurrency):
  - `harish77718/nb09-zero-shot-sensitivity` — main kernel, push folder
    `notebooks/push/nb09/`
  - `harish77718/nb09-tile2-475-5550` — second kernel, push folder
    `notebooks/push/nb09-tile2/`
  Both copies must be manually kept in sync with
  `notebooks/NB09_zeroshot_sensitivity.ipynb` (`cp` after any shared-code
  edit). **Always `grep -c <fresh-edit-marker> <push-folder-file>` before
  every `kaggle kernels push`** to confirm the sync actually took — a
  stale copy silently reruns old code with no error (this happened once:
  v10 wasted ~10 minutes running stale tile-2 code because the push folder
  wasn't synced before pushing).
- **Rendering layout matches the team's own exact plotting code**,
  obtained verbatim via chat and applied line-for-line: `figsize=(20,12)`,
  `plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.15,
  wspace=0.01)`, legend `loc='lower center', bbox_to_anchor=(0.5,0.075),
  frameon=False, fontsize=9, prop={'weight':'bold'}`, a light-gray
  separator `Line2D` at y=0.055, `meta_text` footer format, `dpi=200,
  bbox_inches='tight'`. Confirmed working correctly on tile 2's v2 output.
- **`DEFAULT_BASELINE` + per-tile `cfg["baseline"]` override.** Most tiles
  use the shared default (`prob_thd=0.1, confidence_threshold=0.1,
  slide_crop=1024, slide_stride=768`, NB03's known-good setting). Only
  tile 4 (solar farm) currently overrides this, using the team's exact
  published params (`prob_thd=0.05, confidence_threshold=0.35,
  slide_crop=1500, slide_stride=1200`) for direct comparability with their
  own result on that tile.
- **No downsampling anywhere in NB09's pipeline** — confirmed via code
  search (no `resize`/`downscale`/`max_dim` call exists). This matches the
  team's own base-script constraint (they only downscale above 2048px in
  a *different* script, not the one we're comparing against) — NB09 always
  runs at native resolution.

## 3. The core recurring finding — read this before writing any prompt

**A generic/permissive class prompt (bare "road", "grass", etc.) tends to
WIN the pixel-wise argmax across broad grey or green areas, starving a
more specific, narrower class down to 0 px.** This looked "plausible" at a
glance every single time — the only way it was actually caught was
checking `np.unique(pred, return_counts=True)` on the saved `.npy`
prediction. **Always verify this way, never by visual impression alone.**

Confirmed instances:
- **Tile 1** (`dop20_32_468_5543_1_he`, Frankfurt airport): `road`
  swallowed `runway` completely (0 px). Strengthening the runway prompt
  was tried twice (more synonyms, more texture cues) and had **zero
  effect** both times — the fix was narrowing `road`'s wording (added lane
  markings/curbs/traffic cues that a runway lacks), not strengthening
  `runway`. Lesson: when a class is starved to 0%, check what's *winning*
  those pixels, not just what's losing them.
- **Tile 2** (`dop20_32_475_5550_1_he`, Frankfurt Hbf): `grass` swallowed
  `railway`/`platform`/`road` (all 0 px) via the same mechanism. Narrowed
  grass's wording (added color/texture specificity: "bright green grass
  blades", "mowed grass"). Then, inspecting the Part-C prompt-comparison
  panel directly, found that **`railway` alone as a bare single-word
  prompt got 10.9% detection where the long multi-synonym version got
  0%** in the same run — the *opposite* of tile 1's runway lesson (where
  multi-synonym worked and single-word didn't). **There is no universal
  "always longer" or "always shorter" rule — it is genuinely per-class,
  and has to be tested, not assumed.**
- **Tile 4** (`DOP20_32_525_5604_1_he`, solar farm): no such failure —
  clean detection across every class (building 0.5%, grass 61.6%, road
  20.1%, trees 5.4%, solar panel 12.1%, background only 0.3%). Useful as
  the "control" example of what a working baseline looks like when
  prompts are already scene-appropriate from the start.

## 4. User's corrections/framing shifts (do not re-litigate these)

- **Platform is NOT out of frame in tile 2.** User directly examined the
  source image and confirmed the train shed's arched roofs plus multiple
  light-grey platform islands between the dark rail tracks are clearly
  visible. The earlier "maybe it's just not in this crop" theory was
  wrong — it's a genuine detection failure, not a scene-content gap.
- **Tile 2's lighting**: shot in strong sun with overexposure/shadow.
  User's theory (not yet independently verified, but a stronger
  explanation than "the wording was vague"): road, rail ballast, and
  platform surfaces likely sit in a visually similar washed-out light-grey
  tonal range in *this specific image*, not just similar-sounding words.
  This suggests future prompt wording for this tile should lean on
  **position/geometry cues** (e.g. platform = "long narrow island between
  parallel tracks") rather than color/texture cues that may not be
  reliably visible in this lighting.
- **Tile 2 confirmed: no water present, vehicles present but small and
  sparse.**
- **Deliverable framing correction (important — reread if in doubt):** the
  actual goal of NB09 is **not** to hand-tune each tile until it looks
  clean. It's to produce a **documented, evidence-based comparison** of
  which knob (prompt style, window size, thresholds) helps or hurts for
  which class/scene, with the rendered images as proof artifacts. Whether
  single-word or multi-synonym prompts work better is supposed to be an
  *answered, evidenced* question per class by the end of this work — not
  a default picked once and applied everywhere.
- **Methodology going forward (approved earlier in this session, still the
  standing instruction):**
  1. Ask what's actually in the tile before writing/editing a prompt for
     it — don't infer content from the filename or a small thumbnail.
  2. Change one axis at a time (prompt wording, OR window size, OR
     thresholds) rather than stacking several unconfirmed changes into one
     push — except for confirmed, already-validated fixes (e.g. reapplying
     the road/grass narrowing pattern that's already proven on 2 tiles),
     which are fine to apply pre-emptively.
  3. Verify every fix via the pixel histogram, not visual impression.
  4. Apply confirmed external lessons immediately rather than waiting to
     rediscover them ourselves (e.g. "tree" → "tree canopy" is already
     demonstrated in the team's own reference material).
- **"Tree canopy" is explicitly a worked EXAMPLE of the methodology, not
  just a one-off patch.** The point of documenting it is to show the
  *mechanism*: plain "tree" describes a single discrete countable object
  and matches an isolated tree fine, but fails on a dense forest because a
  forest isn't visually "many trees," it's a continuous textured surface —
  "tree canopy" matches that continuous-texture framing instead. This is
  the template for investigating every future prompt failure: understand
  *why* it failed and *why* the fix addresses that specific mechanism, not
  just "try a different phrase until it works."
- **Last open question, unresolved when the session was interrupted:**
  user wants a "proof" example for each fix (tree canopy, railway wording,
  road/grass narrowing, platform) but explicitly rejected a leading
  multiple-choice framing of what "proof" should consist of (rejected:
  "should every fix get its own rendered before/after image pair, or is
  the devlog write-up + Image C sweep enough"). **Ask this open-ended in
  the new session — do not re-guess at options.** Simply ask: "what do you
  mean by 'example to prove' for each fix — what should that proof look
  like or contain?"

## 5. Reference material already mined (do not re-fetch or re-analyze)

- Google Drive folder `1D9lTjipYNesV2XzORCKxSE3dR25abnLb` ("Hessen") is
  the exact same content as the local folder
  `D:\Downloads\drive-download-20260718T135518Z-1-001\` — already examined
  in full detail: subfolders `D0`, `D1`, `F-5546`, `F-5550`, plus several
  named single-tile folders. Includes a `windowsize_exp` sweep on tile
  `472_5525` (crop/stride from 400/350 up to 1500/1500): 400/350 was
  visibly noisier; 1500/1500 left a black dropout hole in the forest class
  until the prompt was changed from `"tree"` to `"dense tree canopy"`,
  which fixed the hole **at the same crop/stride** — direct proof the fix
  was prompt-driven, not crop-driven.
- The team's exact base script/params were shared via chat and
  cross-checked against this repo's actual code:
  - `max_dim=2048` downscale-guard snippet, and a `configs/custom.yaml`
    class-config structure (`name`/`color`/`is_background` fields) — not
    adopted verbatim in NB09, but confirmed NB09 already satisfies the
    "no downsampling" constraint without needing to change anything.
  - **The team's `conf_thd`/`prob_thd` interaction claim does NOT match
    this repo's actual code.** They described `confidence_threshold` as
    multiplying into the prob score, able to "rescue" a near-threshold
    detection. Verified via an Explore agent reading
    `sam3/model/sam3_image_processor.py`
    (`Sam3Processor._forward_grounding`, ~lines 198-236):
    `confidence_threshold` is a **hard pre-filter**
    (`keep = out_probs > self.confidence_threshold`) applied once, *before*
    `object_score`/`masks_logits`/`presence_score` are ever populated —
    it never multiplies into a later score. `prob_thd` is a fully
    separate, later floor applied in `segearthov3_segmentor.py`
    (`seg_pred[max_vals < prob_thd] = bg_idx`). No shared arithmetic
    exists between the two in this codebase. This may be worth flagging
    back to the team in case it's a fork/version difference or a
    misremembered explanation, since it could cause confusion in their
    presentation if the mental model is wrong.
  - The team's exact rendering/plotting code (see §2) — adopted verbatim.

## 6. Tile-by-tile status (as of session interruption, 2026-07-19 ~13:00)

| # | Tile | Kernel | Status at interruption | Key notes |
|---|---|---|---|---|
| 1 | `dop20_32_468_5543_1_he` (Frankfurt airport) | main | Done, fully iterated (v5-v8) | Runway detection still imperfect after 2 strengthen attempts on the runway prompt itself; the working fix was narrowing `road`. Format/layout fully matches team style (v8 baseline). |
| 2 | `dop20_32_475_5550_1_he` (Frankfurt Hbf) | tile2 kernel | **v3 COMPLETED** (grass-fix-only run) — not yet pulled/reviewed as of interruption | A newer local edit exists **uncommitted, unpushed** in `notebooks/NB09_zeroshot_sensitivity.ipynb`: railway shortened to `"railway track, train tracks, rail line"`; platform **re-added** (was dropped in the v3 push) with position-based wording `"long narrow platform island between parallel railway tracks, station platform"`; tree changed to `"dense tree canopy, tree"`. **This edit must not be lost** — verify it's still in the file, then commit + sync push folder + push as the next iteration once confirmed. |
| 3 | `dop20_32_476_5524_1_he` (Darmstadt) | main kernel | v13 running when interrupted (check status first) | Pre-emptive road+grass narrowing applied before its first run (reusing the confirmed tile 1/2 pattern — this is explicitly allowed by the methodology since it's a validated fix, not a new guess). Tree updated to `"dense tree canopy, tree, forest, ..."`. **User has not yet confirmed this tile's actual content** — the water/sports/residential assumption comes from the filename/original config comment only. Ask before further prompt work on this tile, per the methodology. |
| 4 | `DOP20_32_525_5604_1_he` (solar farm) | main kernel (earlier) | Complete, clean, no issues | Used as the reference "this is what working looks like" example. Rendered with the OLD subplot layout (before the exact-match-code rendering fix landed) — cosmetic gap only. Not re-rendered because the source `.jpg` tile isn't available locally (only the team's own pre-rendered PNG is) and inference/rendering being decoupled means this doesn't block anything — just note it if a clean image is needed for a report. |

**Immediate next step recommended:** pull and review tile 2's v3 output
(already completed), check tile 3's current status, and decide whether to
commit+push the pending railway/platform/tree-canopy edit as a fresh tile-2
iteration before or after reviewing v3's results.

## 7. Kaggle mechanics learned this session (do not rediscover)

- **No API-level stop/cancel for a running kernel exists** — checked
  exhaustively (CLI, kagglesdk, public API docs, Kaggle's own product-
  feedback board confirms it's a standing feature request). Only the
  website's Stop button works. Wait it out, or ask the user to cancel
  manually. **Never use `kernels delete` as a workaround** — it destroys
  the kernel's entire version history, far more destructive than needed.
- `kaggle kernels push` while already at the account's 2-GPU-session cap
  errors immediately with `"Maximum batch GPU session count of 2
  reached"` rather than queuing — but a version that's already running
  keeps running even if push fails, so a stuck/orphaned session (like v10
  was) can silently keep consuming a GPU slot until manually stopped.
- The only way to get genuine two-kernel concurrency is a **second,
  differently-named kernel** — separate push folder + separate
  `kernel-metadata.json`, not a second version of the same kernel.
- `dataset_sources` in `kernel-metadata.json` must explicitly list every
  Kaggle dataset a tile's `find_tile()` search needs — a missing entry
  produces a clean `NOT FOUND` failure at *runtime* (after full conda
  setup, ~10+ min in), not a push-time error. This happened for tile 4's
  `vogelsbergkreis-lautertal-dop20` dataset (v11) — fixed by adding the
  slug and re-pushing as v12.
- Conda/network errors during environment setup
  (`CondaHTTPError: HTTP 000 CONNECTION FAILED for url
  <https://repo.anaconda.com/...>`) are a real, occasionally-occurring
  Kaggle infrastructure issue, not a bug in NB09's code — happened once on
  tile2 kernel's v1, resolved with a plain retry (v2).

## 8. Splitting work across terminals/sessions (user's stated preference)

User wants routine **kernel-status monitoring** (cheap, repetitive) kept
separate from the actual **reasoning/prompt-engineering work** (expensive:
reading images, editing prompts, writing devlog entries) to conserve
tokens. In practice this session already did the right thing for
monitoring — the `Monitor` tool watches `kaggle kernels status <name>` in
a background loop and only notifies on completion/error, which doesn't
consume main-session tokens between notifications. Continue using `Monitor`
for this rather than manually re-polling with `Bash` in the main thread.
For heavier, genuinely self-contained sub-tasks (e.g. "pull this kernel's
output and report the pixel histogram" with no back-and-forth needed), a
background `Agent` call is a reasonable delegation target — but most of
this session's actual fixes required real back-and-forth (confirming image
content, methodology decisions), so keep those in the main conversation
rather than delegating prematurely.

## 9. Key file locations

- `notebooks/NB09_zeroshot_sensitivity.ipynb` — source of truth, edit here
- `notebooks/push/nb09/NB09_zeroshot_sensitivity.ipynb` — main kernel's
  push copy, sync from above before pushing
- `notebooks/push/nb09/kernel-metadata.json` — main kernel's dataset
  sources / config
- `notebooks/push/nb09-tile2/NB09_zeroshot_sensitivity.ipynb` — tile2
  kernel's push copy
- `notebooks/push/nb09-tile2/kernel-metadata.json` — tile2 kernel's config
- `docs/project/2026-07-18_v1_nb09-overnight-iteration-log.md` — the
  detailed, chronological devlog (iteration-by-iteration reasoning) —
  this handoff doc is a *summary*, the devlog has the full narrative
- `docs/project/2026-07-18_v1_nb09-run-params.csv` — per-run parameter log
- Downloaded kernel outputs live under `notebooks/push/nb09/output_v*/`
  and `notebooks/push/nb09-tile2/output_v*/` — pulled via
  `kaggle kernels output <name> -p <dir>`, one folder per pulled version
