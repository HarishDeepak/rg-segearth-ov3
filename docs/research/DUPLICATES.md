# Duplicates manifest

This `docs/research/` folder holds a **cleaned-up, dated copy** of the scattered research notes
that used to live only in `archive/` and `extra/`. Per project policy, nothing was deleted —
the old files in `archive/` and `extra/` are untouched and still exist at their original paths.
This file just documents which new file corresponds to which old one(s), and flags exact
duplicates that existed across both folders.

## Byte-identical duplicates (confirmed via `diff`, both copies still exist untouched)

| Canonical copy (this folder) | Old copy #1 | Old copy #2 |
|---|---|---|
| `2026-06-24_v1_note.md` | `archive/note.md` | `extra/note.md` (identical) |
| `2026-06-22_v1_plan.md` | `archive/plan.md` | `extra/plan.md` (identical) |
| `2026-06-24_v1_understanding.md` | `archive/understanding.md` | `extra/understanding.md` (identical) |

## Near-duplicate (same research content, different line-wrapping — not byte-identical)

| Canonical copy (this folder) | Old copy #1 | Old copy #2 |
|---|---|---|
| `2026-06-22_v1_sam3-remote-sensing-research.md` | `archive/SAM3 Remote Sensing Segmentation Research.md` | `extra/SAM3 Remote Sensing Segmentation Research.md` (reflowed, same content) |

## Single-source files (only existed in one place)

| Canonical copy (this folder) | Original location |
|---|---|
| `2026-06-16_v1_lastclaudechat.md` | `archive/lastclaudechat.md` |
| `2026-06-27_v2_remote-sensing-segmentation-adaptation.md` | `archive/Remote Sensing Segmentation Adaptation.md` |
| `2026-06-29_v1_infoplan.md` | `extra/infoplan.md` |
| `2026-06-27_v1_plan1-handoff.md` | `extra/plan1.md` |
| `2026-06-25_v1_potsdam-darmstadt-independence-notes.md` | `extra/1.md` (was gitignored) |

## Not yet touched

- `archive/segearth03.ipynb`, `archive/segearth-ov3-old/` (old fork snapshot + its own
  `NOTES.md`/`REVIEW.md`) — left exactly as-is, not part of this cleanup pass.
- `extra/image/` — left as-is.
- `teammate/` — already gitignored, correctly separate (real git clone of
  `dummy-irl/SegEarth-OV-3`), not touched.
