# rg-segearth-ov3

Open-vocabulary remote sensing segmentation (Fraunhofer IGD Praktikum, SoSe 2026).
Inference-only pipeline using **SegEarth-OV-3** (SAM3 + open-vocab text prompts) on
Hessen DOP20 (20cm GSD) and Potsdam ISPRS (5cm GSD, for quantitative baseline).

SAM3 is fully frozen — no training. See `CLAUDE.md` for full project context,
confirmed results, and notebook layout.

## Sub-projects

- **[cyclist_painpoints/](cyclist_painpoints/)** — a separate, OSM-only side
  project that scores and ranks cyclist infrastructure pain points (unprotected
  primary/secondary/tertiary roads) from OpenStreetMap road-network data,
  validated in QGIS. CPU-only, does not use SegEarth-OV-3 or orthophoto imagery
  in its analysis pipeline.
