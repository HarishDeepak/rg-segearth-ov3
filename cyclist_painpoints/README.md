# Cyclist Pain Points

A separate, notebook-first side project inside this repo. It studies cyclist
infrastructure pain points using **OpenStreetMap road-network data only** —
it does not use orthophotos, GPUs, or SegEarth-OV-3.

## Workflow

- **Data source**: OpenStreetMap (via OSMnx), not the DOP20 orthophoto imagery
  used elsewhere in this repo.
- **Study area**: a single bbox locked in `notebooks/phase2_osm_bbox.ipynb`,
  kept fixed so outputs stay comparable across runs.
- **Compute**: CPU-only network analysis, no model inference.
- **Logic**: road edges are flagged `unprotected` when `highway` is
  primary/secondary/tertiary and `cycleway` is missing or `no`.

## Outputs

- `data/*.csv`, `data/*.geojson` — tagged OSM road edges for the locked bbox
- `results/*.html` — Folium preview map (red = unprotected, blue = protected)

## Phases

- **Phase 1** — repo/folder scaffold, dependency setup (done)
- **Phase 2** — OSM pull, highway/cycleway tagging, unprotected flagging (current)
- **Phase 3** — only if needed later: more study areas or deeper analysis
