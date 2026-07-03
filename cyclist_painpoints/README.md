# Cyclist Pain Points

A separate, notebook-first side project inside this repo. It studies cyclist
infrastructure pain points using **OpenStreetMap road-network data only** —
it does not use orthophotos, GPUs, or SegEarth-OV-3.

## Workflow

- **Data source**: OpenStreetMap (via OSMnx), not the DOP20 orthophoto imagery
  used elsewhere in this repo.
- **Study areas**: two bboxes locked in `notebooks/phase2_osm_bbox.ipynb`.
  Section 5 is a residential-only bbox near `dop20_32_473_5521_1_he` (no
  unprotected edges by design — a baseline/sanity check). Section 5b is a
  second bbox over central Darmstadt near Rheinstrasse, which carries
  primary/secondary roads and is the one used for pain-point ranking.
- **Compute**: CPU-only network analysis, no model inference.
- **Logic**: road edges are flagged `unprotected` when `highway` is
  primary/secondary/tertiary and `cycleway` is missing or `no`.
- **Pain-point ranking**: unprotected edges are scored as
  `weight[highway] * length_m`, where primary roads carry more weight than
  secondary, and secondary more than tertiary. This is a simple, OSM-tag-only
  proxy for where to look first — not a calibrated risk probability. Fusing
  this with segmentation output from the separate SegEarth-OV-3 work in this
  repo is a natural next step, not yet implemented here.

## Outputs

- `data/*.csv`, `data/*.geojson` — tagged OSM road edges for each locked bbox
- `data/*_pain_points.geojson` — top-N ranked unprotected edges, ready to load
  into QGIS as a separate layer styled by `pain_score`
- `results/*.html` — Folium preview map (red = unprotected, blue = protected)

## Known Overpass reliability issue

The public Overpass API (all mirrors: kumi.systems, overpass-api.de, lz4,
z.overpass-api.de) is prone to congestion and read timeouts, independent of
bbox size or query complexity. If a fetch cell hangs or times out, retry it —
a failed request does not indicate a bug in the notebook. The Section 5b/7/8
outputs currently committed in the notebook were produced by an equivalent
script run using the exact same functions defined in the notebook (fetched
successfully once Overpass responded), then recorded as the notebook's cell
outputs, because in-notebook execution kept timing out during Overpass
congestion at commit time. The underlying data files on disk
(`darmstadt_rheinstrasse_arterial.*`) are the real, unmodified fetch results.

## Phases

- **Phase 1** — repo/folder scaffold, dependency setup (done)
- **Phase 2** — OSM pull, highway/cycleway tagging, unprotected flagging,
  pain-point scoring/ranking, QGIS export (done)
- **Phase 3** — only if needed later: fuse with SegEarth-OV-3 segmentation
  output, validate ranked pain points in QGIS against orthophoto imagery
