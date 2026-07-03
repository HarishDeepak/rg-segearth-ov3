# Cyclist Pain Points

A separate, notebook-first side project inside this repo. It studies cyclist
infrastructure pain points using **OpenStreetMap road-network data only** —
it does not use orthophotos, GPUs, or SegEarth-OV-3.

## Result

For a locked study area in central Darmstadt (Rheinstrasse corridor), 12 of
28 road edges are unprotected primary/secondary roads (no `cycleway` tag).
The top-ranked pain points are on **Hügelstraße** and **Neckarstraße**.

![Pain point map](qgis/pain_points_map.png)

See `data/darmstadt_rheinstrasse_arterial_pain_points.geojson` for the full
ranked list and coordinates.

## Workflow

- **Data source**: OpenStreetMap (via OSMnx), not the DOP20 orthophoto imagery
  used elsewhere in this repo. The orthophoto only appears as a QGIS
  background layer for the validation map (see below) — it is not part of
  the analysis pipeline.
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
- **QGIS validation**: the ranked pain points are loaded into QGIS alongside
  the full road network and a Hessen DOP20 orthophoto tile
  (`dop20_32_474_5524_1_he`, covering the Rheinstrasse bbox), styled by
  `pain_score` on a graduated red ramp, and exported as a single map image.

## Outputs

- `data/*.csv`, `data/*.geojson` — tagged OSM road edges for each locked bbox
- `data/*_pain_points.geojson` — top-N ranked unprotected edges
- `results/*.html` — Folium preview map (red = unprotected, blue = protected)
- `qgis/cyclist_painpoints.qgz` — QGIS project with both GeoJSON layers styled
- `qgis/pain_points_map.png` — exported validation map (roads + orthophoto +
  ranked pain points + legend)

## Reproducing the QGIS map

The DOP20 orthophoto tile is not committed to this repo (large, not our
output, not for redistribution — see `.gitignore`). To reopen
`qgis/cyclist_painpoints.qgz` with the background imagery intact, download
the Hessen DOP20 tile `dop20_32_474_5524_1_he` (`.jpg` + `.jgw`) and place it
anywhere on disk, then re-link it in QGIS if the path has moved. Without the
tile, the project still opens fine with just the road/pain-point layers.

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
  pain-point scoring/ranking (done)
- **Phase 3** — QGIS validation: styled map with orthophoto background,
  legend, exported image (done)
- **Phase 4 (optional, later)** — fuse with SegEarth-OV-3 segmentation output
  from the main repo for a combined OSM + imagery pain-point signal
