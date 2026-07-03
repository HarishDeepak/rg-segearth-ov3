# Interview Notes: Cyclist Pain Points

A concept-by-concept explainer for talking through this project out loud —
what each piece is, why it was chosen, and the honest tradeoffs. Written so
you can explain it without re-deriving anything under pressure.

---

## 1. The one-sentence pitch

"I built a small pipeline that pulls a city's road network from
OpenStreetMap, flags roads that carry fast traffic but have no dedicated
cycling infrastructure, ranks them by how bad the exposure is, and validates
the result visually in QGIS against aerial imagery."

If asked "why does this matter": road authorities can't manually audit every
street for cycling safety. This gives a cheap, fast, explainable first pass
at *where to look* — not a final verdict, a triage tool.

---

## 2. What is OpenStreetMap (OSM), and why use it

OSM is a free, crowd-sourced map of the world — every road, building, and
path is an editable object with **tags** (key-value pairs), e.g.
`highway=primary`, `cycleway=lane`, `name=Rheinstraße`. Anyone can edit it;
that's both its strength (huge, free, current coverage) and its weakness
(inconsistent tagging quality depending on who mapped an area and when).

**Why OSM instead of official government road data**: it's free, has a
mature open-source toolchain (OSMnx), already encodes cycling infrastructure
as a tag (`cycleway`), and covers essentially the whole world in one
consistent schema — official data would vary by country/city and often
requires licensing.

## 3. What is OSMnx

A Python library that wraps the **Overpass API** (OSM's query service) and
converts the result into a **NetworkX graph** — nodes are intersections,
edges are road segments between them. It also converts that graph into
**GeoDataFrames** (GeoPandas tables with a geometry column), which is what
this project actually works with.

**Why a graph, not just a list of roads**: because roads connect at
intersections, a graph naturally represents that topology (which is useful
if you ever want to do routing later — not done here, but it's why the
library is graph-shaped).

## 4. What is the Overpass API, and why it was flaky

Overpass is OSM's query backend — you send it a bounding box and a filter
(e.g. "give me all `highway` ways"), it returns matching OSM data as JSON.
It's a shared public service with a handful of volunteer-run mirrors
(kumi.systems, overpass-api.de, etc.), and during this project it was
frequently overloaded — queries would hang or time out independent of query
size. This is a known characteristic of the public Overpass infrastructure,
not a bug in the code. **Interview framing**: "I hit real-world API
reliability issues and worked around them with retries and by verifying the
underlying fetch functions worked correctly via standalone reproduction,
rather than assuming code was broken."

## 5. Why lock a bounding box (bbox)

A bbox is just `(west, south, east, north)` in latitude/longitude — the
rectangle defining the study area. Locking it means picking one and not
changing it while iterating, so that every run of the notebook is comparable
(you're not accidentally comparing apples to oranges because the study area
silently shifted). This project actually used **two** bboxes: one
residential-only (a sanity check — confirms the code doesn't just flag
everything) and one arterial (Rheinstraße corridor — where the real ranked
result comes from).

## 6. The `highway` tag and road classification

OSM classifies roads by a `highway` tag using a rough functional hierarchy:
`motorway > trunk > primary > secondary > tertiary > residential > service`.
Primary/secondary/tertiary roughly correspond to "carries through-traffic,
not just local access" — these are the roads most likely to have faster,
heavier traffic that's dangerous to share with cyclists without protection.
Residential and service roads are excluded from the "unprotected" definition
on purpose — they're already lower-speed, lower-volume by design.

## 7. The `cycleway` tag and what "unprotected" means here

`cycleway` (and its directional variants `cycleway:left`/`cycleway:right`)
describes what cycling infrastructure exists on a road: `lane` (painted bike
lane), `track` (physically separated path), `shared_lane`, `no`, or simply
absent (never tagged).

**The rule this project uses**:
```
unprotected = highway in {primary, secondary, tertiary}
              AND cycleway is missing or == "no"
```

This is a **binary proxy**, not a nuanced safety score. It says nothing
about actual traffic speed, volume, sight lines, or parking-related dooring
risk — only "is this a busy-class road with no formal cycling
infrastructure tagged." Be upfront about this limitation if asked.

## 8. Why score and rank instead of just listing unprotected roads

A binary "unprotected: yes/no" flag doesn't tell you which unprotected road
is worse. The scoring step adds a simple weighting:

```
pain_score = weight[highway_class] * unprotected_length_m
weight = {primary: 3.0, secondary: 2.0, tertiary: 1.0}
```

**Why length matters**: a 500m unprotected primary road is a bigger problem
than a 20m unprotected connector. **Why road class is weighted**: primary
roads generally carry faster/heavier traffic than tertiary, so the same
unprotected length is riskier on a primary road. **Why these specific
weights (3/2/1)**: a simple, defensible, round-number choice — explicitly
*not* a fitted or calibrated model. If asked "how would you improve this,"
the honest answer is: calibrate against real traffic volume, speed limit,
or accident data if it's available, rather than a hand-picked multiplier.

## 9. GeoPandas, GeoDataFrames, and geometry types

**GeoPandas** = pandas + a `geometry` column that holds actual shapes
(points, lines, polygons) using the `shapely` library underneath. A road
segment here is a `LineString` — an ordered sequence of (lon, lat) points.
This project's `edges` GeoDataFrame is one row per road segment, with normal
columns (`highway`, `cycleway`, `length`, `pain_score`) plus a `geometry`
column you can plot, measure, or export directly to GeoJSON.

## 10. GeoJSON — what it is and why export to it

GeoJSON is a plain-text (JSON) standard for encoding geographic features —
every feature has a `geometry` (coordinates) and `properties` (the tag
data). It's the universal interchange format for GIS tools: QGIS reads it
natively, so does virtually every mapping library (Leaflet, Folium,
Mapbox). Exporting to GeoJSON is what makes the ranked pain points
independently loadable in QGIS without needing Python or this notebook at
all.

## 11. Coordinate Reference Systems (CRS) — the one that trips people up

**This is a good "shows real GIS understanding" topic if it comes up.**

- OSM/GeoJSON data here is in **EPSG:4326** (WGS84) — plain latitude/longitude,
  the same system GPS uses. Good for storage/interchange, bad for measuring
  distance directly (degrees aren't a constant physical distance — a degree
  of longitude shrinks as you move away from the equator).
- The **DOP20 orthophoto tiles** are in **EPSG:25832** (ETRS89 / UTM zone
  32N) — a projected, metric coordinate system where 1 unit = 1 meter. This
  is why the tile's `.jgw` "world file" gives pixel-to-meter coordinates
  directly, and why QGIS needed the CRS confirmed manually the first time
  the raster was loaded (a `.jgw` file alone doesn't self-declare *which*
  CRS its numbers are in).
- QGIS reprojects "on the fly" so the WGS84 GeoJSON layers and the UTM32
  raster tile display correctly aligned on screen without you manually
  converting anything — that's why the overlay just worked once both layers
  were loaded, even though they're stored in different CRSs.
- **Why `length` in the data is in meters despite the coordinates being in
  degrees**: OSMnx internally projects the graph to a local UTM zone before
  computing edge lengths, then can convert back — the `length` column is
  already metric even though the exported `geometry` coordinates are WGS84
  lon/lat. This is a subtlety worth knowing if asked "how is length computed
  from lat/lon coordinates."

## 12. Why QGIS, and what "validation" means here

QGIS is free, open-source desktop GIS software — the standard tool for
viewing, styling, and analyzing geographic data without writing code every
time. **Why validate in QGIS instead of trusting the numbers blindly**:
seeing the ranked pain points drawn on top of the real road network and an
actual aerial photo is a sanity check — it confirms the top-ranked edges are
real, sensible roads (not, say, a mis-tagged parking lot access road), and
it produces a visual artifact that communicates the finding far faster than
a table of numbers.

**Graduated symbology** (used here): instead of one fixed color for every
feature, QGIS bins values of a chosen attribute (`pain_score`) into classes
and assigns each class a shade along a color ramp — so higher-risk edges
render visibly darker/redder than lower-risk ones. This is the standard way
to show a continuous variable on a map.

## 13. Reproducibility and honesty about the process

Two things worth mentioning proactively if asked about the workflow:

- **Overpass unreliability was worked around, not hidden.** When live
  notebook execution kept timing out, the same underlying functions were
  run as a standalone script to get real, unmodified results, which were
  then recorded as the notebook's cell outputs — documented transparently
  in the README rather than silently faking numbers.
- **A sanity-check bbox was deliberately included.** The residential-only
  bbox produces zero unprotected edges, which is the *correct* answer for
  that area (residential roads are excluded from the "unprotected"
  definition by design) — this is evidence the logic responds to real input
  rather than just flagging everything, which is exactly what you'd want to
  check before trusting a scoring pipeline.

## 14. Likely follow-up questions and how to answer them

**"How would you validate this against real accident/safety data?"**
Cross-reference the ranked edges against a public accident dataset (if
available for Hessen/Germany) filtered to cyclist-involved incidents, and
check whether high-`pain_score` edges correlate with real incident
clusters — that would move this from "plausible proxy" to "empirically
grounded."

**"What if OSM tagging is wrong or outdated for an area?"**
The method inherits OSM's data quality. A mitigation: cross-check flagged
edges against imagery (which is exactly what QGIS validation partially
does, and what fusing with SegEarth-OV-3 segmentation would do more
rigorously — checking whether the imagery shows a real bike lane that
just wasn't tagged).

**"Why not use a machine learning model for the risk score?"**
Because there's no labeled training data for "cyclist risk" in this
context, and a transparent, hand-specified formula is easier to justify,
debug, and explain to a non-technical stakeholder (e.g. a city planner)
than an opaque model — especially at this small, proof-of-concept scale.

**"How does this scale beyond one small bbox?"**
The pipeline is bbox-parameterized already — the same functions run for
any bbox. Scaling to a whole city means: looping over a grid of bboxes (to
stay within Overpass's per-query size limits), deduplicating edges that
straddle tile boundaries, and probably caching/batching Overpass requests
given its reliability issues.

**"What's the difference between this and a routing/navigation app's
'bike-friendly route'?"**
Routing optimizes a path for a specific trip; this project instead
produces a static, area-wide *prioritization* of infrastructure gaps —
closer to a planning tool for road authorities than a consumer app.
