"""OSM pseudo-ground-truth evaluation for Darmstadt/Hessen DOP20.

Adapted from rg-geoprompt-peft/src/rg_geoprompt/osm_eval.py.
Standalone — no rg_geoprompt imports.

OSM → 6-class Potsdam/Hessen mapping:
    building=*          → class 2 (building in Hessen, 1 in Potsdam)
    highway=*           → class 3 (road in Hessen, 0 in Potsdam)
    natural=wood/forest → class 1 (forest in Hessen, 3 in Potsdam)
    landuse=grass       → class 0 (agricultural/grass in Hessen, 2 in Potsdam)
Pixels with no OSM coverage → 255 (unknown, excluded from F1).

Always call results "OSM pseudo-GT F1" — never plain "GT F1".
"""
from typing import Dict, Optional

import numpy as np

OSM_EROSION_KERNEL = 3
DOP20_CRS_EPSG = 25832
OSM_TAGS = {
    "building": True,
    "highway": True,
    "natural": ["wood", "grassland"],
    "landuse": ["grass", "forest", "farmland"],
    "water": True,
}


# ---------------------------------------------------------------------------
# OSM download + rasterize
# ---------------------------------------------------------------------------

def download_osm(place: str = "Darmstadt, Germany", tags: Optional[dict] = None):
    """Download OSM features and reproject to EPSG:25832. Requires osmnx."""
    import osmnx as ox
    gdf = ox.features_from_place(place, tags or OSM_TAGS)
    return gdf.to_crs(epsg=DOP20_CRS_EPSG)


def _osm_class_geoms_hessen(gdf):
    """Split GeoDataFrame into per-class geometry lists (Hessen 6-class scheme).

    Class order: 0=agri, 1=forest, 2=building, 3=road, 4=water, 5=background
    Burn order: agri(0) → forest(1) → road(3) → building(2) last (most reliable).
    """
    out = {}
    if "landuse" in gdf.columns:
        agri = gdf[gdf.get("landuse", "").isin(["farmland", "grass", "meadow"])]
        out[0] = list(agri.geometry.dropna())
    if "natural" in gdf.columns or "landuse" in gdf.columns:
        wood = gdf[(gdf.get("natural", "") == "wood") |
                   (gdf.get("landuse", "") == "forest")]
        out[1] = list(wood.geometry.dropna())
    if "highway" in gdf.columns:
        hw = gdf[gdf["highway"].notna()]
        out[3] = [g.buffer(4.0) if g.geom_type == "LineString" else g
                  for g in hw.geometry.dropna()]
    if "building" in gdf.columns:
        out[2] = list(gdf[gdf["building"].notna()].geometry.dropna())
    if "water" in gdf.columns or "natural" in gdf.columns:
        water = gdf[(gdf.get("natural", "") == "water") |
                    (gdf.get("water", "") != "")]
        out[4] = list(water.geometry.dropna())
    return out


def rasterize_osm_patch(gdf, transform, shape=(512, 512),
                        class_geoms=None, scheme="hessen") -> np.ndarray:
    """Rasterize OSM vectors onto a single patch pixel grid."""
    from rasterio import features as rio_features

    mask = np.full(shape, 255, dtype=np.int64)
    if class_geoms is None:
        class_geoms = _osm_class_geoms_hessen(gdf)
    burn_order = sorted(class_geoms.keys())  # later = higher priority
    # buildings (2) always burned last for highest priority
    if 2 in burn_order:
        burn_order.remove(2)
        burn_order.append(2)
    for cls in burn_order:
        if class_geoms[cls]:
            rio_features.rasterize(
                ((g, cls) for g in class_geoms[cls]),
                out=mask, transform=transform,
            )
    return mask


def build_osm_masks(gdf, transforms: dict, stems, shape=(512, 512)) -> dict:
    """Rasterize + erode OSM masks for all patches.

    Args:
        transforms: {stem: affine.Affine or list of 6 floats}
    Returns:
        {stem: eroded int16 mask [H, W]}
    """
    from affine import Affine

    class_geoms = _osm_class_geoms_hessen(gdf)
    masks = {}
    for stem in stems:
        tf_raw = transforms[stem]
        tf = Affine(*tf_raw) if not isinstance(tf_raw, Affine) else tf_raw
        raw = rasterize_osm_patch(gdf, tf, shape=shape, class_geoms=class_geoms)
        masks[stem] = erode_osm_labels(raw)
    return masks


# ---------------------------------------------------------------------------
# Erosion
# ---------------------------------------------------------------------------

def erode_osm_labels(osm_label: np.ndarray,
                     kernel_size: int = OSM_EROSION_KERNEL) -> np.ndarray:
    """Mark uncertain border pixels as 255 via morphological erosion.

    Source: confirmed handoff spec (3×3 kernel, 1 iteration).
    """
    from scipy.ndimage import binary_erosion
    eroded = osm_label.copy()
    iterations = max(1, (kernel_size - 1) // 2)
    unique_cls = [c for c in np.unique(osm_label) if c != 255]
    for cls in unique_cls:
        cls_mask = (osm_label == cls)
        eroded_mask = binary_erosion(cls_mask, iterations=iterations)
        eroded[~eroded_mask & cls_mask] = 255
    return eroded


# ---------------------------------------------------------------------------
# F1 computation
# ---------------------------------------------------------------------------

def f1_from_arrays(preds, labels, num_classes: int = 6,
                   ignore_id: int = 255) -> np.ndarray:
    """Per-class F1 from prediction/label arrays (numpy).

    Returns:
        float32 array [num_classes]
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    valid = labels != ignore_id
    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)
    for cls in range(num_classes):
        pc = (preds == cls) & valid
        lc = (labels == cls) & valid
        tp[cls] = (pc & lc).sum()
        fp[cls] = (pc & ~lc).sum()
        fn[cls] = (~pc & lc).sum()
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    return (2 * prec * rec / (prec + rec + 1e-6)).astype(np.float32)


def osm_pseudo_gt_f1(pred_masks: Dict[str, np.ndarray],
                     osm_masks: Dict[str, np.ndarray],
                     num_classes: int = 6) -> dict:
    """Average per-patch F1 of predictions vs eroded OSM masks.

    Patch-based — never stitches. Report as 'OSM pseudo-GT F1'.

    Returns:
        {"per_class_f1": float32 [num_classes], "mean_f1": float, "n_patches": int}
    """
    stems = sorted(set(pred_masks) & set(osm_masks))
    if not stems:
        raise ValueError("no overlapping patch stems between preds and OSM masks")
    f1s = np.stack([f1_from_arrays(pred_masks[s], osm_masks[s], num_classes)
                    for s in stems])
    per_class = f1s.mean(axis=0)
    return {
        "per_class_f1": per_class,
        "mean_f1": float(per_class.mean()),
        "n_patches": len(stems),
    }


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_osm_masks(masks: dict, path) -> None:
    np.savez_compressed(str(path), **{k: v.astype(np.int16) for k, v in masks.items()})


def load_osm_masks(path) -> dict:
    data = np.load(str(path))
    return {k: data[k].astype(np.int64) for k in data.files}
