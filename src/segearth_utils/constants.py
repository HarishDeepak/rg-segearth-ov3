import numpy as np

# 6-class Hessen open-vocab (matches cls_hessen.txt line order)
HESSEN_CLASSES = [
    "agricultural field",
    "forest",
    "building",
    "road",
    "water body",
    "background",
]

HESSEN_COLOR_MAP = np.array([
    [210, 180, 140],  # agricultural field — tan
    [ 34, 139,  34],  # forest             — forest green
    [220,  20,  60],  # building           — crimson
    [105, 105, 105],  # road               — dim gray
    [ 30, 144, 255],  # water body         — dodger blue
    [200, 200, 200],  # background         — light gray
], dtype=np.uint8)

# 6-class Potsdam ISPRS (matches cls_potsdam.txt line order)
POTSDAM_CLASSES = [
    "impervious surface",
    "building",
    "low vegetation",
    "tree",
    "car",
    "clutter",
]

POTSDAM_COLOR_MAP = np.array([
    [128,   0,   0],  # impervious surface — maroon
    [  0,   0, 255],  # building           — blue
    [  0, 255,   0],  # low vegetation     — lime
    [  0, 128,   0],  # tree               — green
    [255, 255,   0],  # car                — yellow
    [255,   0,   0],  # clutter            — red
], dtype=np.uint8)
