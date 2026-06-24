# config_local.py

import os

# --------------------------------------------------
# environment
# --------------------------------------------------

KAGGLE = (
    "KAGGLE_KERNEL_RUN_TYPE"
    in os.environ
)

# --------------------------------------------------
# inference mode
# --------------------------------------------------

RUN_SINGLE_IMAGE = True

# used only if
# RUN_SINGLE_IMAGE = True

TARGET_IMAGE = (
    "dop20_32_484_5550_1_he.jpg"
)

# --------------------------------------------------
# paths
# --------------------------------------------------

if KAGGLE:

    from pathlib import Path as _Path
    _sam3_root = _Path("/kaggle/input/sam3-weights")
    _hits = list(_sam3_root.rglob("sam3.pt")) if _sam3_root.exists() else []
    SAM3_CHECKPOINT = str(_hits[0]) if _hits else str(_sam3_root / "sam3.pt")

    INPUT_FOLDER = (
        "/kaggle/input/darmstadt-dop20/"
    )

else:

    SAM3_CHECKPOINT = (
        "weights/sam3/"
        "sam3.pt"
    )

    INPUT_FOLDER = (
        "resources/"
    )