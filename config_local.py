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
    # datasets mount under /kaggle/input/datasets/<owner>/<slug>/
    _hits = [p for p in _Path("/kaggle/input").rglob("sam3.pt") if p.is_file()]
    SAM3_CHECKPOINT = str(_hits[0]) if _hits else (
        "/kaggle/input/datasets/dummyirl/sam3-weights/sam3.pt"
    )

    INPUT_FOLDER = (
        "/kaggle/input/datasets/harish77718/"
        "darmstadt-dop20-presliced/"
    )

else:

    SAM3_CHECKPOINT = (
        "weights/sam3/"
        "sam3.pt"
    )

    INPUT_FOLDER = (
        "resources/"
    )