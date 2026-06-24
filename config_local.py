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

    SAM3_CHECKPOINT = (
        "/kaggle/input/sam3-weights/"
        "sam3.pt"
    )

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