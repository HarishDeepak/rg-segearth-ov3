from PIL import Image
from pathlib import Path

import os
import time
import torch
import subprocess
import numpy as np

# force safe matplotlib backend
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from torchvision import transforms
from mmseg.structures import SegDataSample

from segearthov3_segmentor import (
    SegEarthOV3Segmentation
)

from config_local import *


# --------------------------------------------------
# create output folder
# --------------------------------------------------

os.makedirs(
    "output",
    exist_ok=True
)

# --------------------------------------------------
# image extensions
# --------------------------------------------------

IMAGE_EXTENSIONS = [
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg"
]

# --------------------------------------------------
# find images
# --------------------------------------------------

image_paths = []

for ext in IMAGE_EXTENSIONS:

    image_paths.extend(
        Path(INPUT_FOLDER).glob(
            f"*{ext}"
        )
    )

# --------------------------------------------------
# Potsdam filtering
# --------------------------------------------------

# keep only RGB tiles for ISPRS
if "6ISPRS" in INPUT_FOLDER:

    image_paths = [
        p for p in image_paths
        if "_RGB" in p.name
    ]

image_paths = sorted(
    image_paths
)

# --------------------------------------------------
# single image mode
# --------------------------------------------------

if RUN_SINGLE_IMAGE:

    available_images = [
        p.name
        for p in image_paths
    ]

    image_paths = [
        p for p in image_paths
        if p.name == TARGET_IMAGE
    ]

    if len(image_paths) == 0:

        raise FileNotFoundError(
            f"TARGET_IMAGE "
            f"not found:\n"
            f"{TARGET_IMAGE}\n\n"
            f"Available files:\n"
            f"{available_images}"
        )

# --------------------------------------------------
# split workload across GPUs
# --------------------------------------------------

gpu_count = int(
    os.environ.get(
        "TOTAL_GPUS",
        1
    )
)

gpu_id = int(
    os.environ.get(
        "LOCAL_GPU_ID",
        0
    )
)

if gpu_count > 1:

    chunk_size = (
        len(image_paths)
        + gpu_count
        - 1
    ) // gpu_count

    start = (
        gpu_id
        * chunk_size
    )

    end = min(
        start
        + chunk_size,
        len(image_paths)
    )

    image_paths = image_paths[
        start:end
    ]

print(
    f"GPU {gpu_id}: "
    f"{len(image_paths)} images"
)

for p in image_paths:

    print(
        f"  - {p.name}"
    )

# --------------------------------------------------
# prompts
# --------------------------------------------------

# name_list = [
#     'road',
#     'building',
#     'grass',
#     'tree',
#     'car',
#     'clutter'
# ]

name_list = [
    'building',
    'road',
    'construction area',
    'parking lot',
    'grassland',
    'car',
    'tree',
    'water',
    'container',
    'clutter'
]

with open(
    './configs/my_name.txt',
    'w'
) as writers:

    for i in range(
        len(name_list)
    ):

        if i == (
            len(name_list)
            - 1
        ):

            writers.write(
                name_list[i]
            )

        else:

            writers.write(
                name_list[i]
                + '\n'
            )


# COLOR_MAP = np.array([
#     [255, 255, 255],  # 0 impervious
#     [0, 0, 255],      # 1 building
#     [0, 255, 255],    # 2 low vegetation
#     [0, 255, 0],      # 3 tree
#     [255, 255, 0],    # 4 car
#     [255, 0, 0],      # 5 clutter
# ], dtype=np.uint8)


COLOR_MAP = np.array([

    [0, 0, 255],      # 0 building (blue)

    [100, 100, 100],  # 1 road (dark gray)

    [180, 255, 0],    # 2 football field (bright green)

    [180, 180, 180],  # 3 parking lot (light gray)

    [120, 220, 120],  # 4 grassland (light green)

    [255, 255, 0],    # 5 car (yellow)

    [0, 180, 0],      # 6 tree (dark green)

    [0, 255, 255],    # 7 water (cyan)

    [255, 140, 0],    # 8 container (orange)

    [255, 0, 0],      # 9 clutter (red)

], dtype=np.uint8)

# --------------------------------------------------
# create model
# --------------------------------------------------

model = (
    SegEarthOV3Segmentation(
        type='SegEarthOV3Segmentation',
        model_type='SAM3',
        classname_path=
        './configs/my_name.txt',
        prob_thd=0.1,
        bg_idx=9,
        confidence_threshold=0.1,
        slide_stride=768,
        slide_crop=1024,
    )
)

# --------------------------------------------------
# loop images
# --------------------------------------------------

for idx, img_path in enumerate(
    image_paths,
    1
):

    start_time = (
        time.time()
    )

    print("=" * 60)

    print(
        f"[{idx}/"
        f"{len(image_paths)}] "
        f"{img_path.name}"
    )

    # ---------------------------
    # load RGB
    # ---------------------------

    img = Image.open(
        img_path
    ).convert("RGB")

    img_tensor = (
        transforms.Compose([
            transforms.ToTensor(),
        ])(img)
        .unsqueeze(0)
        .to('cuda')
    )

    # ---------------------------
    # metadata
    # ---------------------------

    data_sample = (
        SegDataSample()
    )

    img_meta = {
        'img_path':
        str(img_path),

        'ori_shape':
        img.size[::-1]
    }

    data_sample.set_metainfo(
        img_meta
    )

    print(
        "Running prediction..."
    )

    # ---------------------------
    # predict
    # ---------------------------

    seg_pred = (
        model.predict(
            img_tensor,
            data_samples=[
                data_sample
            ]
        )
    )

    seg_pred = (
        seg_pred[0]
        .pred_sem_seg
        .data
        .cpu()
        .numpy()
        .squeeze(0)
    )

    # ---------------------------
    # convert colors
    # ---------------------------

    seg_rgb = COLOR_MAP[
        np.clip(
            seg_pred,
            0,
            len(COLOR_MAP) - 1
        )
    ]

    # ---------------------------
    # load GT (optional)
    # ---------------------------

    gt_img = None

    # only for Potsdam
    if "_RGB" in img_path.name:

        base_name = (
            img_path.name
            .split("_RGB")[0]
        )

        gt_path = (
            img_path.parent /
            f"{base_name}_label_noBoundary.tif"
        )

        if gt_path.exists():

            gt_img = Image.open(
                gt_path
            )

    # ---------------------------
    # visualize
    # ---------------------------

    if gt_img is not None:

        fig, ax = plt.subplots(
            1,
            3,
            figsize=(24, 8)
        )

    else:

        fig, ax = plt.subplots(
            1,
            2,
            figsize=(18, 8)
        )

    # RGB
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title("RGB Image")

    # prediction overlay
    ax[1].imshow(img)

    ax[1].imshow(
        seg_rgb,
        alpha=0.45
    )

    ax[1].axis('off')

    ax[1].set_title(
        "Segmentation Overlay"
    )

    # GT (optional)
    if gt_img is not None:

        ax[2].imshow(
            gt_img
        )

        ax[2].axis('off')

        ax[2].set_title(
            "Ground Truth"
        )

    # ---------------------------
    # legend
    # ---------------------------

    legend_elements = []

    for class_name, color in zip(
        name_list,
        COLOR_MAP
    ):

        legend_elements.append(
            Patch(
                facecolor=color / 255.0,
                edgecolor='black',
                label=class_name
            )
        )

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True
    )

    plt.tight_layout(
        rect=[0, 0.08, 1, 1]
    )

    # ---------------------------
    # save image
    # ---------------------------

    output_path = (
        "output/"
        f"{img_path.stem}"
        "_segmented.png"
    )

    plt.savefig(
        output_path,
        bbox_inches='tight'
    )

    # show only first image
    if idx == 1:

        plt.show()

    plt.close()

    elapsed = (
        time.time()
        - start_time
    )

    # ---------------------------
    # GPU usage
    # ---------------------------

    gpu_info = (
        subprocess
        .check_output([
            "nvidia-smi",
            "--query-gpu="
            "utilization.gpu,"
            "memory.used,"
            "memory.total",
            "--format="
            "csv,noheader,"
            "nounits"
        ])
        .decode()
        .strip()
    )

    print(
        f"Saved: "
        f"{output_path}"
    )

    print(
        f"Time: "
        f"{elapsed:.1f}s"
    )

    print(
        f"GPU stats: "
        f"{gpu_info}"
    )

print("DONE")
