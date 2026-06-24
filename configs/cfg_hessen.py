_base_ = './base_config.py'

# model settings
# slide_crop=256 + stride=128 (50% overlap) gives the model a "zoomed-in" view
# of 20cm GSD imagery, compensating for the 4x resolution gap vs Potsdam 5cm.
# SAM3 resizes any crop to 1008x1008 internally, so a 256px crop of a 20cm tile
# is processed at the same spatial density as a 64px crop of a 5cm tile would be.
model = dict(
    classname_path='./configs/cls_hessen.txt',
    prob_thd=0.1,
    confidence_threshold=0.1,
    bg_idx=5,

    slide_stride=128,
    slide_crop=256,
)
