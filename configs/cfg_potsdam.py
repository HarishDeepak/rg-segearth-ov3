_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_potsdam.txt',
    prob_thd=0.1,
    confidence_threshold=0.1,
    bg_idx=5,

    slide_stride=1024,
    slide_crop=1024,
)

# dataset settings
dataset_type = 'PotsdamDataset'
# data_root = 'data/Potsdam'
data_root = (
    '/kaggle/working/'
    'PotsdamEval'
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,

        img_suffix='_RGB.tif',
        seg_map_suffix='_label_noBoundary.tif',

        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir_indexed/val'),
        pipeline=test_pipeline))