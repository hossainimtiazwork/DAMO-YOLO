#!/usr/bin/env python3
# Temporal configuration for DAMO-YOLO with spatio-temporal features
# Based on damoyolo_tinynasL45_L.py with temporal extensions

import os

from damo.config import Config as MyConfig


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.miscs.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split('.')[0]
        self.miscs.eval_interval_epochs = 10
        self.miscs.ckpt_interval_epochs = 10
        
        # optimizer
        self.train.batch_size = 128  # Reduced batch size for temporal sequences
        self.train.base_lr_per_img = 0.01 / 64
        self.train.min_lr_ratio = 0.05
        self.train.weight_decay = 5e-4
        self.train.momentum = 0.9
        self.train.no_aug_epochs = 16
        self.train.warmup_epochs = 5

        # augment - reduced augmentation for temporal consistency
        self.train.augment.transform.image_max_range = (640, 640)
        self.train.augment.mosaic_mixup.mixup_prob = 0.0  # Disable mixup for temporal
        self.train.augment.mosaic_mixup.mosaic_prob = 0.0  # Disable mosaic for temporal
        self.train.augment.mosaic_mixup.degrees = 5.0  # Reduced rotation
        self.train.augment.mosaic_mixup.translate = 0.1  # Reduced translation
        self.train.augment.mosaic_mixup.shear = 1.0  # Reduced shear
        self.train.augment.mosaic_mixup.mosaic_scale = (0.1, 2.0)

        # Dataset configuration for temporal sequences
        self.dataset.train_ann = ('coco_2017_train', )
        self.dataset.val_ann = ('coco_2017_val', )
        
        # Temporal-specific parameters
        self.dataset.num_frames = 3  # Number of frames in temporal sequence
        self.dataset.temporal_stride = 1  # Stride between frames

        # backbone - Temporal TinyNAS with multihead attention
        structure = self.read_structure(
            './damo/base_models/backbones/nas_backbones/tinynas_L45_kxkx.txt')
        TinyNAS = {
            'name': 'TinyNAS_csp_temporal',
            'net_structure_str': structure,
            'out_indices': (2, 3, 4),
            'with_spp': True,
            'use_focus': True,
            'act': 'silu',
            'reparam': True,
            # Temporal-specific parameters
            'num_frames': 3,  # Match dataset num_frames
            'temporal_fusion_stages': [2, 3, 4],  # Apply attention at these stages
            'num_heads': 8,  # Number of attention heads
            'dropout': 0.1,  # Dropout for attention
        }

        self.model.backbone = TinyNAS

        GiraffeNeckV2 = {
            'name': 'GiraffeNeckV2',
            'depth': 2.0,
            'hidden_ratio': 1.0,
            'in_channels': [128, 256, 512],
            'out_channels': [128, 256, 512],
            'act': 'silu',
            'spp': False,
            'block_name': 'BasicBlock_3x3_Reverse',
        }

        self.model.neck = GiraffeNeckV2

        ZeroHead = {
            'name': 'ZeroHead',
            'num_classes': 80,
            'in_channels': [128, 256, 512],
            'stacked_convs': 0,
            'reg_max': 16,
            'act': 'silu',
            'nms_conf_thre': 0.05,
            'nms_iou_thre': 0.7,
            'legacy': False,
        }
        self.model.head = ZeroHead

        self.dataset.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    def get_data(self, name):
        """Override to use temporal dataset."""
        if 'coco' in name:
            from damo.config.paths_catalog import DatasetCatalog
            from os.path import join
            
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=join(data_dir, attrs['img_dir']),
                ann_file=join(data_dir, attrs['ann_file']),
            )
            return dict(
                factory='COCOTemporalDataset',  # Use temporal dataset
                args=args,
            )
        else:
            raise RuntimeError('Only support coco format dataset now!')
