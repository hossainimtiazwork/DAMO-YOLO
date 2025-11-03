#!/usr/bin/env python3
# Configuration for Spatio-Temporal Needle Detection Model

import os
from damo.config import Config as MyConfig


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.miscs.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split('.')[0]
        self.miscs.eval_interval_epochs = 10
        self.miscs.ckpt_interval_epochs = 10
        
        # Optimizer configuration
        self.train.batch_size = 64  # Reduced due to temporal sequences
        self.train.base_lr_per_img = 0.01 / 64
        self.train.min_lr_ratio = 0.05
        self.train.weight_decay = 5e-4
        self.train.momentum = 0.9
        self.train.no_aug_epochs = 16
        self.train.warmup_epochs = 5

        # Augmentation configuration
        self.train.augment.transform.image_max_range = (640, 640)
        self.train.augment.mosaic_mixup.mixup_prob = 0.15
        self.train.augment.mosaic_mixup.degrees = 10.0
        self.train.augment.mosaic_mixup.translate = 0.2
        self.train.augment.mosaic_mixup.shear = 2.0
        self.train.augment.mosaic_mixup.mosaic_scale = (0.1, 2.0)

        # Dataset configuration for needle detection
        # Note: You'll need to prepare your custom needle dataset
        self.dataset.train_ann = ('needle_train', )
        self.dataset.val_ann = ('needle_val', )
        
        # Needle types: 2 classes
        self.dataset.class_names = ['needle_type_1', 'needle_type_2']

        # Temporal backbone configuration
        # Uses TinyNAS as base with temporal fusion
        structure = self.read_structure(
            './damo/base_models/backbones/nas_backbones/tinynas_L25_k1kx.txt')
        
        # Base backbone (TinyNAS)
        BaseTinyNAS = {
            'name': 'TinyNAS_res',
            'net_structure_str': structure,
            'out_indices': (2, 4, 5),
            'with_spp': True,
            'use_focus': True,
            'act': 'relu',
            'reparam': True,
        }
        
        # Temporal backbone wrapping the base
        TemporalBackbone = {
            'name': 'TemporalBackbone',
            'base_backbone': BaseTinyNAS,
            'fusion_type': 'conv3d',  # Options: 'conv3d', 'attention', 'lstm', 'average'
            'num_frames': 8,  # Number of frames in temporal sequence
            'fusion_stages': [2, 4, 5],  # Which stages to apply temporal fusion
        }
        
        self.model.backbone = TemporalBackbone

        # Neck configuration (GiraffeNeckV2)
        GiraffeNeckV2 = {
            'name': 'GiraffeNeckV2',
            'depth': 1.0,
            'hidden_ratio': 0.75,
            'in_channels': [128, 256, 512],
            'out_channels': [128, 256, 512],
            'act': 'relu',
            'spp': False,
            'block_name': 'BasicBlock_3x3_Reverse',
        }

        self.model.neck = GiraffeNeckV2

        # Needle detection head configuration
        NeedleHead = {
            'name': 'NeedleHead',
            'num_classes': 2,  # 2 needle types
            'in_channels': [128, 256, 512],
            'stacked_convs': 0,
            'reg_max': 16,
            'act': 'silu',
            'nms_conf_thre': 0.05,
            'nms_iou_thre': 0.7,
            'legacy': False,
            'max_radius': 100.0,  # Maximum expected needle radius in pixels
        }
        
        self.model.head = NeedleHead
        
        # Additional training parameters for temporal model
        self.train.temporal_augment = {
            'enabled': True,
            'temporal_shift': True,  # Enable temporal shift augmentation
            'frame_dropout': 0.1,  # Probability of dropping frames
        }
