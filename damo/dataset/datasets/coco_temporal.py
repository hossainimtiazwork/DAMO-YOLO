# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
# Temporal extension for COCO dataset to support multiple frames

import cv2
import numpy as np
import torch
from torchvision.datasets.coco import CocoDetection

from damo.structures.bounding_box import BoxList

cv2.setNumThreads(0)


class COCOTemporalDataset(CocoDetection):
    """
    COCO Dataset with temporal support.
    Loads sequences of frames for spatio-temporal feature learning.
    """
    def __init__(self, ann_file, root, transforms=None, class_names=None, 
                 num_frames=1, temporal_stride=1):
        """
        Args:
            ann_file: Path to COCO annotation file
            root: Root directory of images
            transforms: Data transformations
            class_names: List of class names
            num_frames: Number of consecutive frames to load (1 for single frame)
            temporal_stride: Stride between frames (1 means consecutive)
        """
        super(COCOTemporalDataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        
        self.num_frames = num_frames
        self.temporal_stride = temporal_stride

        assert (class_names is not None), 'plz provide class_names'

        self.contiguous_class2id = {
            class_name: i
            for i, class_name in enumerate(class_names)
        }
        self.contiguous_id2class = {
            i: class_name
            for i, class_name in enumerate(class_names)
        }

        categories = self.coco.dataset['categories']
        cat_names = [cat['name'] for cat in categories]
        cat_ids = [cat['id'] for cat in categories]
        self.ori_class2id = {
            class_name: i
            for class_name, i in zip(cat_names, cat_ids)
        }
        self.ori_id2class = {
            i: class_name
            for class_name, i in zip(cat_names, cat_ids)
        }

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def _get_temporal_indices(self, idx):
        """
        Get indices for temporal sequence centered around idx.
        For single frame mode (num_frames=1), just returns [idx].
        """
        if self.num_frames == 1:
            return [idx]
        
        indices = []
        # Center the sequence around idx
        start_offset = (self.num_frames - 1) // 2
        
        for i in range(self.num_frames):
            # Calculate frame index with temporal stride
            frame_offset = (i - start_offset) * self.temporal_stride
            frame_idx = idx + frame_offset
            
            # Clamp to valid range
            frame_idx = max(0, min(frame_idx, len(self.ids) - 1))
            indices.append(frame_idx)
        
        return indices

    def __getitem__(self, inp):
        if type(inp) is tuple:
            idx = inp[1]
        else:
            idx = inp
        
        # Get temporal indices
        temporal_indices = self._get_temporal_indices(idx)
        
        # Load all frames in the temporal sequence
        imgs = []
        targets = []
        
        for frame_idx in temporal_indices:
            img, anno = super(COCOTemporalDataset, self).__getitem__(frame_idx)
            # filter crowd annotations
            anno = [obj for obj in anno if obj['iscrowd'] == 0]

            boxes = [obj['bbox'] for obj in anno]
            boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
            target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

            classes = [obj['category_id'] for obj in anno]
            classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                       for c in classes]

            classes = torch.tensor(classes)
            target.add_field('labels', classes)
            target = target.clip_to_image(remove_empty=True)

            # PIL to numpy array
            img = np.asarray(img)  # rgb
            
            imgs.append(img)
            targets.append(target)
        
        # Apply transforms consistently across all frames
        if self._transforms is not None:
            # For temporal data, we need to apply same augmentation to all frames
            transformed_imgs = []
            transformed_targets = []
            
            for img, target in zip(imgs, targets):
                img_t, target_t = self._transforms(img, target)
                transformed_imgs.append(img_t)
                transformed_targets.append(target_t)
            
            imgs = transformed_imgs
            targets = transformed_targets
        
        # Stack frames into temporal dimension if multiple frames
        if self.num_frames == 1:
            return imgs[0], targets[0], idx
        else:
            # Stack images: (T, C, H, W) where T is number of frames
            imgs_stacked = np.stack(imgs, axis=0)
            # Use target from center frame
            center_target = targets[len(targets) // 2]
            return imgs_stacked, center_target, idx

    def pull_item(self, idx):
        """Pull item for evaluation - uses center frame for temporal sequences."""
        # Get temporal indices
        temporal_indices = self._get_temporal_indices(idx)
        # Use center frame
        center_idx = temporal_indices[len(temporal_indices) // 2]
        
        img, anno = super(COCOTemporalDataset, self).__getitem__(center_idx)

        # filter crowd annotations
        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')
        target = target.clip_to_image(remove_empty=True)

        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        obj_masks = []
        for obj in anno:
            obj_mask = []
            if 'segmentation' in obj:
                for mask in obj['segmentation']:
                    obj_mask += mask
                if len(obj_mask) > 0:
                    obj_masks.append(obj_mask)
        seg_masks = [
            np.array(obj_mask, dtype=np.float32).reshape(-1, 2)
            for obj_mask in obj_masks
        ]

        res = np.zeros((len(target.bbox), 5))
        for idx in range(len(target.bbox)):
            res[idx, 0:4] = target.bbox[idx]
            res[idx, 4] = classes[idx]

        img = np.asarray(img)  # rgb

        return img, res, seg_masks, idx

    def load_anno(self, idx):
        """Load annotations for center frame."""
        temporal_indices = self._get_temporal_indices(idx)
        center_idx = temporal_indices[len(temporal_indices) // 2]
        
        _, anno = super(COCOTemporalDataset, self).__getitem__(center_idx)
        anno = [obj for obj in anno if obj['iscrowd'] == 0]
        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        return classes

    def get_img_info(self, index):
        """Get image info for center frame."""
        temporal_indices = self._get_temporal_indices(index)
        center_idx = temporal_indices[len(temporal_indices) // 2]
        
        img_id = self.id_to_img_map[center_idx]
        img_data = self.coco.imgs[img_id]
        return img_data
