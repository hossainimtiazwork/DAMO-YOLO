# Spatio-Temporal Needle Detection Model

This document describes the modifications made to DAMO-YOLO to support spatio-temporal needle detection in noisy environments.

## Overview

The modified DAMO-YOLO model now supports:
1. **Temporal sequences** of frames as input (spatio-temporal modeling)
2. **Four-component output**:
   - Bounding box detection
   - Needle type classification (2 classes)
   - Needle radius prediction
   - Needle direction (orientation angle)

## Architecture

### 1. Temporal Backbone (`temporal_backbone.py`)

The temporal backbone processes sequences of frames to capture temporal information.

**Key Features:**
- Accepts 5D input: `(Batch, Time, Channels, Height, Width)`
- Multiple fusion strategies:
  - `conv3d`: 3D convolutions for spatio-temporal feature extraction
  - `attention`: Multi-head temporal attention mechanism
  - `lstm`: LSTM for temporal sequence modeling
  - `average`: Simple temporal averaging (baseline)
- Wraps existing TinyNAS backbones
- Outputs fused features compatible with standard FPN

**Usage:**
```python
TemporalBackbone = {
    'name': 'TemporalBackbone',
    'base_backbone': BaseTinyNAS,
    'fusion_type': 'conv3d',  # or 'attention', 'lstm', 'average'
    'num_frames': 8,
    'fusion_stages': [2, 4, 5],
}
```

### 2. Needle Detection Head (`needle_head.py`)

Specialized detection head that extends the standard YOLO head with additional output branches.

**Output Components:**

1. **Classification Branch**
   - Predicts needle type (2 classes)
   - Uses Quality Focal Loss (QFL)

2. **Bounding Box Branch**
   - Standard bbox regression with Distribution Focal Loss (DFL)
   - GIoU loss for bbox refinement

3. **Radius Branch**
   - Predicts needle radius (single value)
   - Uses CircleLoss (smooth L1-based)

4. **Direction Branch**
   - Predicts needle orientation as `(sin(θ), cos(θ))`
   - Uses DirectionLoss (cosine similarity-based)
   - Continuous representation avoids angle wrapping issues

**Architecture:**
```
Input Features → Classification Conv Layers → Classification Head (2 classes)
              ↘ BBox Regression Conv Layers → BBox Head (4 * (reg_max + 1))
              ↘ Radius Conv Layers → Radius Head (1 value)
              ↘ Direction Conv Layers → Direction Head (2 values: sin, cos)
```

### 3. Specialized Losses (`needle_losses.py`)

#### CircleLoss
- Loss function for needle radius prediction
- Smooth L1 loss with scale-aware weighting
- Robust to outliers

```python
loss_radius = CircleLoss(loss_weight=1.0)
```

#### DirectionLoss
- Loss function for needle direction prediction
- Based on cosine similarity between predicted and target direction vectors
- Direction represented as `(sin(θ), cos(θ))` for continuity
- Loss: `1 - cosine_similarity(pred, target)`

```python
loss_direction = DirectionLoss(loss_weight=1.0)
```

#### AngleLoss (Alternative)
- Direct angle difference with periodic wrapping
- Can be used as alternative to DirectionLoss

### 4. Temporal Detector (`temporal_detector.py`)

Modified detector that handles both single frames and temporal sequences.

**Features:**
- Automatically detects input dimensionality (4D vs 5D)
- Processes temporal sequences through temporal backbone
- Falls back to standard detector for single frames
- Compatible with existing training/evaluation pipeline

## Configuration

### Model Configuration (`configs/damoyolo_needle_detection.py`)

Example configuration for needle detection:

```python
# Temporal backbone with 8-frame sequences
TemporalBackbone = {
    'name': 'TemporalBackbone',
    'base_backbone': {
        'name': 'TinyNAS_res',
        'net_structure_str': structure,
        'out_indices': (2, 4, 5),
        'with_spp': True,
        'use_focus': True,
        'act': 'relu',
        'reparam': True,
    },
    'fusion_type': 'conv3d',
    'num_frames': 8,
    'fusion_stages': [2, 4, 5],
}

# Needle detection head
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
    'max_radius': 100.0,  # Max expected needle radius in pixels
}
```

## Dataset Format

The model expects data with additional annotations for needle-specific attributes:

```python
# Standard COCO format plus:
{
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,  # needle type (1 or 2)
            "bbox": [x, y, w, h],
            "radius": 15.5,  # Needle radius in pixels
            "direction": 1.57,  # Direction in radians (-π to π)
            "area": ...,
            "iscrowd": 0
        }
    ]
}
```

### Data Preparation

1. Annotate your needle dataset with:
   - Bounding boxes
   - Needle type (class 1 or 2)
   - Needle radius (in pixels)
   - Needle direction (angle in radians)

2. Organize temporal sequences:
   ```
   dataset/
   ├── train/
   │   ├── sequence_001/
   │   │   ├── frame_0000.jpg
   │   │   ├── frame_0001.jpg
   │   │   └── ...
   │   └── sequence_002/
   │       └── ...
   └── annotations/
       ├── train.json
       └── val.json
   ```

## Training

### Basic Training

```bash
# Single GPU
python tools/train.py -f configs/damoyolo_needle_detection.py

# Multi-GPU (8 GPUs)
python -m torch.distributed.launch --nproc_per_node=8 \
    tools/train.py -f configs/damoyolo_needle_detection.py
```

### Training with Distillation

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
    tools/train.py \
    -f configs/damoyolo_needle_detection.py \
    --tea_config configs/damoyolo_needle_detection_teacher.py \
    --tea_ckpt path/to/teacher.pth
```

## Inference

### Single Frame Inference

```python
import torch
from damo.detectors.temporal_detector import build_local_model
from damo.config.base import parse_config

# Load model
config = parse_config('configs/damoyolo_needle_detection.py')
model = build_local_model(config, device='cuda')
model.load_state_dict(torch.load('checkpoint.pth')['model'])
model.eval()

# Single frame
image = torch.randn(1, 3, 640, 640).cuda()
with torch.no_grad():
    output = model(image)

# Output contains: [detections, features]
# detections: [bbox, scores, class_ids, radius, direction]
```

### Temporal Sequence Inference

```python
# Temporal sequence (8 frames)
sequence = torch.randn(1, 8, 3, 640, 640).cuda()
with torch.no_grad():
    output = model(sequence)
```

## Evaluation

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
    tools/eval.py \
    -f configs/damoyolo_needle_detection.py \
    --ckpt path/to/checkpoint.pth
```

## Testing

Run the test suite to verify model functionality:

```bash
python test_needle_model.py
```

The test suite validates:
- Model instantiation
- Temporal backbone functionality
- NeedleHead forward pass
- Loss computation
- Single frame and temporal sequence inference

## Model Output Format

The model outputs a dictionary containing:

```python
{
    'total_loss': total_loss,      # Total combined loss
    'loss_cls': loss_qfl,          # Classification loss (QFL)
    'loss_bbox': loss_bbox,        # Bounding box loss (GIoU)
    'loss_dfl': loss_dfl,          # Distribution Focal Loss
    'loss_radius': loss_radius,    # Needle radius loss
    'loss_direction': loss_direction,  # Needle direction loss
}
```

During inference, the output is:
```python
# List of detections per image
[
    tensor([[x1, y1, x2, y2, confidence, class_id, radius, angle], ...]),
    ...
]
```

## Key Considerations

### 1. Noisy Environment Handling
The temporal modeling helps filter noise by:
- Aggregating information across multiple frames
- Using 3D convolutions to learn spatio-temporal features
- Attention mechanisms to focus on consistent patterns

### 2. Radius Prediction
- Radius is normalized to `[0, max_radius]` range
- Uses sigmoid activation scaled by `max_radius`
- CircleLoss provides robust regression

### 3. Direction Prediction
- Direction represented as `(sin(θ), cos(θ))`
- Avoids discontinuity at angle wrapping points
- DirectionLoss uses cosine similarity for smooth gradients
- Convert to angle: `θ = atan2(sin, cos)`

### 4. Temporal Sequences
- Default: 8 frames per sequence
- Can be adjusted via `num_frames` parameter
- Memory usage scales linearly with sequence length
- Consider reducing batch size for longer sequences

## Performance Tips

1. **Fusion Strategy**
   - `conv3d`: Best for capturing local temporal patterns
   - `attention`: Best for long-range temporal dependencies
   - `lstm`: Best for sequential temporal modeling
   - `average`: Fastest, baseline performance

2. **Batch Size**
   - Reduce batch size when using temporal sequences
   - Recommended: `batch_size = 64` for 8-frame sequences

3. **Loss Weights**
   - Adjust loss weights based on task priority:
     ```python
     loss_cls: 1.0      # Classification
     loss_bbox: 2.0     # Bounding box (higher for tight localization)
     loss_dfl: 0.25     # Distribution focal
     loss_radius: 1.0   # Radius
     loss_direction: 1.0  # Direction
     ```

## Limitations and Future Work

1. **Current Limitations**
   - Fixed temporal sequence length
   - No temporal data augmentation implemented
   - Postprocessing needs extension for radius and direction

2. **Future Improvements**
   - Variable-length temporal sequences
   - Temporal data augmentation (frame dropout, temporal shift)
   - End-to-end temporal NMS
   - Multi-scale temporal fusion
   - Lightweight temporal models for real-time inference

## File Structure

```
damo/
├── base_models/
│   ├── backbones/
│   │   ├── temporal_backbone.py    # NEW: Temporal fusion module
│   │   └── __init__.py             # MODIFIED: Register TemporalBackbone
│   ├── heads/
│   │   ├── needle_head.py          # NEW: Needle detection head
│   │   └── __init__.py             # MODIFIED: Register NeedleHead
│   └── losses/
│       └── needle_losses.py        # NEW: CircleLoss, DirectionLoss
├── detectors/
│   └── temporal_detector.py        # NEW: Temporal detector wrapper
configs/
└── damoyolo_needle_detection.py    # NEW: Needle detection config
test_needle_model.py                 # NEW: Test suite
```

## Citation

If you use this work, please cite:

```bibtex
@article{damoyolo,
  title={DAMO-YOLO: A Report on Real-Time Object Detection Design},
  author={Xianzhe Xu, Yiqi Jiang, Weihua Chen, Yilun Huang, Yuan Zhang and Xiuyu Sun},
  journal={arXiv preprint arXiv:2211.15444v2},
  year={2022},
}
```

## License

Copyright (C) Alibaba Group Holding Limited. All rights reserved.
Licensed under Apache License 2.0.
