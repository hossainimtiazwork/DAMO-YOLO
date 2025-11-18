# Boundary-Aware Dice Loss

A PyTorch implementation of Boundary-Aware Dice Loss for multiclass semantic segmentation with support for ignoring specific indices (e.g., background class).

## Overview

The `BoundaryAwareDiceLoss` combines the standard Dice loss with boundary awareness, giving more weight to boundary regions while allowing specific indices (typically background) to be ignored during loss computation. This is particularly useful for semantic segmentation tasks where:

1. **Boundary precision is important** - The loss emphasizes edge regions between different classes
2. **Background should be ignored** - Pixels with a specific index (e.g., 0 or 255) are excluded from loss calculation
3. **Multiclass segmentation** - Supports any number of classes with proper handling of class imbalance

## Key Features

- ✅ **Multiclass Support**: Works with any number of classes
- ✅ **Boundary Awareness**: Automatically detects and emphasizes boundary regions
- ✅ **Ignore Index**: Skip specific pixels (e.g., background or unlabeled regions)
- ✅ **Flexible Reduction**: Supports 'none', 'mean', and 'sum' reduction modes
- ✅ **Gradient Flow**: Fully differentiable for backpropagation
- ✅ **GPU Support**: Works seamlessly on both CPU and CUDA

## Installation

The loss function is located at `damo/base_models/losses/dice_loss.py`. No additional dependencies beyond PyTorch are required.

## Usage

### Basic Example

```python
import torch
from damo.base_models.losses.dice_loss import BoundaryAwareDiceLoss

# Initialize loss function
num_classes = 21  # e.g., Pascal VOC
loss_fn = BoundaryAwareDiceLoss(num_classes=num_classes)

# Model predictions (logits): (N, C, H, W)
predictions = torch.randn(4, 21, 128, 128)

# Ground truth (class indices): (N, H, W)
ground_truth = torch.randint(0, 21, (4, 128, 128))

# Compute loss
loss = loss_fn(predictions, ground_truth)
```

### With Ignore Index (Background)

```python
# Ignore background class (index 0)
loss_fn = BoundaryAwareDiceLoss(
    num_classes=21,
    ignore_index=0  # Pixels with class 0 are excluded
)

# Ground truth with background
ground_truth = torch.randint(0, 21, (4, 128, 128))
ground_truth[:, :10, :] = 0  # Set border as background

loss = loss_fn(predictions, ground_truth)
```

### Custom Parameters

```python
# Medical image segmentation with high boundary emphasis
loss_fn = BoundaryAwareDiceLoss(
    num_classes=4,
    ignore_index=0,
    smooth=0.1,              # Smaller smooth for sharper boundaries
    boundary_weight=3.0,     # Higher weight on boundaries
    reduction='mean',
    loss_weight=1.0
)

# Combine with other losses
total_loss = 0.5 * dice_loss + 0.5 * cross_entropy_loss
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | int | Required | Number of classes in the segmentation task |
| `ignore_index` | int | `None` | Index to ignore in loss calculation (typically background) |
| `smooth` | float | `1.0` | Smoothing factor to avoid division by zero |
| `boundary_weight` | float | `2.0` | Weight multiplier for boundary regions (higher = more focus on edges) |
| `reduction` | str | `'mean'` | Loss reduction mode: `'none'`, `'mean'`, or `'sum'` |
| `loss_weight` | float | `1.0` | Global weight multiplier for the loss |

## How It Works

### 1. Boundary Detection

The loss automatically detects boundaries by computing gradients in the target segmentation mask:

```python
# Horizontal and vertical differences detect edges
boundary_mask = detect_edges(target)
```

### 2. Weighted Dice Computation

For each class (except ignored ones):

```python
# Standard Dice coefficient with boundary weighting
intersection = (pred * target * weight_mask).sum()
union = (pred + target) * weight_mask.sum()
dice = (2 * intersection + smooth) / (union + smooth)
loss = 1 - dice
```

### 3. Ignore Index Handling

Pixels with `ignore_index` are completely excluded:
- Not included in intersection or union calculations
- Do not contribute to gradient updates
- Allows focus on foreground classes only

## Shape Requirements

- **Input (predictions)**: `(N, C, H, W)` - Batch of logits before softmax
- **Target**: `(N, H, W)` - Batch of class indices
- **Output**: Scalar (if reduction='mean' or 'sum') or `(N, C)` (if reduction='none')

Where:
- `N` = Batch size
- `C` = Number of classes
- `H` = Height
- `W` = Width

## Examples

See `example_dice_loss_usage.py` for comprehensive examples including:
1. Basic usage with default parameters
2. Using ignore_index for background
3. Adjusting boundary awareness
4. Integration in training loops
5. Custom configurations for different domains

Run the examples:
```bash
python example_dice_loss_usage.py
```

## Testing

Unit tests are provided in `test_dice_loss.py`:

```bash
python test_dice_loss.py -v
```

Tests cover:
- Basic functionality
- Ignore index behavior
- Boundary detection
- Gradient flow
- Different reduction modes
- Edge cases (perfect/worst predictions)

## Use Cases

### Medical Image Segmentation
```python
loss_fn = BoundaryAwareDiceLoss(
    num_classes=4,  # Background + 3 organs
    ignore_index=0,
    boundary_weight=3.0  # High boundary emphasis for precise organ boundaries
)
```

### Autonomous Driving
```python
loss_fn = BoundaryAwareDiceLoss(
    num_classes=19,  # Cityscapes
    ignore_index=255,  # Unlabeled pixels
    boundary_weight=1.5
)
```

### General Semantic Segmentation
```python
loss_fn = BoundaryAwareDiceLoss(
    num_classes=21,  # Pascal VOC
    ignore_index=0,
    boundary_weight=2.0
)
```

## Mathematical Formulation

For each class `c` (excluding ignore_index):

```
weight(x, y) = 1 + (boundary_weight - 1) * boundary_mask(x, y)

intersection_c = Σ(pred_c * target_c * weight * valid_mask)
union_c = Σ((pred_c + target_c) * weight * valid_mask)

dice_c = (2 * intersection_c + smooth) / (union_c + smooth)

loss_c = 1 - dice_c

total_loss = mean(loss_c) for all valid classes
```

## Performance Considerations

- **Memory**: Efficient computation without storing large intermediate tensors
- **Speed**: Boundary detection uses simple gradient operations
- **GPU**: Fully optimized for CUDA execution
- **Batch Processing**: Handles variable batch sizes efficiently

## License

Copyright (C) Alibaba Group Holding Limited. All rights reserved.

## Citation

If you use this loss function in your research, please cite:

```bibtex
@misc{boundary_aware_dice_loss,
  title={Boundary-Aware Dice Loss for Semantic Segmentation},
  author={DAMO-YOLO Team},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/hossainimtiazwork/DAMO-YOLO}}
}
```
