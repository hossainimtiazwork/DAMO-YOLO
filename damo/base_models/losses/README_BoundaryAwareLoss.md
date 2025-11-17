# BoundaryAwareLoss Documentation

## Overview

`BoundaryAwareLoss` is a multiclass loss function that emphasizes the boundaries between different classes. It is particularly useful for tasks where precise class boundaries are important, such as semantic segmentation or object detection with segmentation masks.

## Key Features

- **Multiclass Support**: Works with any number of classes
- **Boundary Emphasis**: Automatically detects and emphasizes class boundaries
- **Background Ignore**: Can ignore specific classes (e.g., background class)
- **Configurable**: Flexible parameters for different use cases
- **Standard Interface**: Follows PyTorch and DAMO-YOLO conventions

## Installation

The loss is already integrated into the DAMO-YOLO losses module:

```python
from damo.base_models.losses import BoundaryAwareLoss
```

## Usage Examples

### Basic Usage

```python
import torch
from damo.base_models.losses import BoundaryAwareLoss

# Create loss function for 21 classes (e.g., PASCAL VOC)
loss_fn = BoundaryAwareLoss(num_classes=21)

# Predictions: (batch_size, num_classes, height, width)
pred = torch.randn(4, 21, 64, 64)

# Targets: (batch_size, height, width)
target = torch.randint(0, 21, (4, 64, 64))

# Compute loss
loss = loss_fn(pred, target)
print(f"Loss: {loss.item():.4f}")
```

### With Background Ignore

```python
# Ignore class 0 (background)
loss_fn = BoundaryAwareLoss(
    num_classes=21,
    ignore_class=0,  # Background class
    boundary_weight=2.5
)

pred = torch.randn(4, 21, 64, 64)
target = torch.randint(0, 21, (4, 64, 64))

loss = loss_fn(pred, target)
# Background pixels will not contribute to the loss
```

### Adjusting Boundary Weight

```python
# Higher boundary_weight emphasizes boundaries more
loss_fn = BoundaryAwareLoss(
    num_classes=21,
    boundary_weight=3.0  # Default is 2.0
)

# boundary_weight=1.0 means no additional emphasis on boundaries
# boundary_weight=2.0 means boundaries get 2x weight
# boundary_weight=3.0 means boundaries get 3x weight
```

### Different Reduction Methods

```python
# Mean reduction (default)
loss_fn_mean = BoundaryAwareLoss(num_classes=21, reduction='mean')

# No reduction (returns per-pixel loss)
loss_fn_none = BoundaryAwareLoss(num_classes=21, reduction='none')

# Sum reduction
loss_fn_sum = BoundaryAwareLoss(num_classes=21, reduction='sum')

pred = torch.randn(4, 21, 64, 64)
target = torch.randint(0, 21, (4, 64, 64))

loss_mean = loss_fn_mean(pred, target)  # Scalar
loss_none = loss_fn_none(pred, target)  # Shape: (4, 64, 64)
loss_sum = loss_fn_sum(pred, target)    # Scalar
```

### In Training Loop

```python
import torch
import torch.nn as nn
from damo.base_models.losses import BoundaryAwareLoss

# Setup
model = MySegmentationModel()
loss_fn = BoundaryAwareLoss(
    num_classes=21,
    ignore_class=0,
    boundary_weight=2.5,
    reduction='mean'
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (images, masks) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(images)  # Shape: (B, C, H, W)
        
        # Compute loss
        loss = loss_fn(predictions, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
```

## Parameters

### BoundaryAwareLoss

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | int | Required | Number of classes in the classification task |
| `boundary_weight` | float | 2.0 | Weight multiplier for boundary regions. Higher values emphasize boundaries more |
| `ignore_class` | int or None | None | Class index to ignore (e.g., background). If specified, pixels of this class don't contribute to loss |
| `use_sigmoid` | bool | True | Whether to use sigmoid (binary/multi-label) or softmax (multi-class) |
| `kernel_size` | int | 3 | Kernel size for boundary detection (currently fixed behavior) |
| `reduction` | str | 'mean' | Reduction method: 'none', 'mean', or 'sum' |
| `loss_weight` | float | 1.0 | Overall loss weight multiplier |

### forward() Method

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pred` | torch.Tensor | Required | Predicted logits with shape (N, C, H, W) |
| `target` | torch.Tensor | Required | Ground truth labels with shape (N, H, W) or (N, 1, H, W) |
| `weight` | torch.Tensor or None | None | Element-wise weights |
| `avg_factor` | int or None | None | Average factor for loss reduction |
| `reduction_override` | str or None | None | Override the default reduction method |

## How It Works

1. **Base Loss Calculation**: Computes standard cross-entropy loss for each pixel
2. **Boundary Detection**: Identifies boundaries by comparing neighboring pixels
3. **Weight Computation**: Creates a weight map that emphasizes boundaries
4. **Weighted Loss**: Applies the weight map to the base loss
5. **Class Ignore**: Zeros out contributions from the ignored class (if specified)

### Boundary Detection

The loss detects boundaries by comparing each pixel with its neighbors:
- Horizontal neighbors (left/right)
- Vertical neighbors (up/down)
- Diagonal neighbors (4 directions)

A pixel is considered part of a boundary if any of its neighbors belongs to a different class.

### Weight Calculation

For each pixel, the weight is calculated as:
```
weight = 1.0 + (boundary_weight - 1.0) * boundary_mask
```

Where:
- `boundary_mask` is in [0, 1] (0 for non-boundary, 1 for strong boundary)
- Final weight is in [1.0, boundary_weight]

## Use Cases

### 1. Semantic Segmentation
Perfect for segmentation tasks where boundary precision is critical:
```python
loss_fn = BoundaryAwareLoss(
    num_classes=21,  # PASCAL VOC classes
    ignore_class=255,  # Ignore void class
    boundary_weight=2.5
)
```

### 2. Instance Segmentation
Can be used for instance masks:
```python
loss_fn = BoundaryAwareLoss(
    num_classes=2,  # Binary: object vs background
    ignore_class=0,
    boundary_weight=3.0  # High emphasis on boundaries
)
```

### 3. Multi-Label Classification
For overlapping classes:
```python
loss_fn = BoundaryAwareLoss(
    num_classes=80,  # COCO classes
    use_sigmoid=True,  # Multi-label mode
    boundary_weight=2.0
)
```

## Tips and Best Practices

1. **Choose boundary_weight wisely**:
   - Start with 2.0 and adjust based on validation performance
   - Higher values (2.5-3.0) for tasks requiring very precise boundaries
   - Lower values (1.5-2.0) for tasks where boundaries are less critical

2. **Use ignore_class for background**:
   - Most datasets have a background or void class that should be ignored
   - This prevents the model from learning to predict ignored classes at boundaries

3. **Combine with other losses**:
   ```python
   boundary_loss = BoundaryAwareLoss(num_classes=21, ignore_class=0)
   dice_loss = DiceLoss()
   
   total_loss = boundary_loss(pred, target) + 0.5 * dice_loss(pred, target)
   ```

4. **Monitor boundary vs non-boundary loss**:
   - Use `reduction='none'` to analyze loss distribution
   - Check if boundary pixels have appropriately higher loss

5. **Adjust for class imbalance**:
   - Consider using class weights in combination with boundary weighting
   - Can pass weights through the `weight` parameter in forward()

## Performance Considerations

- **Memory**: Slightly higher memory usage due to boundary mask computation
- **Speed**: Small overhead (~5-10%) compared to standard cross-entropy loss
- **Precision**: Works best with accurate ground truth boundaries

## Comparison with Standard Cross-Entropy

| Feature | Cross-Entropy | BoundaryAwareLoss |
|---------|--------------|-------------------|
| Boundary emphasis | No | Yes (configurable) |
| Class ignore | Manual masking required | Built-in |
| Computation | Fast | Slightly slower |
| Boundary precision | Standard | Improved |
| Use case | General classification | Boundary-critical tasks |

## Troubleshooting

### Loss is NaN
- Check that predictions are not too extreme (use gradient clipping)
- Ensure target values are in valid range [0, num_classes)
- Verify that at least some pixels are not ignored

### Loss is too high
- Reduce `boundary_weight` parameter
- Check if model predictions are reasonable
- Ensure proper data normalization

### No improvement in boundary quality
- Increase `boundary_weight` (try 3.0 or higher)
- Verify that ground truth boundaries are accurate
- Consider combining with other boundary-focused losses

## References

The boundary-aware loss concept is inspired by various works in semantic segmentation:
- Boundary-aware loss functions for semantic segmentation
- Emphasis on hard examples near class boundaries
- Multi-scale boundary refinement

## License

Copyright (C) Alibaba Group Holding Limited. All rights reserved.
