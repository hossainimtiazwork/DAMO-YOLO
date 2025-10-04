# Spatio-Temporal Model Modifications for DAMO-YOLO

## Overview

This document describes the modifications made to DAMO-YOLO to support spatio-temporal processing for video-based object detection. The changes enable the model to process sequences of frames and leverage temporal information for improved detection accuracy.

## Architecture Changes

### 1. Core Operations (`damo/base_models/core/ops.py`)

#### Added Components:

**a) Conv3DBNAct Module**
- 3D convolution block with Batch Normalization and activation
- Supports spatio-temporal feature extraction
- Similar structure to Conv2d but operates on (B, C, T, H, W) tensors

```python
Conv3DBNAct(in_channels, out_channels, ksize, stride=1, act='silu')
```

**b) TemporalFusion Module**
- Aggregates features across temporal dimension
- Three fusion strategies:
  - `conv3d`: Uses 3D convolution with temporal kernel
  - `avg`: Simple average pooling over time
  - `attention`: Temporal attention mechanism

```python
TemporalFusion(in_channels, num_frames=4, fusion_type='conv3d')
```

**c) Enhanced get_norm Function**
- Extended to support both 2D and 3D batch normalization
- Automatically selects appropriate normalization based on `dims` parameter

### 2. Temporal Backbone (`damo/base_models/backbones/tinynas_csp_temporal.py`)

#### TemporalCSPWrapper
- Extension of CSPWrapper with temporal fusion capability
- Can be selectively enabled at different stages
- Maintains spatial feature extraction while adding temporal aggregation

**Key Parameters:**
- `temporal_fusion`: Enable/disable temporal processing
- `num_frames`: Number of frames in input sequence
- `fusion_type`: Type of temporal fusion ('conv3d', 'avg', or 'attention')

#### TinyNAS_Temporal
- Complete temporal backbone based on TinyNAS CSP architecture
- Supports both image and video inputs
- Configurable temporal fusion at different stages

**Input Formats:**
- Images: `(B, C, H, W)`
- Videos: `(B, C, T, H, W)` or `(B*T, C, H, W)`

**Configuration Parameters:**
```python
{
    'name': 'TinyNAS_csp_temporal',
    'num_frames': 4,              # Number of frames per clip
    'temporal_stages': [3, 4],    # Stages with temporal fusion
    'fusion_type': 'conv3d',      # Fusion strategy
    # ... other TinyNAS parameters
}
```

### 3. Configuration Example

See `configs/damoyolo_tinynasL45_L_temporal.py` for a complete example configuration.

## Usage

### Training with Temporal Model

1. **Prepare video dataset** with frame sequences
2. **Configure the model**:

```python
TinyNAS_Temporal = {
    'name': 'TinyNAS_csp_temporal',
    'net_structure_str': structure,
    'out_indices': (2, 3, 4),
    'with_spp': True,
    'use_focus': True,
    'act': 'silu',
    'reparam': True,
    'num_frames': 4,
    'temporal_stages': [3, 4],
    'fusion_type': 'conv3d',
}
```

3. **Input format**: Ensure your dataloader provides frames in shape `(B, C, T, H, W)` or `(B*T, C, H, W)`

### Inference

The temporal model can handle both single images and video sequences:

```python
# Single image
output = model(image)  # (B, C, H, W)

# Video sequence
output = model(video)  # (B, C, T, H, W) or (B*T, C, H, W)
```

## Temporal Fusion Strategies

### 1. Conv3D Fusion
- Uses 3D convolution with temporal kernel (3, 1, 1)
- Learns temporal patterns from data
- Best for capturing motion patterns
- **Recommended for**: Action detection, tracking

### 2. Average Fusion
- Simple average over temporal dimension
- Lightweight, no additional parameters
- Good baseline for static scenes
- **Recommended for**: Static camera scenarios

### 3. Attention Fusion
- Learns to weight different frames based on importance
- More expressive but computationally expensive
- Adapts to varying temporal dynamics
- **Recommended for**: Complex temporal patterns

## Design Principles

1. **Backward Compatibility**: The temporal model can process single images by treating them as single-frame videos
2. **Minimal Changes**: Core architecture remains unchanged; temporal processing is added as optional modules
3. **Flexibility**: Temporal fusion can be enabled/disabled at different stages
4. **Efficiency**: Selective temporal processing reduces computational overhead

## Performance Considerations

### Memory Usage
- Video processing requires more memory due to temporal dimension
- Reduce batch size when using temporal models
- Consider using gradient checkpointing for very long sequences

### Computational Cost
- 3D convolutions are more expensive than 2D
- Temporal fusion adds ~20-30% computational overhead per stage
- Can be optimized by using temporal fusion only at later stages

### Recommended Settings

**For Real-time Applications:**
```python
'num_frames': 2,
'temporal_stages': [4],
'fusion_type': 'avg'
```

**For Accuracy:**
```python
'num_frames': 4,
'temporal_stages': [3, 4],
'fusion_type': 'conv3d'
```

**For Resource-Constrained:**
```python
'num_frames': 2,
'temporal_stages': [4],
'fusion_type': 'avg'
```

## Extension Possibilities

### 1. Video-Specific Augmentations
- Temporal jittering
- Frame dropout
- Motion blur

### 2. Temporal Neck
- Extend GiraffeNeckV2 with temporal processing
- Multi-scale temporal feature pyramid

### 3. Temporal Head
- RNN/LSTM-based detection head
- Temporal NMS for video object detection

### 4. Online Inference
- Sliding window approach for streaming video
- Frame buffering and temporal cache

## Example Code

### Loading a Temporal Model

```python
from damo.base_models.backbones import build_backbone

backbone_cfg = {
    'name': 'TinyNAS_csp_temporal',
    'net_structure_str': structure,
    'out_indices': (2, 3, 4),
    'with_spp': True,
    'use_focus': True,
    'act': 'silu',
    'reparam': True,
    'num_frames': 4,
    'temporal_stages': [3, 4],
    'fusion_type': 'conv3d',
}

backbone = build_backbone(backbone_cfg)
```

### Processing Video Input

```python
import torch

# Video input: (batch, channels, time, height, width)
video_input = torch.randn(2, 3, 4, 640, 640)

# Forward pass
features = backbone(video_input)

# Output: List of feature maps at different scales
for i, feat in enumerate(features):
    print(f"Feature {i}: {feat.shape}")
```

## References

- Original DAMO-YOLO paper: [arxiv.org/abs/2211.15444](https://arxiv.org/abs/2211.15444)
- Temporal CNNs: Temporal Segment Networks (TSN)
- 3D CNNs: I3D, C3D architectures

## Troubleshooting

### Issue: Out of Memory
- Reduce `num_frames`
- Reduce `batch_size`
- Use fewer temporal stages

### Issue: Slow Training
- Use `fusion_type='avg'` for baseline
- Reduce number of temporal stages
- Use mixed precision training

### Issue: Poor Convergence
- Pre-train with spatial model first
- Use lower learning rate for temporal modules
- Increase warmup epochs

## Contributing

When extending the temporal functionality:
1. Maintain backward compatibility with spatial-only models
2. Add comprehensive documentation
3. Include unit tests
4. Consider computational efficiency
