# Spatio-Temporal Features with Multihead Attention

This document describes the spatio-temporal feature extraction capability added to DAMO-YOLO, enabling the model to process multiple frames and fuse them using multihead attention.

## Overview

The temporal extension allows DAMO-YOLO to:
- Process sequences of consecutive frames instead of single images
- Fuse temporal information using multihead attention mechanism
- Maintain compatibility with the existing architecture
- Scale from single-frame to multi-frame operation via configuration

## Architecture

### 1. Temporal Backbone (`TinyNASTemporal`)

Located in `damo/base_models/backbones/tinynas_csp_temporal.py`, the temporal backbone extends the standard TinyNAS architecture with:

**Key Components:**
- `TemporalMultiheadAttention`: Applies self-attention across temporal dimension
- `TemporalFusionModule`: Wraps attention with configurable parameters
- `TinyNASTemporal`: Main backbone that integrates temporal processing

**Features:**
- Processes frames through standard convolution layers
- Applies attention-based fusion at configurable stages
- Supports both single-frame (T=1) and multi-frame (T>1) inputs
- Maintains spatial resolution while fusing temporal information

### 2. Temporal Dataset (`COCOTemporalDataset`)

Located in `damo/dataset/datasets/coco_temporal.py`, handles loading temporal sequences:

**Features:**
- Loads `num_frames` consecutive frames centered around target frame
- Configurable `temporal_stride` for frame sampling
- Consistent augmentation across all frames in sequence
- Graceful handling of sequence boundaries (clamping to valid indices)

**Parameters:**
- `num_frames`: Number of frames in sequence (default: 1 for single-frame)
- `temporal_stride`: Spacing between frames (1 = consecutive, 2 = every other frame)

### 3. Configuration

The temporal config (`configs/damoyolo_tinynasL45_L_temporal.py`) demonstrates usage:

```python
# Dataset temporal parameters
self.dataset.num_frames = 3
self.dataset.temporal_stride = 1

# Backbone temporal parameters
TinyNAS = {
    'name': 'TinyNAS_csp_temporal',
    'num_frames': 3,
    'temporal_fusion_stages': [2, 3, 4],  # Apply attention at these stages
    'num_heads': 8,
    'dropout': 0.1,
    # ... other backbone params
}
```

## Usage

### Training with Temporal Features

1. **Use the temporal config:**
```bash
python tools/train.py -f configs/damoyolo_tinynasL45_L_temporal.py
```

2. **Create custom temporal config:**
```python
from damo.config import Config as MyConfig

class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()
        
        # Configure temporal parameters
        self.dataset.num_frames = 5  # Use 5 frames
        self.dataset.temporal_stride = 2  # Sample every 2nd frame
        
        # Configure backbone
        TinyNAS = {
            'name': 'TinyNAS_csp_temporal',
            'num_frames': 5,
            'temporal_fusion_stages': [3, 4],  # Fusion at later stages only
            'num_heads': 4,  # Fewer attention heads
            # ...
        }
    
    def get_data(self, name):
        # Return COCOTemporalDataset instead of COCODataset
        return dict(factory='COCOTemporalDataset', args=...)
```

### Single-Frame Mode

The temporal backbone works in single-frame mode when `num_frames=1`:
- No temporal fusion is applied
- Behaves identically to standard TinyNAS
- Useful for fine-tuning or comparison

### Inference

The model accepts either:
- Single frame: `(B, C, H, W)` tensor
- Temporal sequence: `(B, T, C, H, W)` tensor

For temporal inference, prepare frames as:
```python
import torch
import numpy as np

# Load consecutive frames
frames = []
for i in range(num_frames):
    img = load_image(f"frame_{i}.jpg")
    img = preprocess(img)  # Apply same preprocessing
    frames.append(img)

# Stack to temporal tensor: (T, C, H, W)
temporal_input = torch.stack(frames, dim=0)
# Add batch dimension: (1, T, C, H, W)
temporal_input = temporal_input.unsqueeze(0)

# Run inference
with torch.no_grad():
    predictions = model(temporal_input)
```

## Configuration Parameters

### Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_frames` | int | 1 | Number of frames in temporal sequence |
| `temporal_stride` | int | 1 | Stride between sampled frames |

### Backbone Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_frames` | int | 1 | Number of frames (should match dataset) |
| `temporal_fusion_stages` | list[int] | [2,3,4] | Stages where temporal fusion is applied |
| `num_heads` | int | 8 | Number of attention heads |
| `dropout` | float | 0.1 | Dropout rate for attention |

## Design Considerations

### Data Augmentation

For temporal sequences:
- **Mosaic/Mixup disabled**: These augmentations are incompatible with temporal consistency
- **Consistent augmentation**: Same random augmentation applied to all frames in sequence
- **Reduced augmentation strength**: Gentler augmentation parameters recommended

### Batch Size

Temporal processing increases memory usage:
- Memory scales with `batch_size * num_frames`
- Recommended: Reduce batch size proportionally to `num_frames`
- Example: If standard batch_size=256, use batch_size=128 for num_frames=3

### Fusion Stages

The `temporal_fusion_stages` parameter controls where attention is applied:
- **Early stages (2)**: Fuse low-level features (edges, textures)
- **Middle stages (3)**: Fuse mid-level features (parts, patterns)
- **Late stages (4)**: Fuse high-level features (objects, semantics)
- **Recommendation**: Start with [2, 3, 4] for full temporal fusion

### Attention Heads

More heads allow the model to attend to different temporal patterns:
- `num_heads=8`: Good balance for most applications
- `num_heads=4`: Lighter, faster, less capacity
- `num_heads=16`: More capacity, higher memory usage

## Implementation Details

### Forward Pass Flow

1. **Input**: `(B, T, C, H, W)` temporal sequence
2. **Reshape**: Flatten batch and time → `(B*T, C, H, W)`
3. **Standard Processing**: Through backbone stages
4. **Temporal Fusion** (at specified stages):
   - Reshape to `(B, T, C', H', W')`
   - Apply multihead attention across T dimension
   - Pool temporal dimension: `(B, C', H', W')`
   - Continue with fused features
5. **Output**: List of stage features, each `(B, C, H, W)`

### Attention Mechanism

The `TemporalMultiheadAttention` module:
1. Reshapes spatial features to sequence format: `(T, B*H*W, C)`
2. Applies PyTorch `nn.MultiheadAttention`
3. Applies layer normalization
4. Averages across temporal dimension
5. Reshapes back to spatial format: `(B, C, H, W)`

This allows the model to learn which frames are most relevant for each spatial location.

## Limitations

1. **Sequential Data**: Assumes frames are sequential (e.g., from video)
2. **Fixed Stride**: Temporal stride is constant across all frames
3. **Memory**: Increases linearly with `num_frames`
4. **COCO Dataset**: Original COCO has images, not videos
   - Simulates temporal sequences by using nearby images
   - For real video data, implement custom dataset

## Future Enhancements

Possible improvements:
1. **Learnable temporal stride**: Different strides at different stages
2. **Temporal positional encoding**: Explicitly encode frame position
3. **Bidirectional attention**: Separate past/future attention
4. **Temporal pyramid**: Multi-scale temporal fusion
5. **Video dataset support**: Native support for video datasets
6. **3D convolutions**: Alternative to attention-based fusion

## Example Results

Expected behavior:
- **Motion objects**: Better detection of moving objects
- **Occlusion handling**: Can see around occlusions using other frames  
- **Temporal consistency**: Smoother predictions across frames
- **Context**: Uses temporal context for disambiguation

## Troubleshooting

**Issue**: Out of memory error
- **Solution**: Reduce batch_size or num_frames

**Issue**: Mosaic augmentation errors
- **Solution**: Ensure mosaic_prob=0 for temporal datasets

**Issue**: Inconsistent results
- **Solution**: Check that dataset and backbone num_frames match

**Issue**: No improvement over single-frame
- **Solution**: May need video-specific dataset; COCO images are not temporal

## Citation

If you use this temporal extension, please cite the original DAMO-YOLO paper and note the temporal modification:

```bibtex
@article{damoyolo,
  title={DAMO-YOLO: A Report on Real-Time Object Detection Design},
  author={...},
  journal={arXiv preprint arXiv:2211.15444},
  year={2022}
}
```

## Contact

For questions or issues related to the temporal features:
- Open an issue on the GitHub repository
- Reference this documentation in your issue
