# DAMO-YOLO Examples

This directory contains example scripts demonstrating various features of DAMO-YOLO.

## Temporal Model Example

### Overview
`temporal_model_example.py` demonstrates how to use the spatio-temporal features for video-based object detection.

### Requirements
```bash
pip install torch torchvision
```

### Usage

Run all examples:
```bash
cd /path/to/DAMO-YOLO
python examples/temporal_model_example.py
```

### What's Included

The script contains 4 examples:

#### Example 1: Basic Usage
- Shows how to configure and use the temporal backbone
- Processes video input (B, C, T, H, W)
- Displays output feature shapes

#### Example 2: Different Fusion Strategies
- Compares three temporal fusion types:
  - `conv3d`: Learnable 3D convolution
  - `avg`: Simple averaging
  - `attention`: Temporal attention mechanism
- Shows parameter counts for each approach

#### Example 3: Image Compatibility
- Demonstrates backward compatibility
- Shows temporal model can process single images
- No temporal fusion needed for static images

#### Example 4: Selective Temporal Processing
- Shows how to apply temporal fusion at different stages
- Compares computational costs
- Demonstrates flexibility in architecture design

### Expected Output

Each example will print:
- Input tensor shapes
- Output feature maps at different stages
- Model statistics (parameters, etc.)

### Configuration Options

When using the temporal model, you can configure:

```python
backbone_cfg = {
    'name': 'TinyNAS_csp_temporal',
    'num_frames': 4,           # Number of frames per clip
    'temporal_stages': [3, 4],  # Which stages use temporal fusion
    'fusion_type': 'conv3d',    # conv3d, avg, or attention
    # ... other standard parameters
}
```

### Performance Tips

1. **For Real-time**: Use `fusion_type='avg'` and `temporal_stages=[4]`
2. **For Accuracy**: Use `fusion_type='conv3d'` and `temporal_stages=[3, 4]`
3. **For Limited Resources**: Reduce `num_frames` to 2

### Troubleshooting

**Issue**: Module not found error
- **Solution**: Run from DAMO-YOLO root directory

**Issue**: Out of memory
- **Solution**: Reduce batch size or image resolution in examples

**Issue**: Structure file not found
- **Solution**: Ensure `damo/base_models/backbones/nas_backbones/tinynas_L45_kxkx.txt` exists

## Additional Resources

- Full documentation: `assets/SpatioTemporalModel.md`
- Configuration example: `configs/damoyolo_tinynasL45_L_temporal.py`
- Main README: `README.md`
