#!/usr/bin/env python3
"""
Example script demonstrating how to use the temporal model for video processing.
This example shows:
1. Loading a temporal backbone
2. Processing video input
3. Different temporal fusion strategies
"""

import torch
import sys
sys.path.insert(0, '.')

from damo.base_models.backbones import build_backbone


def example_1_basic_usage():
    """Example 1: Basic usage with temporal backbone"""
    print("=" * 60)
    print("Example 1: Basic Temporal Model Usage")
    print("=" * 60)
    
    # Read structure file
    with open('./damo/base_models/backbones/nas_backbones/tinynas_L45_kxkx.txt', 'r') as f:
        structure = f.readlines()
    
    # Configure temporal backbone
    backbone_cfg = {
        'name': 'TinyNAS_csp_temporal',
        'net_structure_str': structure,
        'out_indices': (2, 3, 4),
        'with_spp': True,
        'use_focus': True,
        'act': 'silu',
        'reparam': False,
        'num_frames': 4,  # Process 4 frames at a time
        'temporal_stages': [3, 4],  # Apply temporal fusion at stages 3 and 4
        'fusion_type': 'conv3d',
    }
    
    # Build the backbone
    print("\nBuilding temporal backbone...")
    backbone = build_backbone(backbone_cfg)
    print("✓ Backbone built successfully")
    
    # Create sample video input (batch_size=2, channels=3, time=4, height=640, width=640)
    print("\nProcessing video input...")
    video_input = torch.randn(2, 3, 4, 640, 640)
    print(f"Input shape: {video_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        features = backbone(video_input)
    
    print("\nOutput features:")
    for i, feat in enumerate(features):
        print(f"  Stage {i}: {feat.shape}")
    
    print("\n✓ Example 1 completed successfully\n")


def example_2_different_fusion_types():
    """Example 2: Compare different temporal fusion strategies"""
    print("=" * 60)
    print("Example 2: Different Temporal Fusion Strategies")
    print("=" * 60)
    
    # Read structure file
    with open('./damo/base_models/backbones/nas_backbones/tinynas_L45_kxkx.txt', 'r') as f:
        structure = f.readlines()
    
    fusion_types = ['conv3d', 'avg', 'attention']
    
    for fusion_type in fusion_types:
        print(f"\n--- Testing {fusion_type.upper()} fusion ---")
        
        backbone_cfg = {
            'name': 'TinyNAS_csp_temporal',
            'net_structure_str': structure,
            'out_indices': (2, 3, 4),
            'with_spp': True,
            'use_focus': True,
            'act': 'silu',
            'reparam': False,
            'num_frames': 4,
            'temporal_stages': [4],  # Only last stage for comparison
            'fusion_type': fusion_type,
        }
        
        backbone = build_backbone(backbone_cfg)
        video_input = torch.randn(1, 3, 4, 320, 320)  # Smaller input for speed
        
        with torch.no_grad():
            features = backbone(video_input)
        
        print(f"  Input: {video_input.shape}")
        print(f"  Output stages: {[f.shape for f in features]}")
        
        # Count parameters
        num_params = sum(p.numel() for p in backbone.parameters())
        print(f"  Parameters: {num_params:,}")
    
    print("\n✓ Example 2 completed successfully\n")


def example_3_image_compatibility():
    """Example 3: Show backward compatibility with images"""
    print("=" * 60)
    print("Example 3: Backward Compatibility with Images")
    print("=" * 60)
    
    # Read structure file
    with open('./damo/base_models/backbones/nas_backbones/tinynas_L45_kxkx.txt', 'r') as f:
        structure = f.readlines()
    
    # Temporal model can still process single images
    backbone_cfg = {
        'name': 'TinyNAS_csp_temporal',
        'net_structure_str': structure,
        'out_indices': (2, 3, 4),
        'with_spp': True,
        'use_focus': True,
        'act': 'silu',
        'reparam': False,
        'num_frames': 1,  # Single frame
        'temporal_stages': [],  # No temporal fusion for images
        'fusion_type': 'avg',
    }
    
    print("\nBuilding temporal backbone (configured for images)...")
    backbone = build_backbone(backbone_cfg)
    
    # Process single image
    print("\nProcessing single image...")
    image_input = torch.randn(2, 3, 640, 640)  # Standard image input
    print(f"Input shape: {image_input.shape}")
    
    with torch.no_grad():
        features = backbone(image_input)
    
    print("\nOutput features:")
    for i, feat in enumerate(features):
        print(f"  Stage {i}: {feat.shape}")
    
    print("\n✓ Example 3 completed successfully")
    print("✓ Temporal model is backward compatible with single images\n")


def example_4_custom_temporal_stages():
    """Example 4: Selective temporal processing at different stages"""
    print("=" * 60)
    print("Example 4: Selective Temporal Processing")
    print("=" * 60)
    
    # Read structure file
    with open('./damo/base_models/backbones/nas_backbones/tinynas_L45_kxkx.txt', 'r') as f:
        structure = f.readlines()
    
    stage_configs = [
        [4],      # Only last stage
        [3, 4],   # Last two stages
        [2, 3, 4], # Last three stages
    ]
    
    for stages in stage_configs:
        print(f"\n--- Temporal stages: {stages} ---")
        
        backbone_cfg = {
            'name': 'TinyNAS_csp_temporal',
            'net_structure_str': structure,
            'out_indices': (2, 3, 4),
            'with_spp': True,
            'use_focus': True,
            'act': 'silu',
            'reparam': False,
            'num_frames': 4,
            'temporal_stages': stages,
            'fusion_type': 'conv3d',
        }
        
        backbone = build_backbone(backbone_cfg)
        video_input = torch.randn(1, 3, 4, 320, 320)
        
        with torch.no_grad():
            features = backbone(video_input)
        
        num_params = sum(p.numel() for p in backbone.parameters())
        print(f"  Parameters: {num_params:,}")
        print(f"  Output: {[f.shape for f in features]}")
    
    print("\n✓ Example 4 completed successfully\n")


def main():
    print("\n" + "=" * 60)
    print("DAMO-YOLO Temporal Model Examples")
    print("=" * 60 + "\n")
    
    try:
        example_1_basic_usage()
        example_2_different_fusion_types()
        example_3_image_compatibility()
        example_4_custom_temporal_stages()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n⚠ Warning: Structure file not found: {e}")
        print("Please ensure you're running from the DAMO-YOLO root directory")
        print("and that the required files are present.\n")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
