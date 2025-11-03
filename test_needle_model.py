#!/usr/bin/env python3
"""
Test script to validate the needle detection model can be instantiated and run.
"""

import torch
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from damo.config.base import parse_config
from damo.detectors.temporal_detector import build_local_model


def test_needle_model_instantiation():
    """Test that the needle detection model can be instantiated."""
    print("Testing needle detection model instantiation...")
    
    # Parse config
    config_file = os.path.join(os.path.dirname(__file__), 'configs', 'damoyolo_needle_detection.py')
    try:
        config = parse_config(config_file)
        print("✓ Config parsed successfully")
    except Exception as e:
        print(f"✗ Failed to parse config: {e}")
        return False
    
    # Build model
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = build_local_model(config, device)
        print(f"✓ Model built successfully on {device}")
    except Exception as e:
        print(f"✗ Failed to build model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_needle_model_forward():
    """Test forward pass with dummy data."""
    print("\nTesting needle detection model forward pass...")
    
    # Parse config
    config_file = os.path.join(os.path.dirname(__file__), 'configs', 'damoyolo_needle_detection.py')
    config = parse_config(config_file)
    
    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_local_model(config, device)
    model.eval()
    
    # Test with single frame (4D input)
    print("\nTest 1: Single frame input (B, C, H, W)")
    try:
        batch_size = 2
        single_frame = torch.randn(batch_size, 3, 640, 640).to(device)
        
        with torch.no_grad():
            output = model(single_frame)
        
        print(f"✓ Single frame forward pass successful")
        print(f"  Output type: {type(output)}")
        if isinstance(output, (list, tuple)):
            print(f"  Number of outputs: {len(output)}")
    except Exception as e:
        print(f"✗ Single frame forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with temporal sequence (5D input)
    print("\nTest 2: Temporal sequence input (B, T, C, H, W)")
    try:
        batch_size = 2
        num_frames = 8
        temporal_sequence = torch.randn(batch_size, num_frames, 3, 640, 640).to(device)
        
        with torch.no_grad():
            output = model(temporal_sequence)
        
        print(f"✓ Temporal sequence forward pass successful")
        print(f"  Output type: {type(output)}")
        if isinstance(output, (list, tuple)):
            print(f"  Number of outputs: {len(output)}")
    except Exception as e:
        print(f"✗ Temporal sequence forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_model_components():
    """Test individual model components."""
    print("\nTesting individual model components...")
    
    # Test temporal backbone
    print("\n1. Testing TemporalBackbone...")
    try:
        from damo.base_models.backbones.temporal_backbone import TemporalBackbone, TemporalFusion
        
        # Create a simple dummy backbone
        class DummyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, 3, padding=1)
            
            def forward(self, x):
                return [self.conv(x)]
            
            def init_weights(self):
                pass
        
        dummy_backbone = DummyBackbone()
        temporal_backbone = TemporalBackbone(dummy_backbone, fusion_type='conv3d', num_frames=8)
        
        # Test with temporal input
        x = torch.randn(2, 8, 3, 64, 64)  # B, T, C, H, W
        output = temporal_backbone(x)
        
        print(f"✓ TemporalBackbone forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shapes: {[o.shape for o in output]}")
    except Exception as e:
        print(f"✗ TemporalBackbone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test needle head
    print("\n2. Testing NeedleHead...")
    try:
        from damo.base_models.heads.needle_head import NeedleHead
        
        head = NeedleHead(
            num_classes=2,
            in_channels=[128, 256, 512],
            stacked_convs=0,
            reg_max=16,
            act='silu',
        )
        head.eval()
        
        # Create dummy inputs
        dummy_feats = [
            torch.randn(2, 128, 80, 80),
            torch.randn(2, 256, 40, 40),
            torch.randn(2, 512, 20, 20),
        ]
        
        with torch.no_grad():
            output = head(dummy_feats)
        
        print(f"✓ NeedleHead forward pass successful")
        print(f"  Output type: {type(output)}")
    except Exception as e:
        print(f"✗ NeedleHead test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test needle losses
    print("\n3. Testing Needle Losses...")
    try:
        from damo.base_models.losses.needle_losses import CircleLoss, DirectionLoss
        
        circle_loss = CircleLoss(loss_weight=1.0)
        direction_loss = DirectionLoss(loss_weight=1.0)
        
        # Test circle loss
        pred_radius = torch.randn(10, 1)
        target_radius = torch.randn(10, 1)
        loss_c = circle_loss(pred_radius, target_radius)
        print(f"✓ CircleLoss computed: {loss_c.item():.4f}")
        
        # Test direction loss
        pred_direction = torch.randn(10, 2)
        target_direction = torch.randn(10, 2)
        loss_d = direction_loss(pred_direction, target_direction)
        print(f"✓ DirectionLoss computed: {loss_d.item():.4f}")
        
    except Exception as e:
        print(f"✗ Needle losses test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    print("=" * 60)
    print("DAMO-YOLO Needle Detection Model Test Suite")
    print("=" * 60)
    
    tests = [
        ("Model Instantiation", test_needle_model_instantiation),
        ("Model Components", test_model_components),
        ("Model Forward Pass", test_needle_model_forward),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {test_name}")
        print(f"{'=' * 60}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Test Summary")
    print(f"{'=' * 60}")
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print(f"\n{'=' * 60}")
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print(f"{'=' * 60}")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
