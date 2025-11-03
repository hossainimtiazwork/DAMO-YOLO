#!/usr/bin/env python3
"""
Example script demonstrating how to use the spatio-temporal needle detection model.
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from damo.config.base import parse_config
from damo.detectors.temporal_detector import build_local_model


def example_single_frame_inference():
    """Example: Single frame needle detection"""
    print("=" * 60)
    print("Example 1: Single Frame Needle Detection")
    print("=" * 60)
    
    # Load configuration
    config = parse_config('./configs/damoyolo_needle_detection.py')
    
    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_local_model(config, device)
    model.eval()
    
    print(f"Model loaded on {device}")
    
    # Prepare single frame input
    # In practice, load from image file using PIL or cv2
    image = torch.randn(1, 3, 640, 640).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image)
    
    print(f"Output type: {type(outputs)}")
    if isinstance(outputs, (list, tuple)):
        print(f"Number of outputs: {len(outputs)}")
        if len(outputs) > 0 and outputs[0] is not None:
            print(f"Detection shape: {outputs[0].shape}")
            # outputs[0] format: [x1, y1, x2, y2, confidence, class_id]
            # Additional outputs like radius and direction would be in subsequent elements
    
    print("\n✓ Single frame inference completed successfully!\n")


def example_temporal_sequence_inference():
    """Example: Temporal sequence needle detection"""
    print("=" * 60)
    print("Example 2: Temporal Sequence Needle Detection")
    print("=" * 60)
    
    # Load configuration
    config = parse_config('./configs/damoyolo_needle_detection.py')
    
    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_local_model(config, device)
    model.eval()
    
    print(f"Model loaded on {device}")
    
    # Prepare temporal sequence input (8 frames)
    # In practice, load sequence of images
    num_frames = 8
    sequence = torch.randn(1, num_frames, 3, 640, 640).to(device)
    
    print(f"Input sequence shape: {sequence.shape}")
    print(f"  Batch: {sequence.shape[0]}")
    print(f"  Frames: {sequence.shape[1]}")
    print(f"  Channels: {sequence.shape[2]}")
    print(f"  Height: {sequence.shape[3]}")
    print(f"  Width: {sequence.shape[4]}")
    
    # Run inference
    with torch.no_grad():
        outputs = model(sequence)
    
    print(f"\nOutput type: {type(outputs)}")
    if isinstance(outputs, (list, tuple)):
        print(f"Number of outputs: {len(outputs)}")
    
    print("\n✓ Temporal sequence inference completed successfully!\n")


def example_training_setup():
    """Example: Setup for training"""
    print("=" * 60)
    print("Example 3: Training Setup")
    print("=" * 60)
    
    # Load configuration
    config = parse_config('./configs/damoyolo_needle_detection.py')
    
    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_local_model(config, device)
    model.train()
    
    print(f"Model in training mode on {device}")
    
    # Prepare dummy training data
    # In practice, this comes from your dataloader
    batch_size = 2
    num_frames = 8
    images = torch.randn(batch_size, num_frames, 3, 640, 640).to(device)
    
    # Prepare dummy targets (labels)
    # This is a simplified example - actual implementation depends on your dataset
    from damo.structures.bounding_box import BoxList
    
    targets = []
    for i in range(batch_size):
        # Create dummy ground truth for each image
        # 2 needles per image for this example
        num_objects = 2
        
        boxes = torch.rand(num_objects, 4) * 640
        boxes[:, 2:] += boxes[:, :2]  # Convert to x2, y2 format
        
        target = BoxList(boxes, (640, 640), mode="xyxy")
        target.add_field('labels', torch.tensor([0, 1], dtype=torch.long))  # 2 needle types
        target.add_field('radius', torch.tensor([15.0, 20.0]))  # Needle radius
        target.add_field('direction', torch.tensor([0.5, 1.2]))  # Needle direction in radians
        
        targets.append(target)
    
    # Forward pass with targets (training mode)
    outputs = model(images, targets=targets)
    
    print(f"\nTraining outputs (losses):")
    if isinstance(outputs, dict):
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")
            else:
                print(f"  {key}: {value}")
    
    print("\n✓ Training setup completed successfully!\n")


def example_loss_computation():
    """Example: Using individual loss functions"""
    print("=" * 60)
    print("Example 4: Individual Loss Functions")
    print("=" * 60)
    
    from damo.base_models.losses.needle_losses import CircleLoss, DirectionLoss
    
    # Circle Loss for radius
    circle_loss = CircleLoss(loss_weight=1.0)
    
    pred_radius = torch.tensor([[15.5], [20.3], [12.8]])
    target_radius = torch.tensor([[15.0], [20.0], [13.0]])
    
    loss_r = circle_loss(pred_radius, target_radius)
    print(f"Circle Loss (radius): {loss_r.item():.4f}")
    
    # Direction Loss
    direction_loss = DirectionLoss(loss_weight=1.0)
    
    # Predictions and targets in (sin, cos) format
    import math
    angles_pred = torch.tensor([0.5, 1.0, 1.5])  # radians
    angles_target = torch.tensor([0.6, 1.1, 1.4])  # radians
    
    pred_direction = torch.stack([torch.sin(angles_pred), torch.cos(angles_pred)], dim=1)
    target_direction = torch.stack([torch.sin(angles_target), torch.cos(angles_target)], dim=1)
    
    loss_d = direction_loss(pred_direction, target_direction)
    print(f"Direction Loss: {loss_d.item():.4f}")
    
    print("\n✓ Loss computation completed successfully!\n")


def example_temporal_fusion_strategies():
    """Example: Different temporal fusion strategies"""
    print("=" * 60)
    print("Example 5: Temporal Fusion Strategies")
    print("=" * 60)
    
    from damo.base_models.backbones.temporal_backbone import TemporalFusion
    
    fusion_types = ['conv3d', 'attention', 'lstm', 'average']
    
    for fusion_type in fusion_types:
        print(f"\nTesting {fusion_type} fusion...")
        
        temporal_fusion = TemporalFusion(
            in_channels=64,
            fusion_type=fusion_type,
            num_frames=8
        )
        
        # Input: (B, T, C, H, W)
        x = torch.randn(2, 8, 64, 32, 32)
        
        output = temporal_fusion(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  ✓ {fusion_type} fusion successful")
    
    print("\n✓ All fusion strategies tested successfully!\n")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Spatio-Temporal Needle Detection Examples")
    print("=" * 60 + "\n")
    
    examples = [
        ("Single Frame Inference", example_single_frame_inference),
        ("Temporal Sequence Inference", example_temporal_sequence_inference),
        ("Training Setup", example_training_setup),
        ("Loss Computation", example_loss_computation),
        ("Temporal Fusion Strategies", example_temporal_fusion_strategies),
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"✗ Example '{name}' failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
