#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

"""
Example usage of BoundaryAwareDiceLoss for multiclass semantic segmentation.

This script demonstrates:
1. Basic usage with default parameters
2. Usage with ignore_index for background
3. Adjusting boundary awareness
4. Integration in a training loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from damo.base_models.losses.dice_loss import BoundaryAwareDiceLoss


def example_basic_usage():
    """Example 1: Basic usage with default parameters."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Initialize loss function for 21 classes (e.g., Pascal VOC)
    num_classes = 21
    loss_fn = BoundaryAwareDiceLoss(num_classes=num_classes)
    
    # Simulate model predictions and ground truth
    batch_size = 4
    height, width = 128, 128
    
    # Predictions: (N, C, H, W) - logits before softmax
    predictions = torch.randn(batch_size, num_classes, height, width)
    
    # Ground truth: (N, H, W) - class indices
    ground_truth = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Compute loss
    loss = loss_fn(predictions, ground_truth)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    print(f"Loss value: {loss.item():.4f}")
    print()


def example_with_ignore_index():
    """Example 2: Using ignore_index to skip background class."""
    print("=" * 60)
    print("Example 2: With Ignore Index (Background)")
    print("=" * 60)
    
    # Initialize loss function with ignore_index=0 for background
    num_classes = 21
    ignore_index = 0  # Background class
    loss_fn = BoundaryAwareDiceLoss(
        num_classes=num_classes,
        ignore_index=ignore_index
    )
    
    batch_size = 4
    height, width = 128, 128
    
    # Predictions
    predictions = torch.randn(batch_size, num_classes, height, width)
    
    # Ground truth with background pixels (class 0)
    ground_truth = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Set some regions as background (will be ignored in loss)
    ground_truth[:, :20, :] = ignore_index  # Top border
    ground_truth[:, -20:, :] = ignore_index  # Bottom border
    
    # Compute loss (background pixels are ignored)
    loss = loss_fn(predictions, ground_truth)
    
    print(f"Num classes: {num_classes}")
    print(f"Ignore index: {ignore_index}")
    print(f"Background pixels: {(ground_truth == ignore_index).sum().item()}")
    print(f"Foreground pixels: {(ground_truth != ignore_index).sum().item()}")
    print(f"Loss value: {loss.item():.4f}")
    print()


def example_boundary_weight():
    """Example 3: Adjusting boundary awareness."""
    print("=" * 60)
    print("Example 3: Adjusting Boundary Weight")
    print("=" * 60)
    
    num_classes = 5
    batch_size = 2
    height, width = 64, 64
    
    predictions = torch.randn(batch_size, num_classes, height, width)
    ground_truth = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test different boundary weights
    boundary_weights = [1.0, 2.0, 5.0]
    
    for bw in boundary_weights:
        loss_fn = BoundaryAwareDiceLoss(
            num_classes=num_classes,
            boundary_weight=bw
        )
        loss = loss_fn(predictions, ground_truth)
        print(f"Boundary weight: {bw:.1f} -> Loss: {loss.item():.4f}")
    
    print()


def example_training_integration():
    """Example 4: Integration in a training loop."""
    print("=" * 60)
    print("Example 4: Training Loop Integration")
    print("=" * 60)
    
    # Simple segmentation model (placeholder)
    class SimpleSegmentationModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, num_classes, kernel_size=1)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.conv2(x)
            return x
    
    # Setup
    num_classes = 21
    ignore_index = 255  # Common ignore value
    
    model = SimpleSegmentationModel(num_classes)
    loss_fn = BoundaryAwareDiceLoss(
        num_classes=num_classes,
        ignore_index=ignore_index,
        boundary_weight=2.0
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Simulate one training iteration
    batch_size = 2
    height, width = 64, 64
    
    # Input images: (N, 3, H, W)
    images = torch.randn(batch_size, 3, height, width)
    
    # Ground truth: (N, H, W)
    masks = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Forward pass
    predictions = model(images)
    
    # Compute loss
    loss = loss_fn(predictions, masks)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Training iteration completed")
    print(f"Input shape: {images.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Ground truth shape: {masks.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Gradients computed: {predictions.grad is not None or any(p.grad is not None for p in model.parameters())}")
    print()


def example_custom_parameters():
    """Example 5: Custom parameters for specific use cases."""
    print("=" * 60)
    print("Example 5: Custom Parameters")
    print("=" * 60)
    
    # Medical image segmentation with high boundary emphasis
    loss_fn_medical = BoundaryAwareDiceLoss(
        num_classes=4,  # Background, organ1, organ2, organ3
        ignore_index=0,  # Background
        smooth=0.1,  # Smaller smooth for sharper boundaries
        boundary_weight=3.0,  # Higher weight on boundaries
        reduction='mean',
        loss_weight=1.0
    )
    
    # Autonomous driving segmentation with less boundary emphasis
    loss_fn_driving = BoundaryAwareDiceLoss(
        num_classes=19,  # Cityscapes classes
        ignore_index=255,  # Unlabeled
        smooth=1.0,
        boundary_weight=1.5,  # Moderate boundary weight
        reduction='mean',
        loss_weight=0.5  # Combined with other losses
    )
    
    print("Medical imaging configuration:")
    print(f"  - Classes: {loss_fn_medical.num_classes}")
    print(f"  - Boundary weight: {loss_fn_medical.boundary_weight}")
    print(f"  - Smooth: {loss_fn_medical.smooth}")
    
    print("\nAutonomous driving configuration:")
    print(f"  - Classes: {loss_fn_driving.num_classes}")
    print(f"  - Boundary weight: {loss_fn_driving.boundary_weight}")
    print(f"  - Loss weight: {loss_fn_driving.loss_weight}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BoundaryAwareDiceLoss Usage Examples")
    print("=" * 60 + "\n")
    
    # Run all examples
    example_basic_usage()
    example_with_ignore_index()
    example_boundary_weight()
    example_training_integration()
    example_custom_parameters()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
