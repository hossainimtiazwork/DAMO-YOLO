#!/usr/bin/env python3
"""
Example usage of BoundaryAwareLoss for multiclass segmentation.

This example demonstrates:
1. Basic usage of BoundaryAwareLoss
2. Using ignore_class for background
3. Adjusting boundary_weight
4. Visualizing boundary emphasis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from damo.base_models.losses import BoundaryAwareLoss


def example_basic_usage():
    """Example 1: Basic usage of BoundaryAwareLoss"""
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    # Setup
    batch_size = 4
    num_classes = 5
    height, width = 32, 32
    
    # Create loss function
    loss_fn = BoundaryAwareLoss(
        num_classes=num_classes,
        boundary_weight=2.0,
        reduction='mean'
    )
    
    # Create dummy data
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Compute loss
    loss = loss_fn(predictions, targets)
    
    print(f"Batch size: {batch_size}")
    print(f"Number of classes: {num_classes}")
    print(f"Image size: {height}x{width}")
    print(f"Loss value: {loss.item():.4f}")


def example_with_background_ignore():
    """Example 2: Using ignore_class to ignore background"""
    print("\n" + "="*60)
    print("Example 2: Ignoring Background Class")
    print("="*60)
    
    # Setup
    batch_size = 4
    num_classes = 21  # Similar to PASCAL VOC
    height, width = 64, 64
    background_class = 0
    
    # Create loss function that ignores background
    loss_fn = BoundaryAwareLoss(
        num_classes=num_classes,
        ignore_class=background_class,
        boundary_weight=2.5,
        reduction='mean'
    )
    
    # Create dummy data
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Compute loss
    loss = loss_fn(predictions, targets)
    
    # Count background pixels
    num_background = (targets == background_class).sum().item()
    num_foreground = (targets != background_class).sum().item()
    
    print(f"Number of classes: {num_classes}")
    print(f"Ignored class (background): {background_class}")
    print(f"Background pixels: {num_background} ({100*num_background/(height*width*batch_size):.1f}%)")
    print(f"Foreground pixels: {num_foreground} ({100*num_foreground/(height*width*batch_size):.1f}%)")
    print(f"Loss value: {loss.item():.4f}")


def example_boundary_weight_comparison():
    """Example 3: Comparing different boundary weights"""
    print("\n" + "="*60)
    print("Example 3: Boundary Weight Comparison")
    print("="*60)
    
    # Setup
    batch_size = 2
    num_classes = 5
    height, width = 32, 32
    
    # Create dummy data (fixed for fair comparison)
    torch.manual_seed(42)
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test different boundary weights
    boundary_weights = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    print(f"Testing different boundary weights:")
    print(f"{'Boundary Weight':<20} {'Loss Value':<15}")
    print("-" * 35)
    
    for bw in boundary_weights:
        loss_fn = BoundaryAwareLoss(
            num_classes=num_classes,
            boundary_weight=bw,
            reduction='mean'
        )
        loss = loss_fn(predictions, targets)
        print(f"{bw:<20.1f} {loss.item():<15.4f}")
    
    print("\nNote: Higher boundary_weight emphasizes boundaries more")


def example_segmentation_training():
    """Example 4: Using in a training loop"""
    print("\n" + "="*60)
    print("Example 4: Training Loop Integration")
    print("="*60)
    
    # Simple model
    class SimpleSegmentationModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, num_classes, 1)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.conv2(x)
            return x
    
    # Setup
    num_classes = 5
    model = SimpleSegmentationModel(num_classes)
    
    loss_fn = BoundaryAwareLoss(
        num_classes=num_classes,
        ignore_class=0,
        boundary_weight=2.5,
        reduction='mean'
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Simulate training
    print("Simulating training for 5 iterations...")
    print(f"{'Iteration':<12} {'Loss':<12}")
    print("-" * 24)
    
    for i in range(5):
        # Create dummy batch
        images = torch.randn(2, 3, 32, 32)
        masks = torch.randint(0, num_classes, (2, 32, 32))
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_fn(predictions, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"{i+1:<12} {loss.item():<12.4f}")
    
    print("\nTraining completed successfully!")


def example_analyze_boundary_emphasis():
    """Example 5: Analyzing boundary vs non-boundary loss"""
    print("\n" + "="*60)
    print("Example 5: Boundary Emphasis Analysis")
    print("="*60)
    
    # Create a simple segmentation mask with clear boundaries
    batch_size = 1
    num_classes = 3
    height, width = 16, 16
    
    # Create a structured target with clear regions
    target = torch.zeros(batch_size, height, width, dtype=torch.long)
    target[:, :8, :8] = 0  # Top-left: class 0
    target[:, :8, 8:] = 1  # Top-right: class 1
    target[:, 8:, :] = 2   # Bottom: class 2
    
    # Random predictions
    predictions = torch.randn(batch_size, num_classes, height, width)
    
    # Compute loss with reduction='none' to get per-pixel loss
    loss_fn = BoundaryAwareLoss(
        num_classes=num_classes,
        boundary_weight=2.0,
        reduction='none'
    )
    
    loss_map = loss_fn(predictions, target)
    
    # Identify boundary pixels manually
    from damo.base_models.losses.boundary_aware_loss import compute_boundary_mask
    boundary_mask = compute_boundary_mask(target)
    
    # Analyze loss distribution
    is_boundary = boundary_mask[0] > 0.3  # Threshold for boundary
    
    if is_boundary.any():
        boundary_loss = loss_map[0][is_boundary].mean().item()
        non_boundary_loss = loss_map[0][~is_boundary].mean().item()
        ratio = boundary_loss / non_boundary_loss if non_boundary_loss > 0 else 0
        
        print(f"Target shape: {target.shape}")
        print(f"Boundary pixels: {is_boundary.sum().item()}")
        print(f"Non-boundary pixels: {(~is_boundary).sum().item()}")
        print(f"\nAverage loss at boundaries: {boundary_loss:.4f}")
        print(f"Average loss at non-boundaries: {non_boundary_loss:.4f}")
        print(f"Boundary/Non-boundary ratio: {ratio:.2f}x")
    else:
        print("No boundaries detected in the target mask")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("BoundaryAwareLoss Examples")
    print("="*60)
    
    example_basic_usage()
    example_with_background_ignore()
    example_boundary_weight_comparison()
    example_segmentation_training()
    example_analyze_boundary_emphasis()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
