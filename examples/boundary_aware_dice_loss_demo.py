"""
Example demonstrating the usage of Boundary Aware Dice Loss.

This script shows how to use the BoundaryAwareDiceLoss in your training pipeline.
The loss is particularly useful for segmentation tasks where boundary precision is critical.
"""

import torch
import torch.nn as nn
from damo.base_models.losses.boundary_aware_dice_loss import BoundaryAwareDiceLoss


def example_basic_usage():
    """Basic usage example."""
    print("="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    # Initialize the loss function
    loss_fn = BoundaryAwareDiceLoss(
        smooth=1e-5,
        boundary_weight=2.0,  # Increase weight for boundary regions
        boundary_kernel_size=3,
        reduction='mean',
        loss_weight=1.0
    )
    
    # Create sample prediction and target
    batch_size, channels, height, width = 2, 1, 64, 64
    pred = torch.randn(batch_size, channels, height, width)
    target = torch.randint(0, 2, (batch_size, channels, height, width)).float()
    
    # Compute loss
    loss = loss_fn(pred, target)
    
    print(f"Prediction shape: {pred.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Loss value: {loss.item():.6f}")
    print()


def example_with_training_loop():
    """Example showing usage in a training loop."""
    print("="*60)
    print("Example 2: Usage in Training Loop")
    print("="*60)
    
    # Simple model for demonstration
    class SimpleSegmentationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 1, 1)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.conv2(x)
            return x
    
    # Initialize model, loss, and optimizer
    model = SimpleSegmentationModel()
    loss_fn = BoundaryAwareDiceLoss(boundary_weight=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Dummy training data
    inputs = torch.randn(2, 3, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    
    loss.backward()
    optimizer.step()
    
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Loss value: {loss.item():.6f}")
    print(f"Gradients computed: {outputs.requires_grad}")
    print()


def example_boundary_weight_comparison():
    """Example comparing different boundary weight values."""
    print("="*60)
    print("Example 3: Boundary Weight Comparison")
    print("="*60)
    
    # Create target with clear boundaries
    target = torch.zeros(1, 1, 64, 64)
    target[:, :, 20:44, 20:44] = 1.0  # Square object
    
    # Create prediction with boundary errors
    pred = target.clone()
    pred[:, :, 20:22, :] = 0.3  # Error at top boundary
    pred[:, :, 42:44, :] = 0.3  # Error at bottom boundary
    
    # Compare losses with different boundary weights
    boundary_weights = [1.0, 2.0, 3.0, 5.0]
    
    print("Comparing boundary weights on predictions with boundary errors:")
    print()
    
    for bw in boundary_weights:
        loss_fn = BoundaryAwareDiceLoss(boundary_weight=bw)
        loss = loss_fn(pred, target)
        print(f"  boundary_weight={bw:.1f}: loss={loss.item():.6f}")
    
    print()
    print("Observation: Higher boundary weights penalize boundary errors more.")
    print()


def example_multi_channel():
    """Example with multi-channel segmentation."""
    print("="*60)
    print("Example 4: Multi-Channel Segmentation")
    print("="*60)
    
    # Multi-class segmentation (e.g., 3 classes)
    loss_fn = BoundaryAwareDiceLoss()
    
    batch_size, num_classes, height, width = 2, 3, 64, 64
    pred = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, 2, (batch_size, num_classes, height, width)).float()
    
    loss = loss_fn(pred, target)
    
    print(f"Multi-channel prediction shape: {pred.shape}")
    print(f"Multi-channel target shape: {target.shape}")
    print(f"Loss value: {loss.item():.6f}")
    print()


def example_with_custom_parameters():
    """Example showing different parameter configurations."""
    print("="*60)
    print("Example 5: Custom Parameter Configurations")
    print("="*60)
    
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    
    # Configuration 1: Small boundary emphasis
    loss_fn1 = BoundaryAwareDiceLoss(
        boundary_weight=1.5,
        boundary_kernel_size=3
    )
    loss1 = loss_fn1(pred, target)
    print(f"Config 1 (small boundary emphasis): {loss1.item():.6f}")
    
    # Configuration 2: Strong boundary emphasis
    loss_fn2 = BoundaryAwareDiceLoss(
        boundary_weight=3.0,
        boundary_kernel_size=5
    )
    loss2 = loss_fn2(pred, target)
    print(f"Config 2 (strong boundary emphasis): {loss2.item():.6f}")
    
    # Configuration 3: With loss weight scaling
    loss_fn3 = BoundaryAwareDiceLoss(
        boundary_weight=2.0,
        loss_weight=0.5  # Scale the overall loss
    )
    loss3 = loss_fn3(pred, target)
    print(f"Config 3 (with loss_weight=0.5): {loss3.item():.6f}")
    
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*8 + "Boundary Aware Dice Loss Examples" + " "*16 + "║")
    print("╚" + "="*58 + "╝")
    print()
    
    example_basic_usage()
    example_with_training_loop()
    example_boundary_weight_comparison()
    example_multi_channel()
    example_with_custom_parameters()
    
    print("="*60)
    print("All examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    main()
