#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

"""
Unit tests for BoundaryAwareDiceLoss.
"""

import torch
import torch.nn.functional as F
import unittest


class TestBoundaryAwareDiceLoss(unittest.TestCase):
    """Test cases for BoundaryAwareDiceLoss."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Import here to ensure the module is available
        from damo.base_models.losses.dice_loss import BoundaryAwareDiceLoss
        self.BoundaryAwareDiceLoss = BoundaryAwareDiceLoss
        
    def test_basic_initialization(self):
        """Test basic initialization of the loss."""
        loss_fn = self.BoundaryAwareDiceLoss(num_classes=21)
        self.assertEqual(loss_fn.num_classes, 21)
        self.assertIsNone(loss_fn.ignore_index)
        self.assertEqual(loss_fn.smooth, 1.0)
        self.assertEqual(loss_fn.boundary_weight, 2.0)
        
    def test_initialization_with_ignore_index(self):
        """Test initialization with ignore_index."""
        loss_fn = self.BoundaryAwareDiceLoss(
            num_classes=21, 
            ignore_index=0
        )
        self.assertEqual(loss_fn.ignore_index, 0)
        
    def test_perfect_prediction(self):
        """Test loss for perfect prediction (should be close to 0)."""
        num_classes = 5
        batch_size = 2
        H, W = 32, 32
        
        loss_fn = self.BoundaryAwareDiceLoss(num_classes=num_classes)
        
        # Create perfect prediction
        target = torch.randint(0, num_classes, (batch_size, H, W))
        pred = torch.zeros(batch_size, num_classes, H, W)
        
        # Set predictions to match target
        for b in range(batch_size):
            for c in range(num_classes):
                pred[b, c] = (target[b] == c).float()
        
        # Add small logit values to ensure softmax works
        pred = pred * 10.0  # Higher logits for correct class
        
        loss = loss_fn(pred, target)
        
        # Loss should be very small (close to 0)
        self.assertLess(loss.item(), 0.1, 
                       f"Perfect prediction should have low loss, got {loss.item()}")
        
    def test_worst_prediction(self):
        """Test loss for worst prediction (should be close to 1)."""
        num_classes = 5
        batch_size = 2
        H, W = 32, 32
        
        loss_fn = self.BoundaryAwareDiceLoss(num_classes=num_classes)
        
        # Create worst prediction (opposite of target)
        target = torch.zeros(batch_size, H, W, dtype=torch.long)  # All class 0
        pred = torch.zeros(batch_size, num_classes, H, W)
        pred[:, 1, :, :] = 10.0  # Predict class 1 instead of 0
        
        loss = loss_fn(pred, target)
        
        # Loss should be high (averaging over all classes gives moderate loss)
        self.assertGreater(loss.item(), 0.3,
                          f"Worst prediction should have high loss, got {loss.item()}")
        
    def test_ignore_index_functionality(self):
        """Test that ignore_index properly excludes pixels."""
        num_classes = 5
        batch_size = 2
        H, W = 32, 32
        ignore_idx = 0
        
        loss_fn = self.BoundaryAwareDiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_idx
        )
        
        # Create target with some ignore_index pixels
        target = torch.randint(0, num_classes, (batch_size, H, W))
        target[:, :10, :10] = ignore_idx  # Set a region to ignore_index
        
        pred = torch.randn(batch_size, num_classes, H, W)
        
        loss = loss_fn(pred, target)
        
        # Loss should be computed (ignore_index pixels excluded)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))
        
    def test_all_ignore_index(self):
        """Test behavior when all pixels are ignore_index."""
        num_classes = 5
        batch_size = 2
        H, W = 32, 32
        ignore_idx = 0
        
        loss_fn = self.BoundaryAwareDiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_idx
        )
        
        # Create target with all ignore_index pixels
        target = torch.full((batch_size, H, W), ignore_idx, dtype=torch.long)
        pred = torch.randn(batch_size, num_classes, H, W)
        
        loss = loss_fn(pred, target)
        
        # Loss should be zero or very small when all pixels are ignored
        self.assertLessEqual(loss.item(), 0.01)
        
    def test_boundary_detection(self):
        """Test that boundary mask is computed correctly."""
        num_classes = 3
        batch_size = 1
        H, W = 16, 16
        
        loss_fn = self.BoundaryAwareDiceLoss(num_classes=num_classes)
        
        # Create a simple target with clear boundaries
        target = torch.zeros(batch_size, H, W, dtype=torch.long)
        target[:, :, 8:] = 1  # Vertical boundary at column 8
        
        boundary_mask = loss_fn._compute_boundary_mask(target)
        
        # Check that boundary is detected
        self.assertEqual(boundary_mask.shape, (batch_size, H, W))
        # Boundary should have non-zero values
        self.assertGreater(boundary_mask.sum().item(), 0)
        
    def test_one_hot_encoding(self):
        """Test one-hot encoding functionality."""
        num_classes = 5
        batch_size = 2
        H, W = 8, 8
        
        loss_fn = self.BoundaryAwareDiceLoss(num_classes=num_classes)
        
        target = torch.randint(0, num_classes, (batch_size, H, W))
        one_hot = loss_fn._one_hot_encode(target)
        
        # Check shape
        self.assertEqual(one_hot.shape, (batch_size, num_classes, H, W))
        
        # Check that each pixel has exactly one class (sum over classes = 1)
        class_sum = one_hot.sum(dim=1)
        self.assertTrue(torch.allclose(class_sum, torch.ones_like(class_sum)))
        
    def test_one_hot_encoding_with_ignore_index(self):
        """Test one-hot encoding with ignore_index."""
        num_classes = 5
        batch_size = 2
        H, W = 8, 8
        ignore_idx = 0
        
        loss_fn = self.BoundaryAwareDiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_idx
        )
        
        target = torch.randint(0, num_classes, (batch_size, H, W))
        target[0, 0, 0] = ignore_idx  # Add an ignore_index pixel
        
        one_hot = loss_fn._one_hot_encode(target)
        
        # Check that ignore_index pixels have all zeros
        self.assertEqual(one_hot[0, :, 0, 0].sum().item(), 0.0)
        
    def test_reduction_modes(self):
        """Test different reduction modes."""
        num_classes = 5
        batch_size = 2
        H, W = 16, 16
        
        target = torch.randint(0, num_classes, (batch_size, H, W))
        pred = torch.randn(batch_size, num_classes, H, W)
        
        # Test 'mean' reduction
        loss_fn_mean = self.BoundaryAwareDiceLoss(
            num_classes=num_classes,
            reduction='mean'
        )
        loss_mean = loss_fn_mean(pred, target)
        self.assertEqual(loss_mean.dim(), 0)  # Scalar
        
        # Test 'sum' reduction
        loss_fn_sum = self.BoundaryAwareDiceLoss(
            num_classes=num_classes,
            reduction='sum'
        )
        loss_sum = loss_fn_sum(pred, target)
        self.assertEqual(loss_sum.dim(), 0)  # Scalar
        
        # Test 'none' reduction
        loss_fn_none = self.BoundaryAwareDiceLoss(
            num_classes=num_classes,
            reduction='none'
        )
        loss_none = loss_fn_none(pred, target)
        self.assertEqual(loss_none.dim(), 2)  # (N, C) shape
        
    def test_loss_weight(self):
        """Test that loss_weight scales the output correctly."""
        num_classes = 5
        batch_size = 2
        H, W = 16, 16
        
        target = torch.randint(0, num_classes, (batch_size, H, W))
        pred = torch.randn(batch_size, num_classes, H, W)
        
        # Compute loss with weight 1.0
        loss_fn_1 = self.BoundaryAwareDiceLoss(
            num_classes=num_classes,
            loss_weight=1.0
        )
        loss_1 = loss_fn_1(pred, target)
        
        # Compute loss with weight 2.0
        loss_fn_2 = self.BoundaryAwareDiceLoss(
            num_classes=num_classes,
            loss_weight=2.0
        )
        loss_2 = loss_fn_2(pred, target)
        
        # Loss with weight 2.0 should be double
        self.assertTrue(torch.allclose(loss_2, loss_1 * 2.0, rtol=1e-5))
        
    def test_gradient_flow(self):
        """Test that gradients flow correctly through the loss."""
        num_classes = 5
        batch_size = 2
        H, W = 16, 16
        
        loss_fn = self.BoundaryAwareDiceLoss(num_classes=num_classes)
        
        target = torch.randint(0, num_classes, (batch_size, H, W))
        pred = torch.randn(batch_size, num_classes, H, W, requires_grad=True)
        
        loss = loss_fn(pred, target)
        loss.backward()
        
        # Check that gradients are computed
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())
        
    def test_multiclass_segmentation(self):
        """Test with realistic multiclass segmentation scenario."""
        num_classes = 21  # e.g., Pascal VOC
        batch_size = 4
        H, W = 64, 64
        ignore_idx = 255  # Common ignore index
        
        loss_fn = self.BoundaryAwareDiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_idx,
            boundary_weight=2.5
        )
        
        # Create realistic target with boundaries and ignore regions
        target = torch.randint(0, num_classes, (batch_size, H, W))
        target[:, 0, :] = ignore_idx  # Border region
        target[:, -1, :] = ignore_idx  # Border region
        
        pred = torch.randn(batch_size, num_classes, H, W)
        
        loss = loss_fn(pred, target)
        
        # Loss should be a valid scalar
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
        
    def test_boundary_weight_effect(self):
        """Test that boundary_weight affects the loss correctly."""
        num_classes = 3
        batch_size = 1
        H, W = 32, 32
        
        # Create target with clear boundary
        target = torch.zeros(batch_size, H, W, dtype=torch.long)
        target[:, :, 16:] = 1  # Vertical boundary
        
        pred = torch.randn(batch_size, num_classes, H, W)
        
        # Compute loss with low boundary weight
        loss_fn_low = self.BoundaryAwareDiceLoss(
            num_classes=num_classes,
            boundary_weight=1.0
        )
        loss_low = loss_fn_low(pred.clone(), target)
        
        # Compute loss with high boundary weight
        loss_fn_high = self.BoundaryAwareDiceLoss(
            num_classes=num_classes,
            boundary_weight=5.0
        )
        loss_high = loss_fn_high(pred.clone(), target)
        
        # Losses should be different (boundary weight affects computation)
        self.assertNotEqual(loss_low.item(), loss_high.item())
        
    def test_input_validation(self):
        """Test input validation and error handling."""
        num_classes = 5
        batch_size = 2
        H, W = 16, 16
        
        loss_fn = self.BoundaryAwareDiceLoss(num_classes=num_classes)
        
        # Test with wrong number of channels
        target = torch.randint(0, num_classes, (batch_size, H, W))
        pred_wrong = torch.randn(batch_size, num_classes + 1, H, W)
        
        with self.assertRaises(AssertionError):
            loss_fn(pred_wrong, target)
            
    def test_cpu_and_cuda_consistency(self):
        """Test that loss works on both CPU and CUDA (if available)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        num_classes = 5
        batch_size = 2
        H, W = 16, 16
        
        loss_fn = self.BoundaryAwareDiceLoss(num_classes=num_classes)
        
        target = torch.randint(0, num_classes, (batch_size, H, W))
        pred = torch.randn(batch_size, num_classes, H, W)
        
        # CPU computation
        loss_cpu = loss_fn(pred, target)
        
        # CUDA computation
        loss_cuda = loss_fn(pred.cuda(), target.cuda())
        
        # Results should be close
        self.assertTrue(
            torch.allclose(loss_cpu, loss_cuda.cpu(), rtol=1e-4, atol=1e-5)
        )


if __name__ == '__main__':
    unittest.main()
