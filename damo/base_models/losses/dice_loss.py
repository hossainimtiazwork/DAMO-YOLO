# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryAwareDiceLoss(nn.Module):
    """
    Boundary-Aware Dice Loss for multiclass semantic segmentation.
    
    This loss combines the standard Dice loss with boundary awareness,
    giving more weight to boundary regions while allowing specific indices
    (e.g., background) to be ignored.
    
    Args:
        num_classes (int): Number of classes in the segmentation task.
        ignore_index (int, optional): Index to ignore in loss calculation 
            (typically background class). Default: None.
        smooth (float): Smoothing factor to avoid division by zero. Default: 1.0.
        boundary_weight (float): Weight multiplier for boundary regions. 
            Higher values increase focus on boundaries. Default: 2.0.
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'.
        loss_weight (float): Weight of the loss. Default: 1.0.
    
    Shape:
        - Input: (N, C, H, W) where N is batch size, C is number of classes,
                 H and W are height and width.
        - Target: (N, H, W) with class indices or (N, C, H, W) with one-hot encoding.
        - Output: scalar if reduction is 'mean' or 'sum', otherwise (N, C).
    
    Example:
        >>> loss_fn = BoundaryAwareDiceLoss(num_classes=21, ignore_index=0)
        >>> pred = torch.randn(2, 21, 128, 128)
        >>> target = torch.randint(0, 21, (2, 128, 128))
        >>> loss = loss_fn(pred, target)
    """
    
    def __init__(
        self,
        num_classes,
        ignore_index=None,
        smooth=1.0,
        boundary_weight=2.0,
        reduction='mean',
        loss_weight=1.0
    ):
        super(BoundaryAwareDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.boundary_weight = boundary_weight
        self.reduction = reduction
        self.loss_weight = loss_weight
        
    def _compute_boundary_mask(self, target):
        """
        Compute boundary mask using morphological operations.
        
        Args:
            target (Tensor): Target segmentation mask of shape (N, H, W).
            
        Returns:
            Tensor: Boundary mask of shape (N, H, W) with values in [0, 1].
        """
        # Pad the target to handle boundaries at edges
        target_padded = F.pad(target.float(), (1, 1, 1, 1), mode='replicate')
        
        # Compute gradient magnitude using Sobel-like filters
        # Horizontal gradient
        diff_h = (target_padded[:, 1:-1, 2:] != target_padded[:, 1:-1, :-2]).float()
        # Vertical gradient
        diff_v = (target_padded[:, 2:, 1:-1] != target_padded[:, :-2, 1:-1]).float()
        
        # Combine to get boundary mask
        boundary_mask = torch.clamp(diff_h + diff_v, 0, 1)
        
        return boundary_mask
    
    def _one_hot_encode(self, target):
        """
        Convert class indices to one-hot encoding.
        
        Args:
            target (Tensor): Target with shape (N, H, W) containing class indices.
            
        Returns:
            Tensor: One-hot encoded target with shape (N, C, H, W).
        """
        N, H, W = target.shape
        one_hot = torch.zeros(N, self.num_classes, H, W, 
                             dtype=torch.float32, device=target.device)
        
        # Create one-hot encoding, handling ignore_index
        if self.ignore_index is not None:
            # Create a mask for valid pixels (not ignore_index)
            valid_mask = (target != self.ignore_index)
            # Only encode valid pixels
            target_clipped = target.clone()
            target_clipped[~valid_mask] = 0  # Temporarily set to 0 for scatter
            one_hot.scatter_(1, target_clipped.unsqueeze(1), 1.0)
            # Zero out the ignore_index channel and invalid pixels
            if self.ignore_index >= 0 and self.ignore_index < self.num_classes:
                one_hot[:, self.ignore_index, :, :] = 0
            # Zero out all channels for invalid pixels
            one_hot = one_hot * valid_mask.unsqueeze(1).float()
        else:
            one_hot.scatter_(1, target.unsqueeze(1), 1.0)
            
        return one_hot
    
    def forward(self, pred, target):
        """
        Forward pass of the Boundary-Aware Dice Loss.
        
        Args:
            pred (Tensor): Predicted logits with shape (N, C, H, W).
            target (Tensor): Ground truth with shape (N, H, W) containing class indices
                           or (N, C, H, W) for one-hot encoded targets.
                           
        Returns:
            Tensor: Computed loss value.
        """
        assert pred.dim() == 4, f"Expected 4D input, got {pred.dim()}D"
        
        N, C, H, W = pred.shape
        assert C == self.num_classes, \
            f"pred channels ({C}) must match num_classes ({self.num_classes})"
        
        # Convert target to one-hot if needed
        if target.dim() == 3:
            target_indices = target
            target_one_hot = self._one_hot_encode(target)
        else:
            target_one_hot = target
            target_indices = torch.argmax(target, dim=1)
        
        # Apply softmax to predictions
        pred_softmax = F.softmax(pred, dim=1)
        
        # Compute boundary mask
        boundary_mask = self._compute_boundary_mask(target_indices)
        
        # Create weight mask: higher weight for boundaries
        weight_mask = 1.0 + (self.boundary_weight - 1.0) * boundary_mask
        weight_mask = weight_mask.unsqueeze(1)  # (N, 1, H, W)
        
        # Create ignore mask if ignore_index is specified
        if self.ignore_index is not None:
            valid_mask = (target_indices != self.ignore_index).float()
            valid_mask = valid_mask.unsqueeze(1)  # (N, 1, H, W)
            weight_mask = weight_mask * valid_mask
        
        # Compute Dice loss per class
        dice_losses = []
        
        for class_idx in range(self.num_classes):
            # Skip ignore_index class
            if self.ignore_index is not None and class_idx == self.ignore_index:
                continue
                
            pred_class = pred_softmax[:, class_idx:class_idx+1, :, :]  # (N, 1, H, W)
            target_class = target_one_hot[:, class_idx:class_idx+1, :, :]  # (N, 1, H, W)
            
            # Compute intersection and sums with spatial weighting
            # Weight is applied to each pixel position, not to pred/target separately
            intersection = (pred_class * target_class * weight_mask).sum(dim=(2, 3))
            pred_sum = (pred_class * weight_mask).sum(dim=(2, 3))
            target_sum = (target_class * weight_mask).sum(dim=(2, 3))
            
            # Compute Dice coefficient
            dice_coeff = (2.0 * intersection + self.smooth) / \
                        (pred_sum + target_sum + self.smooth)
            
            # Dice loss is 1 - Dice coefficient
            # Squeeze to remove extra dimension (N, 1) -> (N,)
            dice_loss = (1.0 - dice_coeff).squeeze(-1)
            
            dice_losses.append(dice_loss)
        
        # Stack losses for all classes
        if len(dice_losses) > 0:
            dice_losses = torch.stack(dice_losses, dim=1)  # (N, num_valid_classes)
        else:
            # If all classes are ignored, return zero loss
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Apply reduction
        if self.reduction == 'none':
            loss = dice_losses
        elif self.reduction == 'mean':
            loss = dice_losses.mean()
        elif self.reduction == 'sum':
            loss = dice_losses.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")
        
        return self.loss_weight * loss
