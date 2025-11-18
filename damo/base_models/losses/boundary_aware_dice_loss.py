# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """
    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def compute_boundary_weight(target, kernel_size=3):
    """Compute boundary weight map from target mask.
    
    Args:
        target (torch.Tensor): Target segmentation mask with shape (N, H, W) or (N, C, H, W).
        kernel_size (int): Kernel size for computing boundaries. Default: 3.
    
    Returns:
        torch.Tensor: Boundary weight map with same shape as target.
    """
    if target.dim() == 3:
        # Add channel dimension if not present
        target = target.unsqueeze(1)
    
    # Ensure target is float for convolution
    target_float = target.float()
    
    # Create a simple edge detection kernel (Laplacian-like)
    # This helps detect boundaries by finding discontinuities
    batch_size, channels, height, width = target_float.shape
    
    # Compute gradients using max pooling and erosion-like operation
    # This creates a boundary map where edges have higher values
    padding = kernel_size // 2
    
    # Dilate the mask
    max_pool = F.max_pool2d(target_float, kernel_size=kernel_size, 
                            stride=1, padding=padding)
    # Erode the mask  
    min_pool = -F.max_pool2d(-target_float, kernel_size=kernel_size,
                             stride=1, padding=padding)
    
    # Boundary is where max and min differ
    boundary = (max_pool - min_pool).clamp(min=0, max=1)
    
    return boundary


@weighted_loss
def boundary_aware_dice_loss(pred, target, smooth=1e-5, boundary_weight=2.0, 
                             boundary_kernel_size=3):
    """Boundary Aware Dice Loss.
    
    This loss combines the traditional Dice loss with boundary awareness.
    It gives more weight to predictions near object boundaries, which is
    particularly useful for segmentation tasks where boundary precision is critical.
    
    Args:
        pred (torch.Tensor): Predicted segmentation probabilities with shape (N, C, H, W) or (N, H, W).
        target (torch.Tensor): Target segmentation mask with shape (N, C, H, W) or (N, H, W).
        smooth (float): Smoothing factor to avoid division by zero. Default: 1e-5.
        boundary_weight (float): Weight factor for boundary regions. Higher values give more
                                importance to boundary regions. Default: 2.0.
        boundary_kernel_size (int): Kernel size for boundary detection. Default: 3.
    
    Returns:
        torch.Tensor: Computed boundary aware dice loss.
    """
    # Ensure tensors have the same shape
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    
    # Apply sigmoid if pred is logits
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = torch.sigmoid(pred)
    
    # Flatten spatial dimensions for each sample and channel
    batch_size, channels = pred.shape[0], pred.shape[1]
    pred_flat = pred.reshape(batch_size, channels, -1)
    target_flat = target.reshape(batch_size, channels, -1)
    
    # Compute boundary weight map
    boundary_map = compute_boundary_weight(target, kernel_size=boundary_kernel_size)
    boundary_flat = boundary_map.reshape(batch_size, channels, -1)
    
    # Create weight map: give more weight to boundary regions
    weight_map = 1.0 + (boundary_weight - 1.0) * boundary_flat
    
    # Compute Dice coefficient with boundary weighting
    # Instead of weighting both pred and target, we weight the intersection and union
    intersection = (pred_flat * target_flat * weight_map).sum(dim=2)
    pred_sum = (pred_flat * weight_map).sum(dim=2)
    target_sum = (target_flat * weight_map).sum(dim=2)
    
    # Dice coefficient with proper weighting
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    # Dice loss (1 - dice coefficient), clamped to [0, 1]
    loss = (1.0 - dice).clamp(min=0.0, max=1.0)
    
    # Return mean across channels and batch
    return loss.mean()


class BoundaryAwareDiceLoss(nn.Module):
    """Boundary Aware Dice Loss Module.
    
    This loss is particularly effective for segmentation tasks where boundary
    precision is important. It combines traditional Dice loss with boundary
    awareness by assigning higher weights to pixels near object boundaries.
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero. Default: 1e-5.
        boundary_weight (float): Weight factor for boundary regions. Higher values
                                give more importance to boundaries. Default: 2.0.
        boundary_kernel_size (int): Kernel size for boundary detection. Default: 3.
        reduction (str): Reduction method. Options are "none", "mean" and "sum". Default: "mean".
        loss_weight (float): Weight of the loss. Default: 1.0.
    
    Example:
        >>> import torch
        >>> loss_fn = BoundaryAwareDiceLoss()
        >>> pred = torch.randn(2, 1, 32, 32)  # (B, C, H, W)
        >>> target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        >>> loss = loss_fn(pred, target)
    """
    
    def __init__(self, 
                 smooth=1e-5,
                 boundary_weight=2.0,
                 boundary_kernel_size=3,
                 reduction='mean',
                 loss_weight=1.0):
        super(BoundaryAwareDiceLoss, self).__init__()
        self.smooth = smooth
        self.boundary_weight = boundary_weight
        self.boundary_kernel_size = boundary_kernel_size
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.
        
        Args:
            pred (torch.Tensor): Predicted segmentation with shape (N, C, H, W) or (N, H, W).
            target (torch.Tensor): Target segmentation with shape (N, C, H, W) or (N, H, W).
            weight (torch.Tensor, optional): Element-wise weights. Default: None.
            avg_factor (int, optional): Average factor for loss. Default: None.
            reduction_override (str, optional): Reduction method override. Default: None.
        
        Returns:
            torch.Tensor: Computed loss value.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        
        loss = self.loss_weight * boundary_aware_dice_loss(
            pred,
            target,
            weight,
            smooth=self.smooth,
            boundary_weight=self.boundary_weight,
            boundary_kernel_size=self.boundary_kernel_size,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
        )
        
        return loss
