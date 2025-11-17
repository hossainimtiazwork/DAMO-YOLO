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
        avg_factor (float): Average factor when computing the mean of losses.
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


def compute_boundary_mask(target, kernel_size=3, ignore_class=None):
    """Compute boundary mask from target labels.
    
    Detects boundaries by finding pixels where neighboring pixels have
    different class labels.
    
    Args:
        target (torch.Tensor): Target labels with shape (N, H, W) or (N, 1, H, W).
        kernel_size (int): Size of the kernel for boundary detection. Default: 3.
        ignore_class (int, optional): Class index to ignore (e.g., background).
            Pixels of this class will have boundary weight of 0.
    
    Returns:
        torch.Tensor: Boundary mask with shape (N, H, W) with values in [0, 1].
            Higher values indicate boundary regions.
    """
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    
    # Ensure target is long type for proper comparison
    target = target.long()
    
    N, H, W = target.shape
    device = target.device
    
    # Create boundary mask
    boundary_mask = torch.zeros((N, H, W), dtype=torch.float32, device=device)
    
    # Compute boundaries by checking neighboring pixels
    # Horizontal boundaries
    if W > 1:
        h_boundaries = (target[:, :, :-1] != target[:, :, 1:]).float()
        boundary_mask[:, :, :-1] += h_boundaries
        boundary_mask[:, :, 1:] += h_boundaries
    
    # Vertical boundaries
    if H > 1:
        v_boundaries = (target[:, :-1, :] != target[:, 1:, :]).float()
        boundary_mask[:, :-1, :] += v_boundaries
        boundary_mask[:, 1:, :] += v_boundaries
    
    # Diagonal boundaries (optional, for more comprehensive boundary detection)
    if H > 1 and W > 1:
        # Top-left to bottom-right
        d1_boundaries = (target[:, :-1, :-1] != target[:, 1:, 1:]).float()
        boundary_mask[:, :-1, :-1] += d1_boundaries
        boundary_mask[:, 1:, 1:] += d1_boundaries
        
        # Top-right to bottom-left
        d2_boundaries = (target[:, :-1, 1:] != target[:, 1:, :-1]).float()
        boundary_mask[:, :-1, 1:] += d2_boundaries
        boundary_mask[:, 1:, :-1] += d2_boundaries
    
    # Normalize boundary mask to [0, 1]
    max_val = boundary_mask.max()
    if max_val > 0:
        boundary_mask = boundary_mask / max_val
    
    # Ignore specified class
    if ignore_class is not None:
        ignore_mask = (target == ignore_class)
        boundary_mask[ignore_mask] = 0.0
    
    return boundary_mask


@weighted_loss
def boundary_aware_loss(pred, target, boundary_weight=2.0, ignore_class=None, 
                       use_sigmoid=True, kernel_size=3):
    """Boundary-aware loss for multiclass classification.
    
    This loss emphasizes the boundaries between different classes by applying
    higher weights to pixels near class boundaries. It can also ignore a
    specified class (typically background).
    
    Args:
        pred (torch.Tensor): Predicted logits with shape (N, C, H, W), where C is
            the number of classes.
        target (torch.Tensor): Ground truth labels with shape (N, H, W) or (N, 1, H, W).
        boundary_weight (float): Weight multiplier for boundary regions. Default: 2.0.
        ignore_class (int, optional): Class index to ignore (e.g., background class).
        use_sigmoid (bool): Whether to use sigmoid or softmax. Default: True.
        kernel_size (int): Kernel size for boundary detection. Default: 3.
    
    Returns:
        torch.Tensor: Loss tensor with shape (N, H, W).
    """
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    
    N, C, H, W = pred.shape
    
    # Compute base loss (cross-entropy)
    if use_sigmoid:
        # For binary or multi-label classification
        # Convert to one-hot for proper loss computation
        target_one_hot = F.one_hot(target.long(), num_classes=C).permute(0, 3, 1, 2).float()
        base_loss = F.binary_cross_entropy_with_logits(pred, target_one_hot, reduction='none')
        base_loss = base_loss.sum(dim=1)  # Sum over classes
    else:
        # For multi-class classification
        base_loss = F.cross_entropy(pred, target.long(), reduction='none')
    
    # Compute boundary mask
    boundary_mask = compute_boundary_mask(target, kernel_size=kernel_size, 
                                         ignore_class=ignore_class)
    
    # Apply boundary weighting
    # boundary_mask is in [0, 1], so final weight is in [1, boundary_weight]
    weight_map = 1.0 + (boundary_weight - 1.0) * boundary_mask
    
    # Apply weights to loss
    weighted_loss = base_loss * weight_map
    
    return weighted_loss


class BoundaryAwareLoss(nn.Module):
    """Boundary-aware loss for multiclass classification.
    
    This loss emphasizes the boundaries between different classes by applying
    higher weights to pixels near class boundaries. It is particularly useful
    for segmentation-like tasks where precise boundaries are important.
    
    Args:
        num_classes (int): Number of classes.
        boundary_weight (float): Weight multiplier for boundary regions. 
            Higher values emphasize boundaries more. Default: 2.0.
        ignore_class (int, optional): Class index to ignore (e.g., background class).
            If specified, pixels of this class will not contribute to the loss.
        use_sigmoid (bool): Whether to use sigmoid (for binary/multi-label) or 
            softmax (for multi-class). Default: True.
        kernel_size (int): Kernel size for boundary detection. Default: 3.
        reduction (str): Reduction method. Options are 'none', 'mean' and 'sum'.
            Default: 'mean'.
        loss_weight (float): Overall loss weight. Default: 1.0.
    
    Example:
        >>> loss_fn = BoundaryAwareLoss(num_classes=21, ignore_class=0, boundary_weight=2.5)
        >>> pred = torch.randn(2, 21, 64, 64)  # Batch of 2, 21 classes
        >>> target = torch.randint(0, 21, (2, 64, 64))  # Ground truth labels
        >>> loss = loss_fn(pred, target)
    """
    def __init__(self,
                 num_classes,
                 boundary_weight=2.0,
                 ignore_class=None,
                 use_sigmoid=True,
                 kernel_size=3,
                 reduction='mean',
                 loss_weight=1.0):
        super(BoundaryAwareLoss, self).__init__()
        self.num_classes = num_classes
        self.boundary_weight = boundary_weight
        self.ignore_class = ignore_class
        self.use_sigmoid = use_sigmoid
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        
        Args:
            pred (torch.Tensor): Predicted logits with shape (N, C, H, W).
            target (torch.Tensor): Ground truth labels with shape (N, H, W) or (N, 1, H, W).
            weight (torch.Tensor, optional): Element-wise weights.
            avg_factor (int, optional): Average factor for loss reduction.
            reduction_override (str, optional): Override reduction method.
        
        Returns:
            torch.Tensor: Computed loss value.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        
        loss = self.loss_weight * boundary_aware_loss(
            pred,
            target,
            weight,
            boundary_weight=self.boundary_weight,
            ignore_class=self.ignore_class,
            use_sigmoid=self.use_sigmoid,
            kernel_size=self.kernel_size,
            reduction=reduction,
            avg_factor=avg_factor)
        
        return loss
