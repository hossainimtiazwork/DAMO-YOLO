# Specialized losses for needle detection

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CircleLoss(nn.Module):
    """
    Loss for needle radius prediction.
    Combines L1 loss with a scale-aware component.
    """
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CircleLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
    
    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """
        Args:
            pred: Predicted radius values, shape (N, 1)
            target: Ground truth radius values, shape (N, 1)
            weight: Optional weight for each sample
            avg_factor: Average factor for loss computation
            reduction_override: Override the reduction method
        Returns:
            Computed loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        
        # Smooth L1 loss for radius
        loss = F.smooth_l1_loss(pred, target, reduction='none')
        
        # Apply weights if provided
        if weight is not None:
            if weight.dim() == 1:
                weight = weight.unsqueeze(1)
            loss = loss * weight
        
        # Reduce loss
        if reduction == 'mean':
            if avg_factor is not None:
                loss = loss.sum() / avg_factor
            else:
                loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'none':
            pass
        
        return self.loss_weight * loss


class DirectionLoss(nn.Module):
    """
    Loss for needle direction (angle) prediction.
    Uses cosine similarity between predicted and target direction vectors.
    Direction is represented as (sin(angle), cos(angle)) for continuity.
    """
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(DirectionLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
    
    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """
        Args:
            pred: Predicted direction vectors (sin, cos), shape (N, 2)
            target: Ground truth direction vectors (sin, cos), shape (N, 2)
            weight: Optional weight for each sample
            avg_factor: Average factor for loss computation
            reduction_override: Override the reduction method
        Returns:
            Computed loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        
        # Normalize predictions and targets to unit vectors
        pred_norm = F.normalize(pred, dim=1)
        target_norm = F.normalize(target, dim=1)
        
        # Cosine similarity loss (1 - cosine_similarity)
        # cosine_similarity ranges from -1 to 1, so loss ranges from 0 to 2
        cosine_sim = (pred_norm * target_norm).sum(dim=1, keepdim=True)
        loss = 1.0 - cosine_sim
        
        # Apply weights if provided
        if weight is not None:
            if weight.dim() == 1:
                weight = weight.unsqueeze(1)
            loss = loss * weight
        
        # Reduce loss
        if reduction == 'mean':
            if avg_factor is not None:
                loss = loss.sum() / avg_factor
            else:
                loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'none':
            pass
        
        return self.loss_weight * loss


class AngleLoss(nn.Module):
    """
    Alternative direction loss using direct angle difference.
    This can be used as an alternative to DirectionLoss.
    """
    def __init__(self, loss_weight=1.0, reduction='mean', angle_format='radian'):
        super(AngleLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.angle_format = angle_format
    
    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """
        Args:
            pred: Predicted angles, shape (N, 1)
            target: Ground truth angles, shape (N, 1)
            weight: Optional weight for each sample
            avg_factor: Average factor for loss computation
            reduction_override: Override the reduction method
        Returns:
            Computed loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        
        # Compute angle difference considering periodicity
        # Normalize angles to [-pi, pi] for radians or [-180, 180] for degrees
        if self.angle_format == 'radian':
            max_angle = math.pi
        else:
            max_angle = 180.0
        
        # Compute the smallest angle difference
        diff = pred - target
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        
        # L1 loss on angle difference
        loss = torch.abs(diff)
        
        # Apply weights if provided
        if weight is not None:
            if weight.dim() == 1:
                weight = weight.unsqueeze(1)
            loss = loss * weight
        
        # Reduce loss
        if reduction == 'mean':
            if avg_factor is not None:
                loss = loss.sum() / avg_factor
            else:
                loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'none':
            pass
        
        return self.loss_weight * loss


class NeedleLoss(nn.Module):
    """
    Combined loss for needle detection that includes:
    - Classification loss
    - Bounding box loss
    - Radius loss
    - Direction loss
    """
    def __init__(self, 
                 cls_weight=1.0,
                 bbox_weight=2.0,
                 radius_weight=1.0,
                 direction_weight=1.0):
        super(NeedleLoss, self).__init__()
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.radius_weight = radius_weight
        self.direction_weight = direction_weight
        
        self.circle_loss = CircleLoss(loss_weight=radius_weight)
        self.direction_loss = DirectionLoss(loss_weight=direction_weight)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dictionary containing predictions:
                - 'cls': Classification scores
                - 'bbox': Bounding box predictions
                - 'radius': Radius predictions
                - 'direction': Direction predictions
            targets: Dictionary containing targets:
                - 'cls': Classification labels
                - 'bbox': Ground truth bounding boxes
                - 'radius': Ground truth radius
                - 'direction': Ground truth direction
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Circle loss
        if 'radius' in predictions and 'radius' in targets:
            losses['loss_radius'] = self.circle_loss(
                predictions['radius'],
                targets['radius']
            )
        
        # Direction loss
        if 'direction' in predictions and 'direction' in targets:
            losses['loss_direction'] = self.direction_loss(
                predictions['direction'],
                targets['direction']
            )
        
        return losses
