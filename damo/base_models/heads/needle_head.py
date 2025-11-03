import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from damo.utils import postprocess

from ..core.ops import ConvBNAct
from ..core.ota_assigner import AlignOTAAssigner
from ..core.utils import Scale, multi_apply, reduce_mean
from ..core.weight_init import bias_init_with_prob, normal_init
from ..losses.gfocal_loss import (DistributionFocalLoss, GIoULoss,
                                  QualityFocalLoss)
from ..losses.needle_losses import CircleLoss, DirectionLoss

from loguru import logger


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box."""
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances."""
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution."""
    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        """
        b, hw, _, _ = x.size()
        x = x.reshape(b * hw * 4, self.reg_max + 1)
        y = self.project.type_as(x).unsqueeze(1)
        x = torch.matmul(x, y).reshape(b, hw, 4)
        return x


class NeedleHead(nn.Module):
    """
    Specialized head for needle detection with additional outputs:
    - Bounding box
    - Needle type (2 classes)
    - Needle radius
    - Needle direction (angle)
    """
    def __init__(
            self,
            num_classes=2,  # 2 needle types
            in_channels=[128, 256, 512],
            stacked_convs=0,
            feat_channels=256,
            reg_max=12,
            strides=[8, 16, 32],
            norm='gn',
            act='relu',
            nms_conf_thre=0.05,
            nms_iou_thre=0.7,
            nms=True,
            legacy=False,
            last_kernel_size=3,
            export_with_post=True,
            max_radius=100.0,  # Maximum expected needle radius in pixels
            **kwargs):
        super(NeedleHead, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.stacked_convs = stacked_convs
        self.last_kernel_size = last_kernel_size
        self.export_with_post = export_with_post
        self.act = act
        self.strides = strides
        self.max_radius = max_radius
        
        if stacked_convs == 0:
            feat_channels = in_channels
        if isinstance(feat_channels, list):
            self.feat_channels = feat_channels
        else:
            self.feat_channels = [feat_channels] * len(self.strides)
        
        if legacy:
            self.cls_out_channels = num_classes + 1
        else:
            self.cls_out_channels = num_classes
        
        self.reg_max = reg_max
        self.nms = nms
        self.nms_conf_thre = nms_conf_thre
        self.nms_iou_thre = nms_iou_thre

        self.assigner = AlignOTAAssigner(center_radius=2.5,
                                         cls_weight=1.0,
                                         iou_weight=3.0)

        self.feat_size = [torch.zeros(4) for _ in strides]

        self.integral = Integral(self.reg_max)
        
        # Losses
        self.loss_dfl = DistributionFocalLoss(loss_weight=0.25)
        self.loss_cls = QualityFocalLoss(use_sigmoid=False,
                                         beta=2.0,
                                         loss_weight=1.0)
        self.loss_bbox = GIoULoss(loss_weight=2.0)
        self.loss_radius = CircleLoss(loss_weight=1.0)
        self.loss_direction = DirectionLoss(loss_weight=1.0)

        self._init_layers()

    def _build_not_shared_convs(self, in_channel, feat_channels):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        radius_convs = nn.ModuleList()
        direction_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = feat_channels if i > 0 else in_channel
            kernel_size = 3 if i > 0 else 1
            
            # Classification convs
            cls_convs.append(
                ConvBNAct(chn, feat_channels, kernel_size, stride=1,
                         groups=1, norm='bn', act=self.act))
            # Bbox regression convs
            reg_convs.append(
                ConvBNAct(chn, feat_channels, kernel_size, stride=1,
                         groups=1, norm='bn', act=self.act))
            # Radius regression convs
            radius_convs.append(
                ConvBNAct(chn, feat_channels, kernel_size, stride=1,
                         groups=1, norm='bn', act=self.act))
            # Direction regression convs
            direction_convs.append(
                ConvBNAct(chn, feat_channels, kernel_size, stride=1,
                         groups=1, norm='bn', act=self.act))

        return cls_convs, reg_convs, radius_convs, direction_convs

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.radius_convs = nn.ModuleList()
        self.direction_convs = nn.ModuleList()

        for i in range(len(self.strides)):
            cls_convs, reg_convs, radius_convs, direction_convs = \
                self._build_not_shared_convs(self.in_channels[i], self.feat_channels[i])
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
            self.radius_convs.append(radius_convs)
            self.direction_convs.append(direction_convs)

        # Classification head
        self.gfl_cls = nn.ModuleList([
            nn.Conv2d(self.feat_channels[i], self.cls_out_channels,
                     self.last_kernel_size, padding=self.last_kernel_size//2)
            for i in range(len(self.strides))
        ])

        # Bbox regression head
        self.gfl_reg = nn.ModuleList([
            nn.Conv2d(self.feat_channels[i], 4 * (self.reg_max + 1),
                     self.last_kernel_size, padding=self.last_kernel_size//2)
            for i in range(len(self.strides))
        ])
        
        # Radius regression head (1 value)
        self.gfl_radius = nn.ModuleList([
            nn.Conv2d(self.feat_channels[i], 1,
                     self.last_kernel_size, padding=self.last_kernel_size//2)
            for i in range(len(self.strides))
        ])
        
        # Direction regression head (2 values: sin(angle) and cos(angle) respectively)
        self.gfl_direction = nn.ModuleList([
            nn.Conv2d(self.feat_channels[i], 2,
                     self.last_kernel_size, padding=self.last_kernel_size//2)
            for i in range(len(self.strides))
        ])

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        for cls_conv in self.cls_convs:
            for m in cls_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        
        for reg_conv in self.reg_convs:
            for m in reg_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        
        for radius_conv in self.radius_convs:
            for m in radius_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        
        for direction_conv in self.direction_convs:
            for m in direction_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        
        bias_cls = bias_init_with_prob(0.01)
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.gfl_reg[i], std=0.01)
            normal_init(self.gfl_radius[i], std=0.01)
            normal_init(self.gfl_direction[i], std=0.01)

    def forward(self, xin, labels=None, imgs=None, aux_targets=None):
        if self.training:
            return self.forward_train(xin=xin, labels=labels, imgs=imgs, aux_targets=aux_targets)
        else:
            return self.forward_eval(xin=xin, labels=labels, imgs=imgs)

    def forward_train(self, xin, labels=None, imgs=None, aux_targets=None):
        b, c, h, w = xin[0].shape
        
        # Prepare labels during training
        if labels is not None:
            gt_bbox_list = []
            gt_cls_list = []
            gt_radius_list = []
            gt_direction_list = []
            
            for label in labels:
                gt_bbox_list.append(label.bbox)
                gt_cls_list.append((label.get_field('labels')).long())
                
                # Get radius and direction if available
                if label.has_field('radius'):
                    gt_radius_list.append(label.get_field('radius'))
                else:
                    # Default radius if not provided
                    gt_radius_list.append(torch.ones(label.bbox.shape[0], device=label.bbox.device) * 10.0)
                
                if label.has_field('direction'):
                    gt_direction_list.append(label.get_field('direction'))
                else:
                    # Default direction (0 radians) if not provided
                    gt_direction_list.append(torch.zeros(label.bbox.shape[0], device=label.bbox.device))

        # Prepare priors for label assignment and bbox decode
        mlvl_priors_list = [
            self.get_single_level_center_priors(xin[i].shape[0],
                                                xin[i].shape[-2:],
                                                stride,
                                                dtype=torch.float32,
                                                device=xin[0].device)
            for i, stride in enumerate(self.strides)
        ]
        mlvl_priors = torch.cat(mlvl_priors_list, dim=1)

        # Forward for predictions
        cls_scores, bbox_preds, bbox_before_softmax, radius_preds, direction_preds = multi_apply(
            self.forward_single,
            xin,
            self.cls_convs,
            self.reg_convs,
            self.radius_convs,
            self.direction_convs,
            self.gfl_cls,
            self.gfl_reg,
            self.gfl_radius,
            self.gfl_direction,
            self.scales,
        )
        
        cls_scores = torch.cat(cls_scores, dim=1)
        bbox_preds = torch.cat(bbox_preds, dim=1)
        bbox_before_softmax = torch.cat(bbox_before_softmax, dim=1)
        radius_preds = torch.cat(radius_preds, dim=1)
        direction_preds = torch.cat(direction_preds, dim=1)

        # Calculate losses
        loss = self.loss(
            cls_scores,
            bbox_preds,
            bbox_before_softmax,
            radius_preds,
            direction_preds,
            gt_bbox_list,
            gt_cls_list,
            gt_radius_list,
            gt_direction_list,
            mlvl_priors,
        )
        return loss

    def forward_eval(self, xin, labels=None, imgs=None):
        # Prepare priors
        if self.feat_size[0] != xin[0].shape:
            mlvl_priors_list = [
                self.get_single_level_center_priors(xin[i].shape[0],
                                                    xin[i].shape[-2:],
                                                    stride,
                                                    dtype=torch.float32,
                                                    device=xin[0].device)
                for i, stride in enumerate(self.strides)
            ]
            self.mlvl_priors = torch.cat(mlvl_priors_list, dim=1)
            self.feat_size[0] = xin[0].shape

        # Forward for predictions
        cls_scores, bbox_preds, radius_preds, direction_preds = multi_apply(
            self.forward_single,
            xin,
            self.cls_convs,
            self.reg_convs,
            self.radius_convs,
            self.direction_convs,
            self.gfl_cls,
            self.gfl_reg,
            self.gfl_radius,
            self.gfl_direction,
            self.scales,
        )

        if self.export_with_post:
            cls_scores_new, bbox_preds_new, radius_preds_new, direction_preds_new = [], [], [], []
            
            for cls_score, bbox_pred, radius_pred, direction_pred in \
                    zip(cls_scores, bbox_preds, radius_preds, direction_preds):
                N, C, H, W = bbox_pred.size()
                
                # Process bbox predictions
                bbox_pred = F.softmax(bbox_pred.reshape(N, 4, self.reg_max + 1, H, W), dim=2)
                bbox_pred = bbox_pred.reshape(N, 4, self.reg_max + 1, H, W)
                
                # Flatten predictions
                cls_score = cls_score.flatten(start_dim=2).permute(0, 2, 1)
                bbox_pred = bbox_pred.flatten(start_dim=3).permute(0, 3, 1, 2)
                radius_pred = radius_pred.flatten(start_dim=2).permute(0, 2, 1)
                direction_pred = direction_pred.flatten(start_dim=2).permute(0, 2, 1)
                
                cls_scores_new.append(cls_score)
                bbox_preds_new.append(bbox_pred)
                radius_preds_new.append(radius_pred)
                direction_preds_new.append(direction_pred)

            cls_scores = torch.cat(cls_scores_new, dim=1)[:, :, :self.num_classes]
            bbox_preds = torch.cat(bbox_preds_new, dim=1)
            radius_preds = torch.cat(radius_preds_new, dim=1)
            direction_preds = torch.cat(direction_preds_new, dim=1)
            
            # Decode bbox
            bbox_preds = self.integral(bbox_preds) * self.mlvl_priors[..., 2, None]
            bbox_preds = distance2bbox(self.mlvl_priors[..., :2], bbox_preds)
            
            # Decode radius (apply sigmoid and scale)
            radius_preds = torch.sigmoid(radius_preds) * self.max_radius
            
            # Direction is in (sin, cos) format at indices [0, 1], convert to angle
            # angle = atan2(sin, cos)
            angles = torch.atan2(direction_preds[..., 0], direction_preds[..., 1]).unsqueeze(-1)

            if self.nms:
                # Extended postprocess to include radius and direction
                output = self.postprocess_needle(
                    cls_scores, bbox_preds, radius_preds, angles,
                    self.num_classes, self.nms_conf_thre, self.nms_iou_thre, imgs
                )
                return output
        
        return cls_scores, bbox_preds, radius_preds, direction_preds

    def forward_single(self, x, cls_convs, reg_convs, radius_convs, direction_convs,
                      gfl_cls, gfl_reg, gfl_radius, gfl_direction, scale):
        """Forward feature of a single scale level."""
        cls_feat = x
        reg_feat = x
        radius_feat = x
        direction_feat = x

        # Apply convolutions
        for cls_conv, reg_conv, radius_conv, direction_conv in \
                zip(cls_convs, reg_convs, radius_convs, direction_convs):
            cls_feat = cls_conv(cls_feat)
            reg_feat = reg_conv(reg_feat)
            radius_feat = radius_conv(radius_feat)
            direction_feat = direction_conv(direction_feat)

        # Generate predictions
        bbox_pred = scale(gfl_reg(reg_feat)).float()
        cls_score = gfl_cls(cls_feat).sigmoid()
        radius_pred = gfl_radius(radius_feat)
        direction_pred = gfl_direction(direction_feat)
        
        # Normalize direction to unit circle (sin, cos)
        direction_pred = F.normalize(direction_pred, dim=1)

        N, C, H, W = bbox_pred.size()
        
        if self.training:
            bbox_before_softmax = bbox_pred.reshape(N, 4, self.reg_max + 1, H, W)
            bbox_before_softmax = bbox_before_softmax.flatten(start_dim=3).permute(0, 3, 1, 2)

            bbox_pred = F.softmax(bbox_pred.reshape(N, 4, self.reg_max + 1, H, W), dim=2)
            bbox_pred = bbox_pred.reshape(N, 4, self.reg_max + 1, H, W)

            cls_score = cls_score.flatten(start_dim=2).permute(0, 2, 1)
            bbox_pred = bbox_pred.flatten(start_dim=3).permute(0, 3, 1, 2)
            radius_pred = radius_pred.flatten(start_dim=2).permute(0, 2, 1)
            direction_pred = direction_pred.flatten(start_dim=2).permute(0, 2, 1)

            return cls_score, bbox_pred, bbox_before_softmax, radius_pred, direction_pred
        else:
            return cls_score, bbox_pred, radius_pred, direction_pred

    def get_single_level_center_priors(self, batch_size, featmap_size, stride,
                                       dtype, device):
        h, w = featmap_size
        x_range = (torch.arange(0, int(w), dtype=dtype, device=device)) * stride
        y_range = (torch.arange(0, int(h), dtype=dtype, device=device)) * stride

        x = x_range.repeat(h, 1)
        y = y_range.unsqueeze(-1).repeat(1, w)

        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0], ), stride)
        priors = torch.stack([x, y, strides, strides], dim=-1)

        return priors.unsqueeze(0).repeat(batch_size, 1, 1)

    def loss(self, cls_scores, bbox_preds, bbox_before_softmax, radius_preds, 
             direction_preds, gt_bboxes, gt_labels, gt_radius, gt_direction, 
             mlvl_center_priors, gt_bboxes_ignore=None):
        """Compute losses of the head."""
        device = cls_scores[0].device

        # Get decoded bboxes for label assignment
        dis_preds = self.integral(bbox_preds) * mlvl_center_priors[..., 2, None]
        decoded_bboxes = distance2bbox(mlvl_center_priors[..., :2], dis_preds)
        
        cls_reg_targets = self.get_targets(
            cls_scores, decoded_bboxes, gt_bboxes, mlvl_center_priors,
            gt_labels_list=gt_labels, gt_radius_list=gt_radius,
            gt_direction_list=gt_direction
        )

        if cls_reg_targets is None:
            return None

        (labels_list, label_scores_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, dfl_targets_list, radius_targets_list, 
         direction_targets_list, num_pos) = cls_reg_targets

        num_total_pos = max(
            reduce_mean(torch.tensor(num_pos).type(torch.float).to(device)).item(), 1.0)

        labels = torch.cat(labels_list, dim=0)
        label_scores = torch.cat(label_scores_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        dfl_targets = torch.cat(dfl_targets_list, dim=0)
        radius_targets = torch.cat(radius_targets_list, dim=0)
        direction_targets = torch.cat(direction_targets_list, dim=0)

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        bbox_before_softmax = bbox_before_softmax.reshape(-1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)
        radius_preds = radius_preds.reshape(-1, 1)
        direction_preds = direction_preds.reshape(-1, 2)

        # Classification loss
        loss_qfl = self.loss_cls(cls_scores, (labels, label_scores),
                                 avg_factor=num_total_pos)

        pos_inds = torch.nonzero((labels >= 0) & (labels < self.num_classes),
                                 as_tuple=False).squeeze(1)

        weight_targets = cls_scores.detach()
        weight_targets = weight_targets.max(dim=1)[0][pos_inds]
        norm_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)

        if len(pos_inds) > 0:
            # Bbox loss
            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=1.0 * norm_factor,
            )
            
            # DFL loss
            loss_dfl = self.loss_dfl(
                bbox_before_softmax[pos_inds].reshape(-1, self.reg_max + 1),
                dfl_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * norm_factor,
            )
            
            # Radius loss
            loss_radius = self.loss_radius(
                radius_preds[pos_inds],
                radius_targets[pos_inds],
                weight=weight_targets,
                avg_factor=norm_factor,
            )
            
            # Direction loss
            loss_direction = self.loss_direction(
                direction_preds[pos_inds],
                direction_targets[pos_inds],
                weight=weight_targets,
                avg_factor=norm_factor,
            )
        else:
            loss_bbox = bbox_preds.sum() / norm_factor * 0.0
            loss_dfl = bbox_preds.sum() / norm_factor * 0.0
            loss_radius = radius_preds.sum() / norm_factor * 0.0
            loss_direction = direction_preds.sum() / norm_factor * 0.0
            logger.info(f'No Positive Samples on {bbox_preds.device}!')

        total_loss = loss_qfl + loss_bbox + loss_dfl + loss_radius + loss_direction

        return dict(
            total_loss=total_loss,
            loss_cls=loss_qfl,
            loss_bbox=loss_bbox,
            loss_dfl=loss_dfl,
            loss_radius=loss_radius,
            loss_direction=loss_direction,
        )

    def get_targets(self, cls_scores, bbox_preds, gt_bboxes_list, mlvl_center_priors,
                    gt_labels_list=None, gt_radius_list=None, gt_direction_list=None,
                    unmap_outputs=True):
        """Get targets for needle head."""
        num_imgs = mlvl_center_priors.shape[0]

        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        if gt_radius_list is None:
            gt_radius_list = [None for _ in range(num_imgs)]
        if gt_direction_list is None:
            gt_direction_list = [None for _ in range(num_imgs)]

        (all_labels, all_label_scores, all_label_weights, all_bbox_targets,
         all_bbox_weights, all_dfl_targets, all_radius_targets,
         all_direction_targets, all_pos_num) = multi_apply(
             self.get_target_single,
             mlvl_center_priors,
             cls_scores,
             bbox_preds,
             gt_bboxes_list,
             gt_labels_list,
             gt_radius_list,
             gt_direction_list,
         )
        
        if any([labels is None for labels in all_labels]):
            return None
        
        all_pos_num = sum(all_pos_num)

        return (all_labels, all_label_scores, all_label_weights,
                all_bbox_targets, all_bbox_weights, all_dfl_targets,
                all_radius_targets, all_direction_targets, all_pos_num)

    def get_target_single(self, center_priors, cls_scores, bbox_preds,
                         gt_bboxes, gt_labels, gt_radius, gt_direction,
                         unmap_outputs=True, gt_bboxes_ignore=None):
        """Compute targets for anchors in a single image."""
        num_valid_center = center_priors.shape[0]

        labels = center_priors.new_full((num_valid_center, ),
                                        self.num_classes, dtype=torch.long)
        label_weights = center_priors.new_zeros(num_valid_center, dtype=torch.float)
        label_scores = center_priors.new_zeros(num_valid_center, dtype=torch.float)

        bbox_targets = torch.zeros_like(center_priors)
        bbox_weights = torch.zeros_like(center_priors)
        dfl_targets = torch.zeros_like(center_priors)
        radius_targets = torch.zeros(num_valid_center, 1, device=center_priors.device)
        direction_targets = torch.zeros(num_valid_center, 2, device=center_priors.device)

        if gt_labels.size(0) == 0:
            return (labels, label_scores, label_weights, bbox_targets,
                    bbox_weights, dfl_targets, radius_targets, direction_targets, 0)

        assign_result = self.assigner.assign(cls_scores.detach(),
                                             center_priors,
                                             bbox_preds.detach(), gt_bboxes,
                                             gt_labels)

        pos_inds, neg_inds, pos_bbox_targets, pos_assign_gt_inds = self.sample(
            assign_result, gt_bboxes)
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            labels[pos_inds] = gt_labels[pos_assign_gt_inds]
            label_scores[pos_inds] = pos_ious
            label_weights[pos_inds] = 1.0

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            dfl_targets[pos_inds, :] = (bbox2distance(
                center_priors[pos_inds, :2] / center_priors[pos_inds, None, 2],
                pos_bbox_targets / center_priors[pos_inds, None, 2],
                self.reg_max))
            
            # Assign radius targets
            if gt_radius is not None:
                radius_targets[pos_inds, 0] = gt_radius[pos_assign_gt_inds]
            
            # Assign direction targets (convert angle to sin/cos)
            if gt_direction is not None:
                angles = gt_direction[pos_assign_gt_inds]
                direction_targets[pos_inds, 0] = torch.sin(angles)
                direction_targets[pos_inds, 1] = torch.cos(angles)
        
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return (labels, label_scores, label_weights, bbox_targets,
                bbox_weights, dfl_targets, radius_targets, direction_targets,
                pos_inds.size(0))

    def sample(self, assign_result, gt_bboxes):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0,
                                 as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0,
                                 as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]

        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    def postprocess_needle(self, cls_scores, bbox_preds, radius_preds, direction_preds,
                          num_classes, conf_thre, nms_thre, imgs):
        """Custom postprocessing for needle detection with additional outputs."""
        # This is a simplified version - you may need to extend the postprocess function
        output = postprocess(cls_scores, bbox_preds, num_classes, conf_thre, nms_thre, imgs)
        
        # Add radius and direction to output
        # Note: This requires modification to the postprocess function or custom handling
        # For now, we'll return a dictionary with all outputs
        if output is not None:
            for i in range(len(output)):
                if output[i] is not None:
                    # Extract corresponding radius and direction for detected boxes
                    # This is a placeholder - actual implementation depends on NMS output
                    pass
        
        return output
