# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalFusion(nn.Module):
    """
    Temporal fusion module that processes sequences of frames.
    Can use 3D convolutions or temporal attention to aggregate information.
    """
    def __init__(self, in_channels, fusion_type='conv3d', num_frames=8):
        super(TemporalFusion, self).__init__()
        self.fusion_type = fusion_type
        self.num_frames = num_frames
        
        if fusion_type == 'conv3d':
            # Use 3D convolutions for temporal modeling
            self.temporal_conv = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), 
                         padding=(1, 0, 0), bias=False),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
            )
        elif fusion_type == 'attention':
            # Use temporal attention mechanism
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=in_channels,
                num_heads=8,
                batch_first=True
            )
        elif fusion_type == 'lstm':
            # Use LSTM for temporal modeling
            self.lstm = nn.LSTM(
                input_size=in_channels,
                hidden_size=in_channels,
                num_layers=2,
                batch_first=True,
                bidirectional=False
            )
        else:
            # Simple averaging
            pass
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, C, H, W) where T is number of frames
        Returns:
            Tensor of shape (B, C, H, W) after temporal fusion
        """
        B, T, C, H, W = x.shape
        
        if self.fusion_type == 'conv3d':
            # Reshape to (B, C, T, H, W) for 3D conv
            x = x.permute(0, 2, 1, 3, 4)
            x = self.temporal_conv(x)
            # Aggregate along temporal dimension
            x = x.mean(dim=2)  # (B, C, H, W)
            
        elif self.fusion_type == 'attention':
            # Reshape for attention: (B*H*W, T, C)
            x = x.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, T, C)
            x = x.view(B * H * W, T, C)
            x, _ = self.temporal_attention(x, x, x)
            # Take the last time step or average
            x = x.mean(dim=1)  # (B*H*W, C)
            x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
            
        elif self.fusion_type == 'lstm':
            # Reshape for LSTM: (B*H*W, T, C)
            x = x.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, T, C)
            x = x.view(B * H * W, T, C)
            x, _ = self.lstm(x)
            # Take the last time step
            x = x[:, -1, :]  # (B*H*W, C)
            x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
            
        else:
            # Simple averaging along temporal dimension
            x = x.mean(dim=1)  # (B, C, H, W)
        
        return x


class TemporalBackbone(nn.Module):
    """
    Wrapper for existing backbone with temporal fusion capability.
    Processes sequences of frames and fuses them temporally.
    """
    def __init__(self, backbone, fusion_type='conv3d', num_frames=8, fusion_stages=[2, 4, 5]):
        super(TemporalBackbone, self).__init__()
        self.backbone = backbone
        self.num_frames = num_frames
        self.fusion_stages = fusion_stages
        
        # Create temporal fusion modules for each output stage
        # Note: in_channels should match the backbone output channels at each stage
        # For TinyNAS, these are typically [128, 256, 512] for stages [2, 4, 5]
        self.temporal_fusion_modules = nn.ModuleDict()
        
        # We'll create fusion modules dynamically based on the first forward pass
        self.fusion_type = fusion_type
        self.initialized = False
        
    def init_temporal_fusion(self, feature_channels):
        """Initialize temporal fusion modules based on feature channels"""
        for idx, channels in enumerate(feature_channels):
            self.temporal_fusion_modules[str(idx)] = TemporalFusion(
                in_channels=channels,
                fusion_type=self.fusion_type,
                num_frames=self.num_frames
            )
        self.initialized = True
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, C, H, W) or (B, C, H, W)
               If 4D, treats as single frame. If 5D, processes as temporal sequence.
        Returns:
            List of feature maps after temporal fusion
        """
        # Check if input has temporal dimension
        if len(x.shape) == 5:
            B, T, C, H, W = x.shape
            # Reshape to process all frames through backbone
            x = x.view(B * T, C, H, W)
            has_temporal = True
        else:
            has_temporal = False
            B = x.shape[0]
            T = 1
        
        # Forward through backbone
        feature_outs = self.backbone(x)  # List of feature maps
        
        if has_temporal:
            # Initialize temporal fusion modules if not done yet
            if not self.initialized:
                feature_channels = [f.shape[1] for f in feature_outs]
                self.init_temporal_fusion(feature_channels)
            
            # Reshape features back to temporal format and apply temporal fusion
            fused_features = []
            for idx, feat in enumerate(feature_outs):
                _, C, H, W = feat.shape
                # Reshape to (B, T, C, H, W)
                feat = feat.view(B, T, C, H, W)
                # Apply temporal fusion
                feat = self.temporal_fusion_modules[str(idx)](feat)
                fused_features.append(feat)
            
            return fused_features
        else:
            # No temporal dimension, return as is
            return feature_outs
    
    def init_weights(self):
        """Initialize weights"""
        self.backbone.init_weights()
