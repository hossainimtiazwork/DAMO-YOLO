# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
# Spatio-Temporal Extension for DAMO-YOLO with Multihead Attention

import torch
import torch.nn as nn

from ..core.ops import Focus, RepConv, SPPBottleneck, get_activation
from .tinynas_csp import ConvKXBN, ConvKXBNRELU, ResConvBlock, CSPStem, CSPWrapper


class TemporalMultiheadAttention(nn.Module):
    """
    Multihead Attention module for temporal feature fusion.
    Processes multiple frames and fuses them using self-attention mechanism.
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(TemporalMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # MultiheadAttention for temporal fusion
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False  # (T, B, C) format
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: Temporal features of shape (B, T, C, H, W)
               B: batch size, T: temporal length, C: channels, H: height, W: width
        Returns:
            Fused features of shape (B, C, H, W)
        """
        B, T, C, H, W = x.shape
        
        # Reshape to (T, B*H*W, C) for attention
        x_reshaped = x.permute(1, 0, 3, 4, 2).contiguous()  # (T, B, H, W, C)
        x_reshaped = x_reshaped.view(T, B * H * W, C)  # (T, B*H*W, C)
        
        # Apply multihead attention
        attn_output, _ = self.multihead_attn(x_reshaped, x_reshaped, x_reshaped)
        
        # Apply layer normalization
        attn_output = self.norm(attn_output)
        
        # Reshape back to (B, T, C, H, W)
        attn_output = attn_output.view(T, B, H, W, C)
        attn_output = attn_output.permute(1, 0, 4, 2, 3)  # (B, T, C, H, W)
        
        # Aggregate temporal dimension (mean pooling)
        output = torch.mean(attn_output, dim=1)  # (B, C, H, W)
        
        return output


class TemporalFusionModule(nn.Module):
    """
    Module for fusing temporal features at different scales using attention.
    """
    def __init__(self, channels, num_heads=8, dropout=0.1):
        super(TemporalFusionModule, self).__init__()
        self.temporal_attention = TemporalMultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout
        )
        
    def forward(self, x):
        """
        Args:
            x: Input features of shape (B, T, C, H, W) or (B, C, H, W)
        Returns:
            Fused features of shape (B, C, H, W)
        """
        # If input is not temporal (single frame), return as is
        if x.dim() == 4:
            return x
            
        # Apply temporal attention fusion
        return self.temporal_attention(x)


class TinyNASTemporal(nn.Module):
    """
    Temporal extension of TinyNAS backbone with multihead attention.
    Supports processing multiple frames and fusing them using attention mechanism.
    """
    def __init__(self,
                 structure_info=None,
                 out_indices=[2, 3, 4],
                 with_spp=False,
                 use_focus=False,
                 act='silu',
                 reparam=False,
                 num_frames=1,
                 temporal_fusion_stages=[2, 3, 4],
                 num_heads=8,
                 dropout=0.1):
        super(TinyNASTemporal, self).__init__()
        self.out_indices = out_indices
        self.num_frames = num_frames
        self.temporal_fusion_stages = temporal_fusion_stages
        self.block_list = nn.ModuleList()
        self.stride_list = []

        # Build backbone blocks (same as TinyNAS)
        for idx, block_info in enumerate(structure_info):
            the_block_class = block_info['class']
            if the_block_class == 'ConvKXBNRELU':
                if use_focus and idx == 0:
                    the_block = Focus(block_info['in'],
                                      block_info['out'],
                                      block_info['k'],
                                      act=act)
                else:
                    the_block = ConvKXBNRELU(block_info['in'],
                                             block_info['out'],
                                             block_info['k'],
                                             block_info['s'],
                                             act=act)
            elif the_block_class == 'SuperResConvK1KX':
                the_block = CSPStem(block_info['in'],
                                    block_info['out'],
                                    block_info['btn'],
                                    block_info['s'],
                                    block_info['k'],
                                    block_info['L'],
                                    act=act,
                                    reparam=reparam,
                                    block_type='k1kx')
            elif the_block_class == 'SuperResConvKXKX':
                the_block = CSPStem(block_info['in'],
                                    block_info['out'],
                                    block_info['btn'],
                                    block_info['s'],
                                    block_info['k'],
                                    block_info['L'],
                                    act=act,
                                    reparam=reparam,
                                    block_type='kxkx')
            else:
                raise NotImplementedError

            self.block_list.append(the_block)

        # Create CSP stages
        self.csp_stage = nn.ModuleList()
        self.csp_stage.append(self.block_list[0])
        self.csp_stage.append(CSPWrapper(self.block_list[1]))
        self.csp_stage.append(CSPWrapper(self.block_list[2]))
        self.csp_stage.append(
            CSPWrapper((self.block_list[3], self.block_list[4])))
        self.csp_stage.append(CSPWrapper(self.block_list[5],
                                         with_spp=with_spp))
        del self.block_list
        
        # Add temporal fusion modules for specified stages
        self.temporal_fusion = nn.ModuleDict()
        if num_frames > 1:
            # Get channel dimensions from structure_info
            stage_channels = self._get_stage_channels(structure_info)
            for stage_idx in temporal_fusion_stages:
                if stage_idx in out_indices:
                    channels = stage_channels[stage_idx]
                    self.temporal_fusion[str(stage_idx)] = TemporalFusionModule(
                        channels=channels,
                        num_heads=num_heads,
                        dropout=dropout
                    )

    def _get_stage_channels(self, structure_info):
        """Extract channel dimensions for each stage."""
        # Stage 0: First conv
        # Stage 1: CSPWrapper(block_list[1])
        # Stage 2: CSPWrapper(block_list[2])
        # Stage 3: CSPWrapper(block_list[3], block_list[4])
        # Stage 4: CSPWrapper(block_list[5])
        stage_channels = {
            0: structure_info[0]['out'],
            1: structure_info[1]['out'],
            2: structure_info[2]['out'],
            3: structure_info[4]['out'],  # Last block in tuple
            4: structure_info[5]['out'],
        }
        return stage_channels

    def init_weights(self, pretrain=None):
        pass

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, C, H, W) for temporal or (B, C, H, W) for single frame
               B: batch size, T: temporal frames, C: channels, H: height, W: width
        Returns:
            List of stage features with temporal fusion applied
        """
        # Handle both temporal and single-frame inputs
        is_temporal = (x.dim() == 5)
        
        if is_temporal:
            B, T, C, H, W = x.shape
            # Process each frame through backbone
            # Reshape to (B*T, C, H, W) for batch processing
            x = x.view(B * T, C, H, W)
        
        output = x
        stage_feature_list = []
        
        for idx, block in enumerate(self.csp_stage):
            output = block(output)
            
            if idx in self.out_indices:
                # Apply temporal fusion if this is a temporal input and fusion is enabled for this stage
                if is_temporal and str(idx) in self.temporal_fusion:
                    # Reshape back to temporal format (B, T, C, H, W)
                    _, C_out, H_out, W_out = output.shape
                    output_temporal = output.view(B, T, C_out, H_out, W_out)
                    # Apply temporal fusion
                    fused_output = self.temporal_fusion[str(idx)](output_temporal)
                    stage_feature_list.append(fused_output)
                    # Continue processing with fused features
                    output = fused_output.unsqueeze(1).expand(-1, T, -1, -1, -1).contiguous().view(B * T, C_out, H_out, W_out)
                else:
                    if is_temporal:
                        # Reshape and average for stages without fusion
                        _, C_out, H_out, W_out = output.shape
                        output_temporal = output.view(B, T, C_out, H_out, W_out)
                        fused_output = torch.mean(output_temporal, dim=1)
                        stage_feature_list.append(fused_output)
                        output = fused_output.unsqueeze(1).expand(-1, T, -1, -1, -1).contiguous().view(B * T, C_out, H_out, W_out)
                    else:
                        stage_feature_list.append(output)
        
        return stage_feature_list


def load_tinynas_net_temporal(backbone_cfg):
    """Load temporal TinyNAS model."""
    import ast

    struct_str = ''.join([x.strip() for x in backbone_cfg.net_structure_str])
    struct_info = ast.literal_eval(struct_str)
    for layer in struct_info:
        if 'nbitsA' in layer:
            del layer['nbitsA']
        if 'nbitsW' in layer:
            del layer['nbitsW']

    model = TinyNASTemporal(
        structure_info=struct_info,
        out_indices=backbone_cfg.out_indices,
        with_spp=backbone_cfg.with_spp,
        use_focus=backbone_cfg.use_focus,
        act=backbone_cfg.act,
        reparam=backbone_cfg.reparam,
        num_frames=backbone_cfg.get('num_frames', 1),
        temporal_fusion_stages=backbone_cfg.get('temporal_fusion_stages', [2, 3, 4]),
        num_heads=backbone_cfg.get('num_heads', 8),
        dropout=backbone_cfg.get('dropout', 0.1)
    )

    return model
