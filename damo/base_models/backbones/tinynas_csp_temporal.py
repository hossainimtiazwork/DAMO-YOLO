# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import torch
import torch.nn as nn

from ..core.ops import Focus, RepConv, SPPBottleneck, get_activation, TemporalFusion
from .tinynas_csp import ConvKXBN, ConvKXBNRELU, ResConvBlock, CSPStem


class TemporalCSPWrapper(nn.Module):
    """CSP Wrapper with temporal fusion capability"""
    def __init__(self, convstem, act='relu', reparam=False, with_spp=False, 
                 temporal_fusion=False, num_frames=4, fusion_type='conv3d'):
        super(TemporalCSPWrapper, self).__init__()
        self.with_spp = with_spp
        self.temporal_fusion = temporal_fusion
        self.num_frames = num_frames
        
        if isinstance(convstem, tuple):
            in_c = convstem[0].in_channels
            out_c = convstem[-1].out_channels
            hidden_dim = convstem[0].out_channels // 2
            _convstem = nn.ModuleList()
            for modulelist in convstem:
                for layer in modulelist.block_list:
                    _convstem.append(layer)
        else:
            in_c = convstem.in_channels
            out_c = convstem.out_channels
            hidden_dim = out_c // 2
            _convstem = convstem.block_list

        self.convstem = nn.ModuleList()
        for layer in _convstem:
            self.convstem.append(layer)

        self.act = get_activation(act)
        self.downsampler = ConvKXBNRELU(in_c,
                                        hidden_dim * 2,
                                        3,
                                        2,
                                        act=self.act)
        if self.with_spp:
            self.spp = SPPBottleneck(hidden_dim * 2, hidden_dim * 2)
            
        # Add temporal fusion module
        if self.temporal_fusion:
            self.temporal_fusion_module = TemporalFusion(
                hidden_dim * 2, 
                num_frames=num_frames,
                fusion_type=fusion_type
            )
            
        if len(self.convstem) > 0:
            self.conv_start = ConvKXBNRELU(hidden_dim * 2,
                                           hidden_dim,
                                           1,
                                           1,
                                           act=self.act)
            self.conv_shortcut = ConvKXBNRELU(hidden_dim * 2,
                                              out_c // 2,
                                              1,
                                              1,
                                              act=self.act)
            self.conv_fuse = ConvKXBNRELU(out_c, out_c, 1, 1, act=self.act)

    def forward(self, x):
        x = self.downsampler(x)
        if self.with_spp:
            x = self.spp(x)
        
        # Apply temporal fusion if enabled
        if self.temporal_fusion:
            x = self.temporal_fusion_module(x)
            
        if len(self.convstem) > 0:
            shortcut = self.conv_shortcut(x)
            x = self.conv_start(x)
            for block in self.convstem:
                x = block(x)
            x = torch.cat((x, shortcut), dim=1)
            x = self.conv_fuse(x)
        return x


class TinyNAS_Temporal(nn.Module):
    """TinyNAS backbone with temporal processing capabilities for video inputs"""
    def __init__(self,
                 structure_info=None,
                 out_indices=[2, 3, 4],
                 with_spp=False,
                 use_focus=False,
                 act='silu',
                 reparam=False,
                 num_frames=4,
                 temporal_stages=[3, 4],  # Which stages to apply temporal fusion
                 fusion_type='conv3d'):
        super(TinyNAS_Temporal, self).__init__()
        self.out_indices = out_indices
        self.num_frames = num_frames
        self.temporal_stages = temporal_stages
        self.block_list = nn.ModuleList()
        self.stride_list = []

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

        # Build CSP stages with temporal fusion
        self.csp_stage = nn.ModuleList()
        self.csp_stage.append(self.block_list[0])
        
        # Stage 1
        self.csp_stage.append(
            TemporalCSPWrapper(
                self.block_list[1],
                temporal_fusion=(1 in temporal_stages),
                num_frames=num_frames,
                fusion_type=fusion_type
            )
        )
        
        # Stage 2
        self.csp_stage.append(
            TemporalCSPWrapper(
                self.block_list[2],
                temporal_fusion=(2 in temporal_stages),
                num_frames=num_frames,
                fusion_type=fusion_type
            )
        )
        
        # Stage 3
        self.csp_stage.append(
            TemporalCSPWrapper(
                (self.block_list[3], self.block_list[4]),
                temporal_fusion=(3 in temporal_stages),
                num_frames=num_frames,
                fusion_type=fusion_type
            )
        )
        
        # Stage 4 (with SPP)
        self.csp_stage.append(
            TemporalCSPWrapper(
                self.block_list[5],
                with_spp=with_spp,
                temporal_fusion=(4 in temporal_stages),
                num_frames=num_frames,
                fusion_type=fusion_type
            )
        )
        del self.block_list

    def init_weights(self, pretrain=None):
        pass

    def forward(self, x):
        """
        Forward pass supporting both image and video inputs
        Args:
            x: Input tensor of shape (B, C, H, W) for images 
               or (B, C, T, H, W) or (B*T, C, H, W) for videos
        Returns:
            stage_feature_list: List of feature maps at different stages
        """
        # Handle video input format
        if x.dim() == 5:
            # Convert (B, C, T, H, W) to (B*T, C, H, W) for processing
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
            x = x.view(b * t, c, h, w)
            
        output = x
        stage_feature_list = []
        for idx, block in enumerate(self.csp_stage):
            output = block(output)
            if idx in self.out_indices:
                stage_feature_list.append(output)
        return stage_feature_list


def load_tinynas_temporal_net(backbone_cfg):
    """Load TinyNAS temporal model from configuration"""
    import ast

    struct_str = ''.join([x.strip() for x in backbone_cfg.net_structure_str])
    struct_info = ast.literal_eval(struct_str)
    for layer in struct_info:
        if 'nbitsA' in layer:
            del layer['nbitsA']
        if 'nbitsW' in layer:
            del layer['nbitsW']

    model = TinyNAS_Temporal(
        structure_info=struct_info,
        out_indices=backbone_cfg.out_indices,
        with_spp=backbone_cfg.with_spp,
        use_focus=backbone_cfg.use_focus,
        act=backbone_cfg.act,
        reparam=backbone_cfg.reparam,
        num_frames=getattr(backbone_cfg, 'num_frames', 4),
        temporal_stages=getattr(backbone_cfg, 'temporal_stages', [3, 4]),
        fusion_type=getattr(backbone_cfg, 'fusion_type', 'conv3d')
    )

    return model
