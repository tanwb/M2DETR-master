import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation
from pytorch_wavelets import *
import pywt
from src.core import register

__all__ = ['MultiBandMultiScaleDenoisingEncoder']


class CrossScaleConvolutionalAttentionDenoising(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[3, 5, 7]):
        super(CrossScaleConvolutionalAttentionDenoising, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.conv_scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                ## The model reaches AP 0.4111 on IP102 dataset even without nn.BatchNorm2d(in_channels)
                ## See README.md for logs, weights, and code.
                nn.SiLU(inplace=True)
            ) for kernel_size in kernel_sizes
        ])
        self.conv_attn = nn.Sequential(
            nn.Conv2d(in_channels * len(kernel_sizes), in_channels, 1),
            nn.BatchNorm2d(in_channels),
            ## The model reaches AP 0.4111 on IP102 dataset even without nn.BatchNorm2d(in_channels)
            ## See README.md for logs, weights, and code.
            nn.SiLU()
        )

        self.conv_squeeze = nn.Conv2d(2, 1, 3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn_scales = [conv(x) for conv in self.conv_scales]
        attn_fused = torch.cat(attn_scales, dim=1)
        attn_weights = self.sigmoid(self.conv_attn(attn_fused))
        attn_weighted = [attn_weights * attn_s for attn_s in attn_scales]
        attn_weighted = sum(attn_weighted)
        avg_attn = torch.mean(attn_fused, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn_fused, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        attn_weights2 = self.sigmoid(self.conv_squeeze(agg))

        attn_applied = attn_weighted + attn_weights2.expand_as(x) * x

        return attn_applied



class MultiBandMultiScaleDownsamplingDenoising(nn.Module):
    def __init__(self,in_channels,size_flag):
        super(MultiBandMultiScaleDownsamplingDenoising, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        if size_flag == 0:
            kernel_sizes = [3,5, 7]
        else:
            kernel_sizes = [1,3, 5]
        self.y_ccad = nn.ModuleList()
        for i in range(3):
            self.y_ccad.append(
                CrossScaleConvolutionalAttentionDenoising(in_channels, kernel_sizes)
            )

    def forward(self, x):
        LL, yH = self.wt(x)
        HL = yH[0][:, :, 0, ::]
        LH = yH[0][:, :, 1, ::]
        HH = yH[0][:, :, 2, ::]
        y_hl = self.y_ccad[0](HL)
        y_lh = self.y_ccad[1](LH)
        y_hh = self.y_ccad[2](HH)

        x0 = torch.cat([LL, y_hl, y_lh, y_hh], dim=1)

        return x0


class MultiBandMultiScaleDD(nn.Module):
    def __init__(self, in_channels, out_channels, size_flag):
        super(MultiBandMultiScaleDD, self).__init__()
        self.m2_downsampling_denoising = MultiBandMultiScaleDownsamplingDenoising(in_channels, size_flag)
        self.conv_scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(4 * in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.SiLU(inplace=True)
            )  for _ in range(2)
        ])
        self.rep_vgg_blocks = nn.Sequential(*[
            RepVggBlock(out_channels, out_channels, act='relu') for _ in range(3)
        ])

    def forward(self, x):
        m2_downsampling_denoising = self.m2_downsampling_denoising(x)
        m2_fused=self.conv_scales[0](m2_downsampling_denoising)
        m2dd_out = self.rep_vgg_blocks(m2_fused) + self.conv_scales[1](m2_downsampling_denoising)

        return m2dd_out


class MultiScaleEnhancementDenoising(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 bias=None,
                 act="silu",
                 size_flag=0):
        super(MultiScaleEnhancementDenoising, self).__init__()
        self.rep_vgg_blocks= nn.Sequential(*[
            RepVggBlock(out_channels, out_channels, act=act) for _ in range(num_blocks)
        ])
        self.conv_norm_layer1 = ConvNormLayer(in_channels, out_channels, 1, 1, bias=bias, act=act)
        self.conv_norm_layer2 = ConvNormLayer(in_channels, out_channels, 1, 1, bias=bias, act=act)
        if size_flag == 0:
            kernel_sizes = [3, 5, 7]
        else:
            kernel_sizes = [1, 3, 5]
        self.cs_conv_attn_denoise = CrossScaleConvolutionalAttentionDenoising(out_channels, kernel_sizes)

    def forward(self, x):
        x_1= self.conv_norm_layer1(x)
        x_2 = self.rep_vgg_blocks(self.cs_conv_attn_denoise(self.conv_norm_layer2(x)))
        x_fused = x_1 + x_2
        return x_fused


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register
class MultiBandMultiScaleDenoisingEncoder(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        self.conv_norm_layers = nn.ModuleList()
        self.mfed1_blocks = nn.ModuleList()
        for i in range(len(in_channels) - 1, 0, -1):
            self.conv_norm_layers.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.mfed1_blocks.append(
                MultiScaleEnhancementDenoising(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act,
                                               size_flag=i)  # size_flag: dynamic multi-scale flag
            )
        self.m2dd = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.m2dd.append(
                MultiBandMultiScaleDD(hidden_dim, hidden_dim, size_flag=i)  # size_flag: dynamic multi-scale flag
            )

        self.mfed2_blocks = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.mfed2_blocks.append(
                MultiScaleEnhancementDenoising(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act,
                                               size_flag=i)  # size_flag: dynamic multi-scale flag
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):

        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            high_level_feat = inner_outs[0]
            low_level_feat = proj_feats[idx - 1]
            high_level_feat = self.conv_norm_layers[len(self.in_channels) - 1 - idx](high_level_feat)
            inner_outs[0] = high_level_feat
            up_sample_feat = F.interpolate(high_level_feat, scale_factor=2., mode='nearest')
            mfed_out = self.mfed1_blocks[len(self.in_channels) - 1 - idx](torch.concat([up_sample_feat, low_level_feat], dim=1))
            inner_outs.insert(0, mfed_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            low_level_feat = outs[-1]
            high_level_feat = inner_outs[idx + 1]
            m2dd_feat = self.m2dd[idx](low_level_feat)
            out = self.mfed2_blocks[idx](torch.concat([m2dd_feat, high_level_feat], dim=1))
            outs.append(out)

        return outs
