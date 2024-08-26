import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from SDEConv import SDEConv
from einops import rearrange


class SDE_Encoder(nn.Module):
    def __init__(self, in_c, n_feat):
        super(SDE_Encoder, self).__init__()
        self.img_to_feat = Base_Block(in_c, n_feat // 8)
        self.aspp1 = ASPP(dim=n_feat // 8, in_dim=n_feat // 8)
        encoder_level1 = []
        encoder_level1.append(Base_Block(n_feat // 8, n_feat // 4))
        encoder_level1.append(ASPP(dim=n_feat // 4, in_dim=n_feat // 4))

        self.compress1 = nn.Conv2d(in_channels=n_feat // 4, out_channels=n_feat // 8, kernel_size=1)

        self.sdec1 = SDEConv(n_feat // 8)

        encoder_level2 = []
        encoder_level2.append(Base_Block(n_feat // 4, n_feat // 2))
        encoder_level2.append(ASPP(dim=n_feat // 2, in_dim=n_feat // 2))

        self.compress2_1 = nn.Conv2d(in_channels=n_feat // 2, out_channels=n_feat // 4, kernel_size=1)

        self.sdec2 = SDEConv(n_feat // 4)
        encoder_level3 = []
        encoder_level3.append(Base_Block(n_feat // 2, n_feat))

        self.encoder_level1 = nn.Sequential(*encoder_level1)
        self.encoder_level2 = nn.Sequential(*encoder_level2)
        self.encoder_level3 = nn.Sequential(*encoder_level3)

        self.down12 = DownSample()
        self.down23 = DownSample()

    def forward(self, x):
        x = self.img_to_feat(x)
        inf1 = self.encoder_level1(x)
        reduce_enc1 = self.compress1(inf1)
        inf1 = torch.cat([self.sdec1(reduce_enc1), reduce_enc1], 1)
        x = self.down12(inf1)
        inf2 = self.encoder_level2(x)
        reduce_enc2 = self.compress2_1(inf2)
        inf2 = torch.cat([self.sdec2(reduce_enc2), reduce_enc2], 1)
        inf3 = self.down23(inf2)
        inf3 = self.encoder_level3(inf3)
        return inf1, inf2, inf3


class Base_Block(nn.Module):
    def __init__(self, inc, outc , DW_Expand=1, FFN_Expand=2):
        super().__init__()
        dw_channel = inc * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=inc, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        # Channel Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )

        self.GELU = nn.GELU()

        ffn_channel = FFN_Expand * inc
        self.conv4 = nn.Conv2d(in_channels=inc, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=inc, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv6 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.beta = nn.Parameter(torch.zeros((1, inc, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, inc, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.conv1(x)
        x = self.conv2(x)
        x = x * self.se(x)
        x = self.GELU(x)
        x = self.conv3(x)
        x = self.GELU(x)
        y = inp + x * self.beta
        x = self.conv4(y)
        x = self.GELU(x)
        x = self.conv5(x)
        x = self.GELU(x)

        x = y + x * self.gamma

        x = self.conv6(x)
        x = self.GELU(x)
        return x
class ChannelAttention(nn.Module):

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        if num_feat<=8:
            squeeze_factor = 8
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.GELU(),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='BiasFree'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
# multi scale  module
class ASPP(nn.Module):

    def __init__(self, dim, in_dim):
        super(ASPP, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), LayerNorm(in_dim),
                                       nn.GELU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=1), LayerNorm(down_dim), nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2),
                                   LayerNorm(down_dim), nn.GELU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4),
                                   LayerNorm(down_dim), nn.GELU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6),
                                   LayerNorm(down_dim), nn.GELU())
        self.conv5 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=1), LayerNorm(down_dim), nn.GELU())
        self.catt = ChannelAttention(num_feat=5 * down_dim)
        self.fuse = nn.Sequential(nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), LayerNorm(in_dim), nn.GELU())

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = self.conv5(x)
        # conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear')
        fuse_conv = torch.cat((conv1, conv2, conv3, conv4, conv5), 1)
        fuse_conv = fuse_conv + self.catt(fuse_conv)
        return self.fuse(fuse_conv)


class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False))

    def forward(self, x):
        x = self.down(x)
        return x

