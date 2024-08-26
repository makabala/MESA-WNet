import torch
import torch.nn as nn
import warnings
import math
from torchvision.ops.deform_conv import DeformConv2d


class SCARM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SCARM, self).__init__()

        self.conv_head = Patch_Embed_stage(3, out_channels)

        self.dilated_block_LH = Dilated_Resblock(out_channels, out_channels)
        self.dilated_block_HL = Dilated_Resblock(out_channels, out_channels)

        self.cross_attention0 = cross_attention(out_channels, num_heads=8)
        self.dilated_block_HH = Dilated_Resblock(out_channels, out_channels)

        self.conv_HH = nn.Conv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1)
        self.cross_attention1 = cross_attention(out_channels, num_heads=8)

        self.fft_block = FFT_Block(nChannels=out_channels)
        self.conv = nn.Conv2d(in_channels*2,in_channels,3,1,1)
        self.conv_tail = Patch_Embed_stage(out_channels, in_channels)

    def forward(self, x):

        b, c, h, w = x.shape

        residual = x

        x = self.conv_head(x)
        fft,mag , pha = self.fft_block(x)

        x_HL, x_LH, x_HH = x[:b//3, ...], x[b//3:2*b//3, ...], x[2*b//3:, ...]
        fft_HL, fft_LH, fft_HH = fft[:b // 3, ...], x[b // 3:2 * b // 3, ...], x[2 * b // 3:, ...]

        x_HH_LH = self.cross_attention0(x_LH, x_HH)
        x_HH_HL = self.cross_attention1(x_HL, x_HH)
        #
        ffx_HH_LH = self.cross_attention0(fft_LH, fft_HH)
        ffx_HH_HL = self.cross_attention1(fft_HL, fft_HH)

        x_HL = self.dilated_block_HL(x_HL)
        x_LH = self.dilated_block_LH(x_LH)

        x_HH = self.dilated_block_HH(self.conv_HH(torch.cat((x_HH_LH, x_HH_HL), dim=1)))
        fft_HH = self.dilated_block_HH(self.conv_HH(torch.cat((ffx_HH_LH, ffx_HH_HL), dim=1)))

        out = self.conv_tail(torch.cat((x_HL, x_LH, x_HH), dim=0))
        fftout = self.conv_tail(torch.cat((fft_HL, fft_LH, fft_HH), dim=0))
        out = self.conv(torch.cat((out,fftout),dim=1))

        return out + residual, mag, pha




class cross_attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(cross_attention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = Depth_conv(in_ch=dim, out_ch=dim)
        self.key = Depth_conv(in_ch=dim, out_ch=dim)
        self.value = Depth_conv(in_ch=dim, out_ch=dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        '''
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.attention_head_size,
        )
        print(new_x_shape)
        x = x.view(*new_x_shape)
        '''
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ctx):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layer = self.value(ctx)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        ctx_layer = torch.matmul(attention_probs, value_layer)
        ctx_layer = ctx_layer.permute(0, 2, 1, 3).contiguous()

        return ctx_layer


class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class DWConv2d_BN(nn.Module):

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
            offset_clamp=(-1, 1)
    ):
        super().__init__()
        # dw
        # self.conv=torch.nn.Conv2d(in_ch,out_ch,kernel_size,stride,(kernel_size - 1) // 2,bias=False,)
        # self.mask_generator = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3,
        #                                                 stride=1, padding=1, bias=False, groups=in_ch),
        #                                       nn.Conv2d(in_channels=in_ch, out_channels=9,
        #                                                 kernel_size=1,
        #                                                 stride=1, padding=0, bias=False)
        #                                      )
        self.offset_clamp = offset_clamp
        self.offset_generator = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3,
                                                        stride=1, padding=1, bias=False, groups=in_ch),
                                              nn.Conv2d(in_channels=in_ch, out_channels=18,
                                                        kernel_size=1,
                                                        stride=1, padding=0, bias=False)

                                              )
        self.dcn = DeformConv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=in_ch
        )  # .cuda(7)
        self.pwconv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

        # self.bn = norm_layer(out_ch)
        self.act = act_layer() if act_layer is not None else nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                # print(m)

        #   elif isinstance(m, nn.BatchNorm2d):
        #     m.weight.data.fill_(bn_weight_init)
        #      m.bias.data.zero_()

    def forward(self, x):

        # x=self.conv(x)
        # x = self.bn(x)
        # x = self.act(x)
        # mask= torch.sigmoid(self.mask_generator(x))
        # print('1')
        offset = self.offset_generator(x)
        # print('2')
        if self.offset_clamp:
            offset = torch.clamp(offset, min=self.offset_clamp[0], max=self.offset_clamp[1])  # .cuda(7)1
        # print(offset)
        # print('3')
        # x=x.cuda(7)
        x = self.dcn(x, offset)
        # x=x.cpu()
        # print('4')
        x = self.pwconv(x)
        # print('5')
        # x = self.bn(x)
        x = self.act(x)
        return x


class DWCPatchEmbed(nn.Module):
    """Depthwise Convolutional Patch Embedding layer Image to Patch
    Embedding."""

    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 idx=0,
                 act_layer=nn.Hardswish,
                 offset_clamp=(-1, 1)):
        super().__init__()

        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=act_layer,
            offset_clamp=offset_clamp
        )
        """
        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=act_layer,
        )
        """

    def forward(self, x):
        """foward function"""
        x = self.patch_conv(x)

        return x
class Patch_Embed_stage(nn.Module):
    """Depthwise Convolutional Patch Embedding stage comprised of
    `DWCPatchEmbed` layers."""

    def __init__(self, in_chans, embed_dim, num_path=4, isPool=False, offset_clamp=(-1, 1)):
        super(Patch_Embed_stage, self).__init__()

        # self.patch_embeds = nn.ModuleList([
        self.DW = DWCPatchEmbed(
                in_chans=in_chans,
                embed_dim=embed_dim,
                patch_size=3,
                stride=1,
                idx=0,
                offset_clamp=offset_clamp
            )


    def forward(self, x):
        """foward function"""
        # att_inputs = []
        # for pe in self.patch_embeds:
        x = self.DW(x)
        #$att_inputs.append(x)

        return x
class Dilated_Resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dilated_Resblock, self).__init__()

        sequence = list()
        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),

            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=3, dilation=(3, 3)),

            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),

            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1))

        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x) + x

        return out


class make_fdense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=1):
        super(make_fdense, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False),nn.BatchNorm2d(growthRate)
        )
        self.bat = nn.BatchNorm2d(growthRate),
        self.leaky=nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x):
        out = self.leaky(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class FFT_Block(nn.Module):
    def __init__(self, nChannels, nDenselayer=2, growthRate=32):
        super(FFT_Block, self).__init__()
        nChannels_1 = nChannels
        nChannels_2 = nChannels
        modules1 = []
        for i in range(nDenselayer):
            modules1.append(make_fdense(nChannels_1, growthRate))
            nChannels_1 += growthRate
        self.dense_layers1 = nn.Sequential(*modules1)
        modules2 = []
        for i in range(nDenselayer):
            modules2.append(make_fdense(nChannels_2, growthRate))
            nChannels_2 += growthRate
        self.dense_layers2 = nn.Sequential(*modules2)
        self.conv_1 = nn.Conv2d(nChannels_1, nChannels, kernel_size=1, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(nChannels_2, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        _, _, H, W = x.shape

        x_freq = torch.fft.rfft2(x, norm='backward')
        # print(x_freq.shape)
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)


        mag = self.dense_layers1(mag)

        mag = self.conv_1(mag)
        # print(mag.shape)
        pha = self.dense_layers2(pha)
        pha = self.conv_2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)

        x_out = torch.complex(real, imag)
        out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        out = out + x
        return out , mag , pha


