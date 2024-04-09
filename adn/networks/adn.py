import torch
import torch.nn as nn
import functools
from copy import deepcopy, copy
from .blocks import ConvolutionBlock, ResidualBlock, FullyConnectedBlock
from ..models.segformer import SegFormer
from ..utils import print_model, FunctionModel
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_ch, base_ch, num_down, num_residual, res_norm='instance', down_norm='instance'):
        super(Encoder, self).__init__()

        self.conv0 = ConvolutionBlock(
            in_channels=input_ch, out_channels=base_ch, kernel_size=7, stride=1,
            padding=3, pad='reflect', norm=down_norm, activ='relu')
        
        output_ch = base_ch
        for i in range(1, num_down+1):
            m = ConvolutionBlock(
                in_channels=output_ch, out_channels=output_ch * 2, kernel_size=4,
                stride=2, padding=1, pad='reflect', norm=down_norm, activ='relu')
            setattr(self, "conv{}".format(i), m)
            output_ch *= 2

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                ResidualBlock(output_ch, pad='reflect', norm=res_norm, activ='relu'))

        self.layers = [getattr(self, "conv{}".format(i)) for i in range(num_down+1)] + \
            [getattr(self, "res{}".format(i)) for i in range(num_residual)]
        
    def forward(self, x):
        sides = []
        for layer in self.layers:
            x = layer(x)
            sides.append(x)
        print(x.shape, len(sides[::-1]),sides[::-1][0].shape,sides[::-1][1].shape,sides[::-1][2].shape,sides[::-1][3].shape,sides[::-1][4].shape,sides[::-1][5].shape,sides[::-1][6].shape)
        return x, sides[::-1]


class Decoder(nn.Module):
    def __init__(self, output_ch, base_ch, num_up, num_residual, num_sides, res_norm='instance', up_norm='layer', fuse=False):
        super(Decoder, self).__init__()
        input_ch = base_ch * 2 ** num_up
        input_chs = []

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                ResidualBlock(input_ch, pad='reflect', norm=res_norm, activ='lrelu'))
            input_chs.append(input_ch)

        for i in range(num_up):
            m = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvolutionBlock(
                    in_channels=input_ch, out_channels=input_ch // 2, kernel_size=5,
                    stride=1, padding=2, pad='reflect', norm=up_norm, activ='lrelu'))
            setattr(self, "conv{}".format(i), m)
            input_chs.append(input_ch)
            input_ch //= 2

        m = ConvolutionBlock(
            in_channels=base_ch, out_channels=output_ch, kernel_size=7,
            stride=1, padding=3, pad='reflect', norm='none', activ='tanh')
        setattr(self, "conv{}".format(num_up), m)
        input_chs.append(base_ch)
        
        self.layers = [getattr(self, "res{}".format(i)) for i in range(num_residual)] + \
            [getattr(self, "conv{}".format(i)) for i in range(num_up + 1)]

        if fuse:
            input_chs = input_chs[-num_sides:]
            for i in range(num_sides):
                setattr(self, "fuse{}".format(i),
                    nn.Conv2d(input_chs[i] * 2, input_chs[i], 1)) #nn.Conv2d是二维卷积方法
                    #nn.Transformer(input_chs[i] * 2, input_chs[i], 1))
            self.fuse = lambda x, y, i: getattr(self, "fuse{}".format(i))(torch.cat((x, y), 1))
        else:
            self.fuse = lambda x, y, i: x + y

    def forward(self, x, sides):
        m, n = len(self.layers), len(sides)
        assert m >= n, "Invalid side inputs"

        for i in range(m - n):
            x = self.layers[i](x)

        for i, j in enumerate(range(m - n, m)):

            print(x.shape, sides.shape)
            x = self.fuse(x, sides, i)

        return x


class Decoder_new(nn.Module):
    def __init__(self, output_ch, base_ch, num_up, num_residual, num_sides, res_norm='instance', up_norm='layer',
                 fuse=False):
        super(Decoder_new, self).__init__()
        input_ch = base_ch * 2 ** num_up
        input_chs = []

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                    ResidualBlock(input_ch, pad='reflect', norm=res_norm, activ='lrelu'))
            input_chs.append(input_ch)

        for i in range(num_up):
            m = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvolutionBlock(
                    in_channels=input_ch, out_channels=input_ch // 2, kernel_size=5,
                    stride=1, padding=2, pad='reflect', norm=up_norm, activ='lrelu'))
            setattr(self, "conv{}".format(i), m)
            input_chs.append(input_ch)
            input_ch //= 2

        m = ConvolutionBlock(
            in_channels=base_ch, out_channels=output_ch, kernel_size=7,
            stride=1, padding=3, pad='reflect', norm='none', activ='tanh')
        setattr(self, "conv{}".format(num_up), m)
        input_chs.append(base_ch)

        self.layers = [getattr(self, "res{}".format(i)) for i in range(num_residual)] + \
                      [getattr(self, "conv{}".format(i)) for i in range(num_up + 1)]

        if fuse:
            input_chs = input_chs[-num_sides:]
            for i in range(num_sides):
                setattr(self, "fuse{}".format(i),
                        nn.Conv2d(input_chs[i] * 2, input_chs[i], 1))  # nn.Conv2d是二维卷积方法
                # nn.Transformer(input_chs[i] * 2, input_chs[i], 1))
            self.fuse = lambda x, y, i: getattr(self, "fuse{}".format(i))(torch.cat((x, y), 1))
        else:
            self.fuse = lambda x, y, i: x + y

    def forward(self, x, sides):
        m, n = len(self.layers), len(sides)

        for i, j in enumerate(range(m - n, m)):
            print(x.shape, sides.shape)
            x = self.fuse(x, sides, i)
        return x


class ADN(nn.Module):
    def __init__(self, input_ch=1, base_ch=64, num_down=2, num_residual=4, num_sides="all",
        res_norm='instance', down_norm='instance', up_norm='layer', fuse=True, shared_decoder=False):
        super(ADN, self).__init__()

        self.n = num_down + num_residual + 1 if num_sides == "all" else num_sides
        self.encoder_low = SegFormer(num_classes=1, phi='b5', pretrained=True)
        self.encoder_high = SegFormer(num_classes=1, phi='b5', pretrained=True)
        self.encoder_art = SegFormer(num_classes=1, phi='b5', pretrained=True)
        self.decoder = Decoder_new(input_ch, base_ch, num_down, num_residual, self.n, res_norm, up_norm, fuse=False)
        self.decoder_art = self.decoder if shared_decoder else deepcopy(self.decoder)

    def forward1(self, x_low):
        y1 = self.encoder_art(x_low)  # encode artifact
        self.saved = (x_low, y1)
        y2 = self.encoder_low(x_low)  # encode low quality image
        print(y2.shape)
        return y1, y2

    def forward2(self, x_low, x_high):
        if hasattr(self, "saved") and self.saved[0] is x_low: sides = self.saved[1]
        else: sides = self.encoder_art(x_low)
        y2 = self.encoder_high(x_high)
        y1 = self.decoder_art(y2, sides)
        return y1, y2

    def forward_lh(self, x_low):
        y = self.encoder_low(x_low)
        return y

    def forward_hl(self, x_low, x_high):
        sides = self.encoder_art(x_low)
        code= self.encoder_high(x_high)
        y = self.decoder_art(code, sides)
        return y


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) is str:
            norm_layer = {
                "layer": nn.LayerNorm,
                "instance": nn.InstanceNorm2d,
                "batch": nn.BatchNorm2d,
              "none": None}[norm_layer]

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)] + \
                ([norm_layer(ndf * nf_mult)] if norm_layer else []) + [nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)] + \
            ([norm_layer(ndf * nf_mult)] if norm_layer else []) + [nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)