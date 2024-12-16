import torch.nn as nn
from torch.nn import functional as F
import torch
from LossAndEval import DiceLoss
import math
from collections import OrderedDict
import re
import torch.utils.model_zoo as model_zoo

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

class CAS(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        # F_g为编码器特征
        # F_l为解码器特征
        # F_int为输出通道
        super(CAS, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        r = 16
        if F_int <= 64:
            r = F_int // 8
        self.se = SEBlock(F_int, r=r)

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)

        psi_se = self.se(psi)

        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi_se)
        # 返回加权的 x
        return x * psi


# 无残差的通道注意力
class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),  # 缩减通道,减少参数量
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),  # 将值规范到0~1
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale
        y = torch.mul(x, y)  # 通道注意力

        return y


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        # F_g为编码器特征
        # F_l为解码器特征
        # F_int为输出通道
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi

class _DenseLayer(nn.Sequential):
    """Basic unit of DenseBlock (using bottleneck layer) """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size*growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    """DenseBlock"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size,
                                drop_rate)
            self.add_module("denselayer%d" % (i+1,), layer)


# 作用，减少通道数，去除了池化操作
class _Transition(nn.Sequential):
    """Transition layer between two adjacent DenseBlock"""
    def __init__(self, num_input_feature, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_feature))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_feature, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        # self.add_module("pool", nn.AvgPool2d(2, stride=2))


# 权重标准化的卷积操作
class Conv2d(nn.Conv2d):
    '''
    shape:
    input: (Batch_size, in_channels, H_in, W_in)
    output: ((Batch_size, out_channels, H_out, W_out))
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)

    # 权重标椎化
    def forward(self, x):
        weight = self.weight  # self.weight 的shape为(out_channels, in_channels, kernel_size_w, kernel_size_h)
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def conv2d_3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, weight_std=False):
    '''3x3 convolution with padding'''
    if weight_std:
        return Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, bias=bias)


class NoBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, multi_grid=1, weight_std=False,
                 att=False):
        super(NoBottleneck, self).__init__()
        self.weight_std = weight_std

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv2d_3x3(inplanes, planes, kernel_size=3, stride=stride, padding=1,
                                dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv2d_3x3(planes, planes, kernel_size=3, stride=1, padding=1, dilation=dilation * multi_grid,
                                bias=False, weight_std=self.weight_std)


        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x  # 当x的通道数与out的通道数不同时，在_make_layer中进行了下采样，主要就是针对残差连接
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        return out


class Up(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=False):
        super(Up, self).__init__()
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_size, out_size, 1))

    def forward(self, high_feature):
        outputs0 = self.up(high_feature)

        return outputs0

class Decoder_d1(nn.Module):
    def __init__(self, inplanes, weight_std, channels, is_deconv, n_classes_d1):
        super(Decoder_d1, self).__init__()

        self.inplanes = inplanes
        self.weight_std = weight_std

        self.up1_d1 = Up(2 * channels[3], channels[3], is_deconv)
        self.d1_resb_d1 = self._make_layer(NoBottleneck, 2 * channels[3], channels[3], 1, stride=1)

        self.up2_d1 = Up(channels[3], channels[2], is_deconv)
        self.d2_resb_d1 = self._make_layer(NoBottleneck, 2 * channels[2], channels[2], 1, stride=1)

        self.up3_d1 = Up(channels[2], channels[1], is_deconv)
        self.d3_resb_d1 = self._make_layer(NoBottleneck, 2 * channels[1], channels[1], 1, stride=1)

        self.up4_d1 = Up(channels[1], channels[0], is_deconv)
        self.d4_resb_d1 = self._make_layer(NoBottleneck, 2 * channels[0], channels[0], 1, stride=1)

        # 解码器1-跳跃连接
        self.skip0_att_d1 = Attention_block(channels[0], channels[0], channels[0] // 2)
        self.skip1_att_d1 = Attention_block(channels[1], channels[1], channels[1] // 2)
        # self.skip2_att_d1 = Attention_block(channels[2], channels[2], channels[2] // 2)

        self.cls_conv_d1 = nn.Conv2d(channels[0], n_classes_d1, kernel_size=1)

    def forward(self, x, skip3_low, skip2_low, skip1_low, skip0_low):
        # 解码器1 jsrt
        x_d1 = self.up1_d1(x)
        skip3_d1 = torch.cat((skip3_low, x_d1), dim=1)
        d1_d1 = self.d1_resb_d1(skip3_d1)

        x_d1 = self.up2_d1(d1_d1)
        # skip2_att_d1 = self.skip2_att_d1(x_d1, skip2_low)
        skip2_d1 = torch.cat((skip2_low, x_d1), dim=1)
        d2_d1 = self.d2_resb_d1(skip2_d1)

        x_d1 = self.up3_d1(d2_d1)
        skip1_att_d1 = self.skip1_att_d1(x_d1, skip1_low)
        skip1_d1 = torch.cat((skip1_att_d1, x_d1), dim=1)
        d3_d1 = self.d3_resb_d1(skip1_d1)

        x_d1 = self.up4_d1(d3_d1)
        skip0_att_d1 = self.skip0_att_d1(x_d1, skip0_low)
        skip0_d1 = torch.cat((skip0_att_d1, x_d1), dim=1)
        d4_d1 = self.d4_resb_d1(skip0_d1)

        output_d1 = self.cls_conv_d1(d4_d1)

        return output_d1

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, dilation=1, multi_grid=1, att=False):
        # block为使用的nobottleneck, blocks为使用几块block
        downsample = None  # 具体是否进行下采样由步长和输入输出的通道决定

        if stride != 1 or inplanes != planes:  # 步长不为1， 输入与输出不等
            downsample = nn.Sequential(
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=True),
                conv2d_3x3(inplanes, planes, kernel_size=1, stride=stride, padding=0, weight_std=self.weight_std),
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1

        if stride == 1 and att:
            layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                                multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std, att=att))
        else:
            layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                                multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))

        # self.inplanes = planes
        # 只有块>1才会进入循环
        for i in range(1, blocks):
            if i == blocks - 1:
                # 最后一个添加att
                layers.append(
                    block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                          weight_std=self.weight_std, att=att))
            else:
                layers.append(
                    block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                          weight_std=self.weight_std))

        return nn.Sequential(*layers)


class Decoder_d2(nn.Module):
    def __init__(self, inplanes, weight_std, channels, is_deconv, n_classes_d2):
        super(Decoder_d2, self).__init__()

        self.inplanes = inplanes
        self.weight_std = weight_std

        # 解码器2 bone
        self.up1_d2 = Up(2 * channels[3], channels[3], is_deconv)
        self.d1_resb_d2 = self._make_layer(NoBottleneck, 2 * channels[3], channels[3], 1, stride=1)

        self.up2_d2 = Up(channels[3], channels[2], is_deconv)
        self.d2_resb_d2 = self._make_layer(NoBottleneck, 2 * channels[2], channels[2], 1, stride=1)

        self.up3_d2 = Up(channels[2], channels[1], is_deconv)
        self.d3_resb_d2 = self._make_layer(NoBottleneck, 2 * channels[1], channels[1], 1, stride=1)

        self.up4_d2 = Up(channels[1], channels[0], is_deconv)
        self.d4_resb_d2 = self._make_layer(NoBottleneck, 2 * channels[0], channels[0], 1, stride=1)

        # 解码器2-跳跃连接
        self.skip0_att_d2 = CAS(channels[0], channels[0], channels[0] // 2)
        self.skip1_att_d2 = CAS(channels[1], channels[1], channels[1] // 2)
        # self.skip2_att_d2 = CAS(channels[2], channels[2], channels[2] // 2)

        self.cls_conv_d2 = nn.Conv2d(channels[0], n_classes_d2, kernel_size=1)


    def forward(self, x, skip3_low, skip2_low, skip1_low, skip0_low):
        x_d2 = self.up1_d2(x)
        skip3_d2 = torch.cat((skip3_low, x_d2), dim=1)
        d1_d2 = self.d1_resb_d2(skip3_d2)

        x_d2 = self.up2_d2(d1_d2)
        # skip2_att_d2 = self.skip2_att_d2(x_d2, skip2_low)
        skip2_d2 = torch.cat((skip2_low, x_d2), dim=1)
        d2_d2 = self.d2_resb_d2(skip2_d2)

        x_d2 = self.up3_d2(d2_d2)
        skip1_att_d2 = self.skip1_att_d2(x_d2, skip1_low)
        skip1_d2 = torch.cat((skip1_att_d2, x_d2), dim=1)
        d3_d2 = self.d3_resb_d2(skip1_d2)

        x_d2 = self.up4_d2(d3_d2)
        skip0_att_d2 = self.skip0_att_d2(x_d2, skip0_low)
        skip0_d2 = torch.cat((skip0_att_d2, x_d2), dim=1)
        d4_d2 = self.d4_resb_d2(skip0_d2)

        output_d2 = self.cls_conv_d2(d4_d2)

        return output_d2

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, dilation=1, multi_grid=1, att=False):
        # block为使用的nobottleneck, blocks为使用几块block
        downsample = None  # 具体是否进行下采样由步长和输入输出的通道决定

        if stride != 1 or inplanes != planes:  # 步长不为1， 输入与输出不等
            downsample = nn.Sequential(
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=True),
                conv2d_3x3(inplanes, planes, kernel_size=1, stride=stride, padding=0, weight_std=self.weight_std),
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1

        if stride == 1 and att:
            layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                                multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std, att=att))
        else:
            layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                                multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))

        # self.inplanes = planes
        # 只有块>1才会进入循环
        for i in range(1, blocks):
            if i == blocks - 1:
                # 最后一个添加att
                layers.append(
                    block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                          weight_std=self.weight_std, att=att))
            else:
                layers.append(
                    block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                          weight_std=self.weight_std))

        return nn.Sequential(*layers)

class DenseNet_resDecoder(nn.Module):
    "DenseNet-BC model"
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, compression_rate=0.5, drop_rate=0.2, n_classes_d1=3, n_classes_d2=4):
        """
        :param growth_rate: (int) number of filters used in DenseLayer, `k` in the paper
        :param block_config: (list of 4 ints) number of layers in each DenseBlock
        :param num_init_features: (int) number of filters in the first Conv2d
        :param bn_size: (int) the factor using in the bottleneck layer
        :param compression_rate: (float) the compression rate used in Transition Layer
        :param drop_rate: (float) the drop rate after each DenseLayer
        :param num_classes: (int) number of classes for classification
        """
        super(DenseNet_resDecoder, self).__init__()

        self.inplanes = 128
        self.weight_std = False
        channels = [64, 128, 256, 512]

        # first Conv2d
        self.conv1 = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("conv1", nn.Conv2d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ("norm1", nn.BatchNorm2d(num_init_features)),
            ("relu1", nn.ReLU(inplace=True)),
            # ("pool0", nn.MaxPool2d(3, stride=2, padding=1))
        ]))

        # 最大池化
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # 平均池化
        # self.avgpool = nn.AvgPool2d(kernel_size=2)
        # 上采样
        # self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.upsamplex2 = nn.UpsamplingBilinear2d(scale_factor=2)

        # DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i != len(block_config) - 1:
                block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
                self.add_module("denseblock%d" % (i + 1), block)
                num_features += num_layers*growth_rate
            if i != len(block_config) - 1:  #
                transition = _Transition(num_features, int(num_features*compression_rate))
                self.add_module("transition%d" % (i + 1), transition)
                num_features = int(num_features * compression_rate)

        self.center = _DenseBlock(block_config[-1], num_features, bn_size, growth_rate, drop_rate)

        self.deconder_d1 = Decoder_d1(self.inplanes, self.weight_std, channels, False, n_classes_d1)
        self.deconder_d2 = Decoder_d2(self.inplanes, self.weight_std, channels, False, n_classes_d2)

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear) and m.bias == True:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mode='train', task_type='bone'):
        skip0_low = self.conv1(x)
        x = self.maxpool(skip0_low)

        x = self.denseblock1(x)
        skip1_low = self.transition1(x)
        x = self.maxpool(skip1_low)

        x = self.denseblock2(x)
        skip2_low = self.transition2(x)
        x = self.maxpool(skip2_low)

        x = self.denseblock3(x)
        skip3_low = self.transition3(x)
        x = self.maxpool(skip3_low)

        x = self.center(x)

        if mode == 'train':
            if task_type == 'bone':
                output_d2 = self.deconder_d2(x, skip3_low, skip2_low, skip1_low, skip0_low)
                return output_d2

            if task_type == 'jsrt':
                output_d1 = self.deconder_d1(x, skip3_low, skip2_low, skip1_low, skip0_low)
                return output_d1
        else:
            output_d1 = self.deconder_d1(x, skip3_low, skip2_low, skip1_low, skip0_low)
            output_d2 = self.deconder_d2(x, skip3_low, skip2_low, skip1_low, skip0_low)

            return [output_d1, output_d2]




def densenet121_resDecoder(pretrained=True, num_classes_d1=3, num_classes_d2=4, **kwargs):
    """DenseNet121"""
    print('using densenet121_resDecoder 1center-denseblock bone 2CAS jsrt 2AG 2layer split')
    model = DenseNet_resDecoder(num_init_features=64, n_classes_d1=num_classes_d1, n_classes_d2=num_classes_d2, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)

    model_dict = model.state_dict()

    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        pattern2 = re.compile(
            r'^(.*transition\d+\.(?:norm|relu|conv))\.(?:weight|bias|running_mean|running_var)'
        )

        # pattern3 = re.compile(r'^(features.conv0|feature.norm0)')

        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            # print(key)
            res = pattern.match(key)
            res_t = pattern2.match(key)
            # print(res_t)

            # 最开始一层的加载
            if 'features.conv0' in key or 'features.norm0' in key:
                # new_key = 'conv1.' + key[9:]
                # state_dict[new_key] = state_dict[key]
                del state_dict[key]

            # denseblock有norm.1.weight这种形式的
            if res:
                new_key = res.group(1)[9:] + res.group(2)
                # print(new_key)
                if 'denseblock4' in key:
                    state_dict['center.' + new_key[12:]] = state_dict[key]
                    # state_dict['center_d2.' + new_key[12:]] = state_dict[key]
                else:
                    state_dict[new_key] = state_dict[key]
                del state_dict[key]
            # transition1层：features.transition1.norm.bias
            if res_t:
                new_key = key[9:]
                # print(new_key)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
            # 删除最后的分类
            if 'features.norm5' in key or 'classifier' in key:
                del state_dict[key]

        pretrained_dict = {k: v for k, v in state_dict.items() if k in state_dict}
        model_dict.update(pretrained_dict)

        model.load_state_dict(model_dict)
        # model.load_state_dict(state_dict)
    return model



