import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# from read_ini import read_ini
# from GetPath import PathParam

# _iniPath = PathParam().getiniPath()    
# cf, val = read_ini(_iniPath) 







__all__ = ['VoVNet', 'vovnet_slim', 'vovnet39', 'vovnet57', 'vovnet_slice']

model_urls = {
    'vovnet39': 'https://dl.dropbox.com/s/1lnzsgnixd8gjra/vovnet39_torchvision.pth?dl=1',
    'vovnet57': 'https://dl.dropbox.com/s/6bfu9gstbwfw31m/vovnet57_torchvision.pth?dl=1'
}


class eSEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1,
                            padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


'''
def DFConv3x3(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, 
                with_modulated_dcn=None, deformable_groups=None):
    """3x3 convolution with padding"""
    return [
        (f'{module_name}_{postfix}/conv',
         DFConv2d(in_channels, out_channels, with_modulated_dcn=with_modulated_dcn,
            kernel_size=kernel_size, stride=stride, groups=groups, 
            deformable_groups=deformable_groups, bias=False)),
        (f'{module_name}_{postfix}/norm',
            group_norm(out_channels) if _GN else FrozenBatchNorm2d(out_channels)
        ),
        (f'{module_name}_{postfix}/relu', nn.ReLU(inplace=True))
    ]
'''


def conv3x3(in_channels, out_channels, module_name, postfix,
            stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=groups,
                   bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
         nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
         nn.ReLU(inplace=True)),
        # nn.LeakyReLU(inplace=True)),
    ]


def conv1x1(in_channels, out_channels, module_name, postfix,
            stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=groups,
                   bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
         nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
         nn.ReLU(inplace=True)),
    ]


class _OSA_module(nn.Module):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 layer_per_block,
                 module_name,
                 SE=True,
                 identity=False):
        super(_OSA_module, self).__init__()

        self.identity = identity
        self.layers = nn.ModuleList()
        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(nn.Sequential(
                OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1(in_channel, concat_ch, module_name, 'concat')))
        self.ese = eSEModule(concat_ch)

    def forward(self, x):
        identity_feat = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)
        xt = self.ese(xt)
        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 block_per_stage,
                 layer_per_block,
                 stage_num,
                 SE=True, ):
        super(_OSA_stage, self).__init__()

        if not stage_num == 2:
            self.add_module('Pooling',
                            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        if block_per_stage != 1:
            SE = False
        module_name = f'OSA{stage_num}_1'
        self.add_module(module_name,
                        _OSA_module(in_ch,
                                    stage_ch,
                                    concat_ch,
                                    layer_per_block,
                                    module_name,
                                    SE=SE, ))
        for i in range(block_per_stage - 1):
            if i != block_per_stage - 2:  # last block
                SE = False
            module_name = f'OSA{stage_num}_{i + 2}'
            self.add_module(module_name,
                            _OSA_module(concat_ch,
                                        stage_ch,
                                        concat_ch,
                                        layer_per_block,
                                        module_name,
                                        SE=SE,
                                        identity=True))


class VoVNet(nn.Module):
    def __init__(self,
                 config_stage_ch,
                 config_concat_ch,
                 block_per_stage,
                 layer_per_block,
                 num_classes=9):
        super(VoVNet, self).__init__()

        # Stem module
        stem = conv3x3(3, 64, 'stem', '1', 2)
        stem += conv3x3(64, 64, 'stem', '2', 1)
        stem += conv3x3(64, 128, 'stem', '3', 2)
        self.add_module('stem', nn.Sequential(OrderedDict(stem)))

        stem_out_ch = [128]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        self.stage_names = []
        for i in range(4):  # num_stages
            name = 'stage%d' % (i + 2)
            self.stage_names.append(name)
            self.add_module(name,
                            _OSA_stage(in_ch_list[i],
                                       config_stage_ch[i],
                                       config_concat_ch[i],
                                       block_per_stage[i],
                                       layer_per_block,
                                       i + 2, True))

        # self.classifier = nn.Linear(config_concat_ch[-1], num_classes)
        # 参数0.4为要置为0的元素比例，也可以理解为要丢掉（drop）的元素比例
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(config_concat_ch[-1], num_classes),
        )
        # self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for name in self.stage_names:
            x = getattr(self, name)(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        # x = self.softmax(x)
        return x


def _vovnet(arch,
            config_stage_ch,
            config_concat_ch,
            block_per_stage,
            layer_per_block,
            pretrained,
            progress,
            **kwargs):
    model = VoVNet(config_stage_ch, config_concat_ch,
                   block_per_stage, layer_per_block,
                   **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def vovnet57(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-57 model as described in 
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet('vovnet57', [128, 160, 192, 224], [256, 512, 768, 1024],
                   [1, 1, 4, 3], 5, pretrained, progress, **kwargs)


def vovnet39(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet('vovnet39', [128, 160, 192, 224], [256, 512, 768, 1024],
                   [1, 1, 2, 2], 5, pretrained, progress, **kwargs)


def vovnet_se_simple(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # return _vovnet('vovnet39', [64, 96, 128, 192], [64, 128, 192, 256],
    #                [1,1,1,2], 3, pretrained, progress, **kwargs)
    return _vovnet('vovnet39', [128, 160, 192, 224], [256, 512, 768, 1024],
                   [1, 1, 2, 2], 3, pretrained, progress, **kwargs)


def vovnet_se_slim(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet('vovnet_slim', [32, 40, 48, 56], [64, 128, 192, 256],
                   [1, 1, 1, 1], 4, pretrained, progress, **kwargs)


def vovnet_se_slice(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet('vovnet_slim', [32, 64, 96, 128], [64, 128, 256, 512],
                   [1, 1, 1, 1], 3, False, progress, **kwargs)


def vovnet_se_slice_bak(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet('vovnet_slim', [32, 40, 48, 56], [64, 96, 128, 160],
                   [1, 1, 1, 1], 3, False, progress, **kwargs)


#from torchsummary import torchsummary

#if __name__ == "__main__":
#    model = vovnet_se_slice()
#    #print(model)
#    # if you have GPU, You can use the following code to show this net
#    device = torch.device('cuda:0')
#    model.to(device=device)
#    torchsummary.summary(model, input_size=(3, 160, 160))
