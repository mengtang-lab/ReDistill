import re
import json

import torch
import torch.nn as nn

from .mobilenetv2 import *
from .mobilenetv3 import *
from .resnext import *

def name_parse(name):
    name_list = name.split("-")
    assert len(name_list) == 2 or len(name_list) == 3
    arch_name = name_list[0]
    aggressive_stride = [name_list[1]]
    if len(name_list) == 3:
        regular_stride = list(name_list[2])
    else:
        regular_stride = []
    pool_strides = list(map(int, aggressive_stride + regular_stride))
    return arch_name, pool_strides


class Network(nn.Module):
    def __init__(self, name, num_classes, image_size=None, **kwargs):
        arch, pool_strides = name_parse(name)
        arch_dict = {
            'mobilenetv2': mobilenetv2,
            'mobilenetv3_small': mobilenetv3_small,
            'mobilenetv3_large': mobilenetv3_large,
            'resnext18': resnext18,
            'resnext34': resnext34,
            'resnext50': resnext50,
            'resnext101': resnext101,
            'resnext152': resnext152,
            # 'deit_b': deit_b,
            # 'vit_b': vit_b,
            # 'vit_l': vit_l,
            # 'vit_h': vit_h,
            # 'swin_mlp': SwinMLP,
            # 'swin_transformer_v2': SwinTransformerV2,
        }
        super(Network, self).__init__()
        # ========== backbone =================
        if arch in arch_dict.keys():
            if arch in ['swin_mlp', 'swin_transformer_v2']:
                win_base = 32
                assert image_size is not None and image_size % win_base == 0
                self.net = arch_dict[arch](num_classes=num_classes, img_size=image_size, window_size=image_size//win_base, **kwargs)
            elif arch in ['deit_b', 'vit_b', 'vit_l', 'vit_h']:
                assert image_size is not None and len(pool_strides) == 1
                self.net = arch_dict[arch](patch_size=pool_strides[0], num_classes=num_classes, image_size=image_size, **kwargs)
            else:
                self.net = arch_dict[arch](pool_strides=pool_strides, num_classes=num_classes, **kwargs)
        else:
            raise Exception("Undefined Backbone Type!")

    def get_feat_modules(self):
        return self.net.get_feat_modules()

    def forward(self, x, **kwargs):
        return self.net(x, **kwargs)


if __name__ == "__main__":
    model = Network("mobilenetv2-2-1222111", 1000)
    # model = Network("mobilenetv3_small-2-22121111211", 1000)
    # model = Network("mobilenetv3_large-2-121211211111211", 1000)
    # model = Network("resnet18-2-21222", 1000)
    # model = Network("resnext18-2-21222", 1000)
    # model = Network("deit_b-16", 1000, image_size=224)
    # model = Network("vit_l-16", 1000, image_size=224)
    print(model)