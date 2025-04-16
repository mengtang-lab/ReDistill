import re
import json

import torch
import torch.nn as nn

from .mobilenetv2 import *
from .mobilenetv3 import *
from .resnet import *
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
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152,
            'resnext18': resnext18,
            'resnext34': resnext34,
            'resnext50': resnext50,
            'resnext101': resnext101,
            'resnext152': resnext152,
        }
        super(Network, self).__init__()
        # ========== backbone =================
        if arch in arch_dict.keys():
            self.net = arch_dict[arch](pool_strides=pool_strides, num_classes=num_classes, **kwargs)
        else:
            raise Exception("Undefined Backbone Type!")

    def forward(self, x, **kwargs):
        return self.net(x, **kwargs)


if __name__ == "__main__":
    model = Network("mobilenetv2-2-1222111", 1000)
    # model = Network("mobilenetv3_small-2-22121111211", 1000)
    # model = Network("mobilenetv3_large-2-121211211111211", 1000)
    # model = Network("resnet18-2-21222", 1000)
    # model = Network("resnext18-2-21222", 1000)
    print(model)