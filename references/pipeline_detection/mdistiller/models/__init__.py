from .resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnet18_aggressive_pool,
    resnet34_aggressive_pool,
    resnet50_aggressive_pool,
    resnet101_aggressive_pool,
    resnet152_aggressive_pool,
)
from .mobilenetv2 import mobilenetv2, mobilenetv2_aggressive_pool_x2, mobilenetv2_aggressive_pool_x4


model_dict = {
    "ResNet18": resnet18,
    "ResNet34": resnet34,
    "ResNet50": resnet50,
    "ResNet101": resnet101,
    "ResNet152": resnet152,
    "MobileNetV2": mobilenetv2,


    "ResNet18_aggressive_pool": resnet18_aggressive_pool,
    "ResNet34_aggressive_pool": resnet34_aggressive_pool,
    "ResNet50_aggressive_pool": resnet50_aggressive_pool,
    "ResNet101_aggressive_pool": resnet101_aggressive_pool,
    "ResNet152_aggressive_pool": resnet152_aggressive_pool,
    "MobileNetV2_aggressive_pool_x2": mobilenetv2_aggressive_pool_x2,
    "MobileNetV2_aggressive_pool_x4": mobilenetv2_aggressive_pool_x4,
}
