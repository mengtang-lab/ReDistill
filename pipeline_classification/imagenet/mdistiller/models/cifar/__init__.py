import os
from .resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,

    resnet18_aggressive_pool,
    resnet34_aggressive_pool,
    resnet50_aggressive_pool,
    resnet101_aggressive_pool
)
from .mobilenetv2 import mobile_half, mobile_half_aggressive_pool


cifar100_model_prefix = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "../../../download_ckpts/cifar100_imsize128_teachers/"
)

cifar_model_dict = {
    # teachers
    "resnet18": (
        resnet18,
        cifar100_model_prefix + "resnet18_cifar100/resnet18_cifar100_best.pth",
    ),
    "resnet50": (
        resnet50,
        cifar100_model_prefix + "resnet50_cifar100/resnet50_cifar100_best.pth",
    ),
    "mobilenetv2": (
        mobile_half,
        cifar100_model_prefix + "mobilenetv2_cifar100/mobilenetv2_cifar100_best.pth",
    ),
    # # students
    "resnet18_aggressive_pool": (resnet18_aggressive_pool, None),
    "resnet50_aggressive_pool": (resnet50_aggressive_pool, None),
    "mobilenetv2_aggressive_pool": (mobile_half_aggressive_pool, None),
}
