from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights,
)

from torchvision.models import (
    mobilenet_v3_small,
    mobilenet_v3_large,

    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
)

from torchvision.models import (
    resnext50_32x4d,
    resnext101_32x8d,
    resnext101_64x4d,

    ResNeXt50_32X4D_Weights,
    ResNeXt101_32X8D_Weights,
    ResNeXt101_64X4D_Weights,
)

from torchvision.models import (
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,

    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
)

from torchvision.models import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,

    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    EfficientNet_B6_Weights,
    EfficientNet_B7_Weights,
)

from torchvision.models import (
    swin_t,
    swin_s,
    swin_b,
    swin_v2_t,
    swin_v2_s,
    swin_v2_b,

    Swin_T_Weights,
    Swin_S_Weights,
    Swin_B_Weights,
    Swin_V2_T_Weights,
    Swin_V2_S_Weights,
    Swin_V2_B_Weights,
)

from torchvision.models import (
    vit_b_16,
    vit_b_32,
    vit_l_16,
    vit_l_32,
    vit_h_14,

    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    ViT_L_32_Weights,
    ViT_H_14_Weights,
)

model_dict = {
        'mobilenet_v2': [mobilenet_v2, MobileNet_V2_Weights],
        'mobilenet_v3_small': [mobilenet_v3_small, MobileNet_V3_Small_Weights],
        'mobilenet_v2_large': [mobilenet_v3_large, MobileNet_V3_Large_Weights],

        'resnext50_32x4d': [resnext50_32x4d, ResNeXt50_32X4D_Weights],
        'resnext101_32x8d': [resnext101_32x8d, ResNeXt101_32X8D_Weights],
        'resnext101_64x4d': [resnext101_64x4d, ResNeXt101_64X4D_Weights],

        'convnext_tiny': [convnext_tiny, ConvNeXt_Tiny_Weights],
        'convnext_small': [convnext_small, ConvNeXt_Small_Weights],
        'convnext_base': [convnext_base, ConvNeXt_Base_Weights],
        'convnext_large': [convnext_large, ConvNeXt_Large_Weights],

        'efficientnet_b0': [efficientnet_b0, EfficientNet_B0_Weights],
        'efficientnet_b1': [efficientnet_b1, EfficientNet_B1_Weights],
        'efficientnet_b2': [efficientnet_b2, EfficientNet_B2_Weights],
        'efficientnet_b3': [efficientnet_b3, EfficientNet_B3_Weights],
        'efficientnet_b4': [efficientnet_b4, EfficientNet_B4_Weights],
        'efficientnet_b5': [efficientnet_b5, EfficientNet_B5_Weights],
        'efficientnet_b6': [efficientnet_b6, EfficientNet_B6_Weights],
        'efficientnet_b7': [efficientnet_b7, EfficientNet_B7_Weights],

        'swin_t': [swin_t, Swin_T_Weights],
        'swin_s': [swin_s, Swin_S_Weights],
        'swin_b': [swin_b, Swin_B_Weights],
        'swin_v2_t': [swin_t, Swin_T_Weights],
        'swin_v2_s': [swin_v2_s, Swin_V2_S_Weights],
        'swin_v2_b': [swin_v2_b, Swin_V2_B_Weights],

        'vit_b_16': [vit_b_16, ViT_B_16_Weights],
        'vit_b_32': [vit_b_32, ViT_B_32_Weights],
        'vit_l_16': [vit_l_16, ViT_L_16_Weights],
        'vit_l_32': [vit_l_32, ViT_L_32_Weights],
        'vit_h_14': [vit_h_14, ViT_H_14_Weights],
    }

dont_require_image_size = [
    'mobilenet_v2',
    'mobilenet_v3_small',
    'mobilenet_v3_large',
    'resnext50_32x4d',
    'resnext101_32x8d',
    'resnext101_64x4d',
    'convnext_tiny',
    'convnext_small',
    'convnext_base',
    'convnext_large',
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2',
    'efficientnet_b3',
    'efficientnet_b4',
    'efficientnet_b5',
    'efficientnet_b6',
    'efficientnet_b7',
]

def build_model(model_type, is_pretrained=True, **kwargs):
    if model_type not in model_dict.keys(): raise NotImplementedError(f"Unkown model: {model_type}")
    if model_type in dont_require_image_size: kwargs.pop("image_size")
    model, pretrained_weight = model_dict[model_type]
    if is_pretrained:
        # return model(weights = pretrained_weight.IMAGENET1K_V1, **kwargs)
        return model(weights=pretrained_weight.DEFAULT, **kwargs)
    return model(weights = None, **kwargs)