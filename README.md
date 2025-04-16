### ImageNet Experiments
- please edit .sh file to enable multi-gpu training
- change the last argument (config file) to switch distillation method
```shell
> cd pipeline_classification/imagenet/

> bash run_distill.sh dist /mnt/data/imagenet2012/ configs/imagenet/resnet50/fitnet.yaml
> bash run_distill.sh dist /mnt/data/imagenet2012/ configs/imagenet/resnet50/reviewkd.yaml
> bash run_distill.sh dist /mnt/data/imagenet2012/ configs/imagenet/resnet50/rkd.yaml

> bash run_distill.sh dist /mnt/data/imagenet2012/ configs/imagenet/resnet18/fitnet.yaml
> bash run_distill.sh dist /mnt/data/imagenet2012/ configs/imagenet/resnet18/reviewkd.yaml
> bash run_distill.sh dist /mnt/data/imagenet2012/ configs/imagenet/resnet18/rkd.yaml

> bash run_distill.sh dist /mnt/data/imagenet2012/ configs/imagenet/mobilenetv2/fitnet.yaml
> bash run_distill.sh dist /mnt/data/imagenet2012/ configs/imagenet/mobilenetv2/reviewkd.yaml
> bash run_distill.sh dist /mnt/data/imagenet2012/ configs/imagenet/mobilenetv2/rkd.yaml
```

### DeiT Experiments
- please edit .sh file to enable multi-gpu training
```shell
> cd pipeline_deit/

# deit
> bash run_deit.sh deit deit_base_distilled_patch16_224 /mnt/data/imagenet2012/ ./output/deit/
# our method
> bash run_deit.sh reed deit_base_distilled_patch16_224 /mnt/data/imagenet2012/ ./output/reed+deit/ configs/deit_base/reed+deit_base-16-alpha_150.yaml 
```

### Efficiency Analysis
- please modify the 'ROOT' variable of the shell script 'analysis/run_sota_analysis.sh' to your root directory

```shell
> cd analysis/
# STL10 dataset
# mobilenetv2
> bash run_sota_analysis.sh stl10 mobilenetv2-1-1222121 mobilenetv2-8-1211111 mobilenetv2_x8-red.yaml 100
# mobilenetv3-small
> bash run_sota_analysis.sh stl10 mobilenetv3_small-1-22121111211 mobilenetv3_small-4-22111111111 mobilenetv3_small-red.yaml 500
# resnext18
> bash run_sota_analysis.sh stl10 resnext18-1-21222 resnext18-4-11221 resnext18-red.yaml 100

# ImageNet dataset
# resnet18
> bash run_sota_analysis.sh imagenet ResNet18 ResNet18_aggressive_pool reed resnet18/reed.yaml resnet18/reed_dist_config.yaml 100
# resnet50
> bash run_sota_analysis.sh imagenet ResNet50 ResNet50_aggressive_pool reed resnet50/reed.yaml resnet50/reed_dist_config.yaml 100
# mobilenetv2
> bash run_sota_analysis.sh imagenet ResNet152 MobileNetV2_aggressive_pool_x2 reed mobilenetv2/reed.yaml mobilenetv2/reed_dist_config.yaml 100

# DDPM
> bash run_sota_analysis.sh ddpm ddpm ddpmx2 reedx2_dist_config_alpha0.1.yaml 200
```
