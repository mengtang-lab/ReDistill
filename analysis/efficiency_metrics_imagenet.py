import argparse
import torch
import torch.nn as nn

from reed import ReED

from pipeline_classification.imagenet.mdistiller.models import cifar_model_dict, imagenet_model_dict
from pipeline_classification.imagenet.mdistiller.distillers import distiller_dict
from pipeline_classification.imagenet.mdistiller.engine.cfg import CFG as cfg

from peak_memory import PeakMemory
from latency import Latency
from flops import FLOPs
from model_size import ModelSize


class Distiller(nn.Module):
    def __init__(self, model):
        super(Distiller, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model.forward_test(x)
    def graduate(self):
        self.model.teacher = nn.ModuleList([])


def model_info_log(info_name="", **kwargs):
    print(f"{'='*10} {info_name} {'='*10}")

    @PeakMemory
    @FLOPs
    @Latency
    @ModelSize
    def __print_log_info(**kwargs):
        pass

    __print_log_info(**kwargs)
    print('='*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Efficiency Metrics',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--student', default='', type=str, help='student network name')
    parser.add_argument('-t', '--teacher', default='', type=str, help='teacher network name')
    parser.add_argument('-d', '--dist_method', default='', type=str, help='distillation method')
    parser.add_argument('--dist_config', default='', type=str, help='distillation config yaml file')
    parser.add_argument('--reed_config', default='', type=str, help='reed distillation config yaml file')
    parser.add_argument('--im_size', default=224, type=int, help='image size')
    parser.add_argument('--num_images', default=100, type=int, help='number of images')
    parser.add_argument('--latency_repetitions', default=300, type=int, help='Latency measurement repetitions')
    parser.add_argument('--device', default='cuda:0', type=str, help='cuda:x | cpu')

    args = parser.parse_args()

    student_name        = args.student
    teacher_name        = args.teacher
    dist_method         = args.dist_method
    dist_config         = args.dist_config
    reed_config         = args.reed_config
    im_size             = args.im_size
    num_images          = args.num_images
    latency_repetitions = args.latency_repetitions
    device         = args.device

    model_dict = imagenet_model_dict
    # im_size = 224
    num_classes = 1000
    teacher = model_dict[teacher_name](pretrained=False, num_classes=num_classes)
    student = model_dict[student_name](pretrained=False, num_classes=num_classes)

    def get_model_type(name):
        if "ResNet" in name: return "imagenet-resnet"
        elif "MobileNetV2" in name: return "imagenet-mobilenetv2"
        else: raise NotImplementedError

    dummy_inputs = torch.randn(num_images, 3, im_size, im_size)

    teacher_kwargs = {
        "model_type": get_model_type(teacher_name),
        "model": teacher,
        "distill": None,
        "inputs": [dummy_inputs],
        "im_size": (im_size, im_size),
        "latency_repetitions": latency_repetitions,
        "device": device,
        "dtype": "fp32"
    }
    model_info_log(info_name=f"model: {teacher_name}", **teacher_kwargs)

    student_kwargs = {
        "model_type": get_model_type(student_name),
        "model": student,
        "distill": None,
        "inputs": [dummy_inputs],
        "im_size": (im_size, im_size),
        "latency_repetitions": latency_repetitions,
        "device": device,
        "dtype": "fp32"
    }
    model_info_log(info_name=f"model: {student_name}", **student_kwargs)

    if dist_method.lower() == "reed":
        model = ReED(student.cpu(), [teacher.cpu()], dist_config=reed_config, dummy_input=dummy_inputs.cpu())
    elif dist_method.lower() == "crd":
        cfg.merge_from_file(dist_config)
        model = Distiller(distiller_dict[dist_method](student.cpu(), teacher.cpu(), cfg, num_data=100))
    else:
        cfg.merge_from_file(dist_config)
        model = Distiller(distiller_dict[dist_method](student.cpu(), teacher.cpu(), cfg))
    print(model)
    model.graduate()
    distill_kwargs = {
        "model_type": get_model_type(student_name),
        "model": model,
        "distill": dist_method.lower(),
        "inputs": [dummy_inputs],
        "im_size": (im_size, im_size),
        "latency_repetitions": latency_repetitions,
        "device": device,
        "dtype": "fp32"
    }
    model_info_log(info_name=f"distilled_model: {student_name} from {teacher_name}", **distill_kwargs)