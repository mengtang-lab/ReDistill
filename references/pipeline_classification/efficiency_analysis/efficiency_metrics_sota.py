import argparse
import torch
import torch.nn as nn

from backbones import Network
from reed import ReED, FLOPsHook
from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.engine.cfg import CFG as cfg

from latency import Latency
from memory_usage import memory_usage, max_memory

class Distiller(nn.Module):
    def __init__(self, model):
        super(Distiller, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model.forward_test(x)
    def graduate(self):
        self.model.teacher = nn.ModuleList([])

def ModelSize(func):
    def decorator(**kwargs):
        model = kwargs["model"]
        dummy_input = kwargs["inputs"][0]
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        print('Model Size: {:.3f} MB'.format(size_all_mb))
        return func(**kwargs)
    return decorator

def FLOPs(func):
    def decorator(**kwargs):
        model = kwargs["model"]
        dummy_input = kwargs["inputs"][0]
        flops_hook = FLOPsHook()
        flops_hook.compute_flops(model, dummy_input, show_details = False)
        return func(**kwargs)
    return decorator

def model_info_log(info_name="", **kwargs):
    print(f"{'='*10} {info_name} {'='*10}")

    @FLOPs
    @Latency
    @ModelSize
    # @memory_usage
    # @max_memory
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
    parser.add_argument('--dataset', default='cifar100', type=str, help='dataset')
    parser.add_argument('--latency_repetitions', default=300, type=int, help='Latency measurement repetitions')
    parser.add_argument('--device', default='cuda:0', type=str, help='cuda:x | cpu')

    args = parser.parse_args()

    student_name        = args.student
    teacher_name        = args.teacher
    dist_method         = args.dist_method
    dist_config         = args.dist_config
    dataset             = args.dataset
    latency_repetitions = args.latency_repetitions
    device         = args.device

    if dataset.lower() in ('cifar', 'cifar100', 'stl', 'stl10'):
        model_dict = cifar_model_dict
        im_size = 128
        num_classes = 100
    if dataset.lower() in ('imagenet'):
        model_dict = imagenet_model_dict
        im_size = 224
        num_classes = 1000

    dummy_inputs = torch.randn(100, 3, im_size, im_size)

    teacher = model_dict[teacher_name][0](num_classes=num_classes)
    teacher_kwargs = {
        "model": teacher,
        "inputs": [dummy_inputs],
        "im_size": (im_size, im_size),
        "latency_repetitions": latency_repetitions,
        "device": device
    }
    model_info_log(info_name=f"model: {teacher_name}", **teacher_kwargs)

    student = model_dict[student_name][0](num_classes=num_classes)
    student_kwargs = {
        "model": student,
        "inputs": [dummy_inputs],
        "im_size": (im_size, im_size),
        "latency_repetitions": latency_repetitions,
        "device": device
    }
    model_info_log(info_name=f"model: {student_name}", **student_kwargs)

    if dist_method.lower() == "reed":
        model = ReED(student.cpu(), [teacher.cpu()], dist_config=dist_config, dummy_input=dummy_inputs.cpu())
    elif dist_method.lower() == "crd":
        model = Distiller(distiller_dict[dist_method](student.cpu(), teacher.cpu(), cfg, num_data=100))
    else:
        model = Distiller(distiller_dict[dist_method](student.cpu(), teacher.cpu(), cfg))
    model.graduate()
    distill_kwargs = {
        "model": model,
        "inputs": [dummy_inputs],
        "im_size": (im_size, im_size),
        "latency_repetitions": latency_repetitions,
        "device": device
    }
    model_info_log(info_name=f"distilled_model: {student_name} from {teacher_name}", **distill_kwargs)