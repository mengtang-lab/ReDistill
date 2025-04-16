import argparse
import torch

from copy import deepcopy

from base.backbones import Network
from reed import ReED


from peak_memory import PeakMemory
from latency import Latency
from flops import FLOPs
from model_size import ModelSize



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
    parser.add_argument('-d', '--dist_config', default='', type=str, help='distillation config yaml file')
    parser.add_argument('--im_size', default=128, type=int, help='image size')
    parser.add_argument('--num_images', default=100, type=int, help='number of images')
    parser.add_argument('--latency_repetitions', default=300, type=int, help='Latency measurement repetitions')
    parser.add_argument('--device', default='cuda:0', type=str, help='cuda:x | cpu')

    args = parser.parse_args()

    student_name        = args.student
    teacher_name        = args.teacher
    dist_config         = args.dist_config
    im_size             = args.im_size
    num_images          = args.num_images
    latency_repetitions = args.latency_repetitions
    device         = args.device

    def get_model_type(name):
        if "mobilenetv3_small" in name: return "stl10-mobilenetv3_small"
        elif "mobilenetv2" in name: return "stl10-mobilenetv2"
        elif "resnext" in name: return "stl10-resnext"
        else: raise NotImplementedError

    dummy_inputs = [torch.randn(num_images, 3, im_size, im_size)]

    teacher = Network(teacher_name, image_size=im_size, num_classes=10)

    teacher_kwargs = {
        "model_type": get_model_type(teacher_name),
        "model": teacher,
        "distill": None,
        "inputs": dummy_inputs,
        "im_size": (im_size, im_size),
        "latency_repetitions": latency_repetitions,
        "device": device,
        "dtype": "fp32"
    }
    model_info_log(info_name=f"model: {teacher_name}", **teacher_kwargs)

    # inc_teacher_kwargs = teacher_kwargs.copy()
    # inc_teacher_kwargs["dtype"] = "int8"
    # inc_teacher_kwargs["model"] = teacher.cpu()
    # model_info_log(info_name=f"model: {teacher_name} - INC", **inc_teacher_kwargs)



    student = Network(student_name, image_size=im_size, num_classes=10)
    student_kwargs = {
        "model_type": get_model_type(student_name),
        "model": student,
        "distill": None,
        "inputs": dummy_inputs,
        "im_size": (im_size, im_size),
        "latency_repetitions": latency_repetitions,
        "device": device,
        "dtype": "fp32"
    }
    model_info_log(info_name=f"model: {student_name}", **student_kwargs)


    model = ReED(student.cpu(), [teacher.cpu()], dist_config=dist_config, dummy_input=dummy_inputs[0].cpu())
    model.graduate()

    distill_kwargs = {
        "model_type": get_model_type(student_name),
        "model": model,
        "distill": "reed",
        "inputs": dummy_inputs,
        "im_size": (im_size, im_size),
        "latency_repetitions": latency_repetitions,
        "device": device,
        "dtype": "fp32"
    }
    model_info_log(info_name=f"distilled_model: {student_name} from {teacher_name}", **distill_kwargs)