import argparse
import torch

from backbones import Network
from reed import ReED, FLOPsHook

from latency import Latency
from memory_usage import memory_usage, max_memory

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
    parser.add_argument('-d', '--dist_config', default='', type=str, help='distillation config yaml file')
    parser.add_argument('--im_size', default=224, type=int, help='image size')
    parser.add_argument('--latency_repetitions', default=300, type=int, help='Latency measurement repetitions')
    parser.add_argument('--device', default='cuda:0', type=str, help='cuda:x | cpu')

    args = parser.parse_args()

    student_name        = args.student
    teacher_name        = args.teacher
    dist_config         = args.dist_config
    im_size             = args.im_size
    latency_repetitions = args.latency_repetitions
    device         = args.device

    dummy_inputs = [torch.randn(100, 3, im_size, im_size)]

    teacher = Network(teacher_name, image_size=im_size, num_classes=1000)
    teacher_kwargs = {
        "model": teacher,
        "inputs": dummy_inputs,
        "im_size": (im_size, im_size),
        "latency_repetitions": latency_repetitions,
        "device": device
    }
    model_info_log(info_name=f"model: {teacher_name}", **teacher_kwargs)

    student = Network(student_name, image_size=im_size, num_classes=1000)
    student_kwargs = {
        "model": student,
        "inputs": dummy_inputs,
        "im_size": (im_size, im_size),
        "latency_repetitions": latency_repetitions,
        "device": device
    }
    model_info_log(info_name=f"model: {student_name}", **student_kwargs)


    model = ReED(student.cpu(), [teacher.cpu()], dist_config=dist_config, dummy_input=dummy_inputs[0].cpu())
    model.graduate()

    distill_kwargs = {
        "model": model,
        "inputs": dummy_inputs,
        "im_size": (im_size, im_size),
        "latency_repetitions": latency_repetitions,
        "device": device
    }
    model_info_log(info_name=f"distilled_model: {student_name} from {teacher_name}", **distill_kwargs)