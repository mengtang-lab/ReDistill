import argparse
import torch

from pipeline_diffusion.model import UNet
from reed import ReED_Diffuser


from peak_memory import PeakMemoryDDPM
from latency import Latency
from flops import FLOPsDDPM
from model_size import ModelSize



def model_info_log(info_name="", **kwargs):
    print(f"{'='*10} {info_name} {'='*10}")

    @PeakMemoryDDPM
    @FLOPsDDPM
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
    parser.add_argument('--im_size', default=32, type=int, help='image size')
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

    dummy_inputs = [torch.randn(num_images, 3, im_size, im_size), torch.randint(1000, size=(num_images, ))]

    teacher = UNet(
        T=1000, ch=128, ch_mult=[1,2,2,2], attn=[1],
        num_res_blocks=2, dropout=0.1, pool_strides=[1,2,2,2])
    teacher_kwargs = {
        "model_type": "ddpm",
        "model": teacher,
        "distill": None,
        "inputs": dummy_inputs,
        "im_size": (im_size, im_size),
        "latency_repetitions": latency_repetitions,
        "device": device
    }
    model_info_log(info_name="model: teacher-ddpm", **teacher_kwargs)

    student = UNet(
        T=1000, ch=128, ch_mult=[1,2,2,2], attn=[1],
        num_res_blocks=2, dropout=0.1, pool_strides=[2,2,2,1])
    student_kwargs = {
        "model_type": "ddpm",
        "model": student,
        "distill": None,
        "inputs": dummy_inputs,
        "im_size": (im_size, im_size),
        "latency_repetitions": latency_repetitions,
        "device": device
    }
    model_info_log(info_name="model: student-ddpmx2", **student_kwargs)


    model = ReED_Diffuser(student.cpu(), [teacher.cpu()], dist_config, dummy_input={'x': torch.randn(1, 3, im_size, im_size),
                                                                        't': torch.randint(1000, size=(1, ))})
    # print(model)
    model.graduate()

    distill_kwargs = {
        "model_type": "ddpm",
        "model": model,
        "distill": "reed",
        "inputs": dummy_inputs,
        "im_size": (im_size, im_size),
        "latency_repetitions": latency_repetitions,
        "device": device
    }
    model_info_log(info_name="distilled_model: ddpmx2 from ddpm", **distill_kwargs)