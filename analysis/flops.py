import torch
import torch.nn as nn

from reed import HookHelper


class FLOPsHook(HookHelper):
    def __init__(self):
        super(FLOPsHook, self).__init__()

    @staticmethod
    def flops_hook(name: str, info_dict: dict):
        def hook(module, input, output):
            flops = 0
            # activation_memory = torch.numel(input[0])*input[0].element_size()
            if isinstance(module, nn.Conv2d):
                b               = input[0].shape[0]
                c_in            = input[0].shape[1]
                # h_in, w_in      = input[0].shape[-2:]
                c_out           = output.shape[1]
                h_out, w_out    = output.shape[-2:]
                if isinstance(module.kernel_size, int): k_h = k_w = module.kernel_size
                else: k_h, k_w = module.kernel_size

                flops           = b * (2 * h_out * w_out * (c_in * k_h * k_w) * c_out)

            if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                b = input[0].shape[0]
                c_in = input[0].shape[1]
                h_out, w_out = output.shape[-2:]
                if isinstance(module.kernel_size, int): k_h = k_w = module.kernel_size
                else: k_h, k_w = module.kernel_size

                flops = b * (h_out * w_out * (c_in * k_h * k_w))

            if isinstance(module, (nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d)):
                b = input[0].shape[0]
                c_in = input[0].shape[1]
                h_in, w_in = input[0].shape[-2:]

                flops = b * (h_in * w_in * c_in)

            if isinstance(module, nn.Linear):
                assert len(input[0].shape) in (2, 3)
                n               = input[0].shape[0]
                d_in            = input[0].shape[-1]
                d_out           = output.shape[-1]
                if len(input[0].shape) == 3:
                    n           = n * input[0].shape[1]

                flops           = n * (2 * d_in * d_out)

            info_dict[name] = {"module": module, "flops": flops}
        return hook

    @staticmethod
    def __print_info_table(info_table: dict, show_details: bool = False):
        FLOPs = []
        for name, value in info_table.items():
            module, flops = value["module"], value["flops"]
            if show_details and flops != 0:
                print("{}, {}, FLOPs: {}".format(name, type(module), flops))
            FLOPs.append(flops)
        print("FLOPs: {} G".format(sum(FLOPs) / 1024 ** 3))

    @staticmethod
    def compute_flops(model, dummy_input:torch.Tensor = None, show_details: bool = False):
        info_table = {}
        for name, module in model.named_modules():
            module.register_forward_hook(FLOPsHook.flops_hook(name, info_table))
        if dummy_input is not None: FLOPsHook.calibrate(model, dummy_input)
        FLOPsHook.__print_info_table(info_table, show_details)


def FLOPs(func):
    def decorator(**kwargs):
        model = kwargs["model"]
        dummy_input = kwargs["inputs"][0]
        FLOPsHook.compute_flops(model, dummy_input, show_details = False)
        return func(**kwargs)
    return decorator

def FLOPsDDPM(func):
    def decorator(**kwargs):
        model = kwargs["model"]
        dummy_input = {'x': kwargs["inputs"][0], 't': kwargs["inputs"][1]}
        FLOPsHook.compute_flops(model, dummy_input, show_details = False)
        return func(**kwargs)
    return decorator