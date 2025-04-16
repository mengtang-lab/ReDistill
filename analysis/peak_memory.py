import numpy as np
import torch
import torch.nn as nn

from reed import ReED, HookHelper
from base.backbones.mobilenetv3 import InvertedResidual as MobileNetV3_InvertedResidual, SELayer as MobileNetV3_SELayer
from base.backbones.mobilenetv2 import InvertedResidual as MobileNetV2_InvertedResidual
from pipeline_diffusion.model import TimeEmbedding, ResBlock, DownSample, UpSample

FP32 = False
INT8 = False


def record_in_out_mem(m, x, y):
    if len(x) == 0: return None
    x = x[0]
    if type(y) == tuple: y = y[0]
    m.input_shape = list(x.shape)
    m.input_elem_mem = x.element_size()
    m.output_shape = list(y.shape)
    m.output_elem_mem = y.element_size()

    # m.input_mem = torch.numel(x) * x.element_size()
    # m.output_mem = torch.numel(y) * y.element_size()

    if FP32:
        m.input_mem = np.prod(m.input_shape[1:]).item() * x.element_size()
        m.output_mem = np.prod(m.output_shape[1:]).item() * y.element_size()

    if INT8:
        m.input_mem = np.prod(m.input_shape[1:]).item()
        m.output_mem = np.prod(m.output_shape[1:]).item()


def add_io_hooks(m_):
    m_.register_forward_hook(record_in_out_mem)

def mem2mb(n):
    return n / 1024**2

def estimate_conv(m, INPLACE_DW=False):
    assert isinstance(m, nn.Conv2d), type(m)
    if INPLACE_DW and m.in_channels == m.out_channels == m.groups:
        return m.input_mem + np.prod(m.output_shape[2:]) * m.output_elem_mem  # add a buffer with size equal to one channel
    else:  # normal conv
        return m.input_mem + m.output_mem
    # if INPLACE_DW and m.in_channels == m.out_channels == m.groups:
    #     return np.prod(m.output_shape[2:]) * m.output_elem_mem  # add a buffer with size equal to one channel
    # else:  # normal conv
    #     return m.output_mem


class PeakMemoryHook(HookHelper):
    def __init__(self):
        super(PeakMemoryHook, self).__init__()

    def estimate_layer(self, module, input, output):
        raise NotImplementedError

    def print_info_table(self, info_table: dict, show_details: bool = False):
        ACT_Mems = []
        for name, value in info_table.items():
            module, act_mem = value["module"], value["act_mem"]
            if show_details and act_mem != 0:
                print("{}, {}, Memory: {} MB".format(name, type(module), mem2mb(act_mem)))
            ACT_Mems.append(act_mem)
        print("Memory List: ", [mem2mb(m) for m in ACT_Mems])
        print("Peak Memory: {} MB".format(mem2mb(max(ACT_Mems))))

    def compute_peak_memory(self, model, dummy_input:torch.Tensor = None, show_details: bool = False):
        raise NotImplementedError


class TorchVisionMobileNetV2_PeakMemHook(PeakMemoryHook):
    def __init__(self):
        super(TorchVisionMobileNetV2_PeakMemHook, self).__init__()

    def estimate_layer(self, module):
        if hasattr(module, "conv"):
            print(module)
            assert len(module.conv) in (3, 4)
            if module.use_res_connect:
                if len(module.conv) == 3:
                    return max([
                        estimate_conv(module.conv[0][0]),
                        estimate_conv(module.conv[1]),
                    ])
                else:
                    return max([
                        estimate_conv(module.conv[0][0]),
                        estimate_conv(module.conv[1][0]) + module.input_mem,
                        estimate_conv(module.conv[2])
                    ])
            else:
                if len(module.conv) == 3:
                    return max([
                        estimate_conv(module.conv[0][0]),
                        estimate_conv(module.conv[1])
                    ])
                else:
                    return max([
                        estimate_conv(module.conv[0][0]),
                        estimate_conv(module.conv[1][0]),
                        estimate_conv(module.conv[2])
                    ])
        else:
            return estimate_conv(module[0])

    def compute_peak_memory(self, model, dummy_input:torch.Tensor = None, show_details: bool = False):
        model = model.features
        model.eval()
        model.apply(add_io_hooks)
        if dummy_input is not None: self.calibrate(model, dummy_input)

        info_table = {}
        for name, module in model.named_children():
            act_mem = self.estimate_layer(module)
            info_table[name] = {"module": module, "act_mem": act_mem}

        self.print_info_table(info_table, show_details)


class ImageNet_MobileNetV2_PeakMemHook(PeakMemoryHook):
    def __init__(self):
        super(ImageNet_MobileNetV2_PeakMemHook, self).__init__()

    def estimate_layer(self, module):
        if isinstance(module, nn.Sequential):
            act_mems = []
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    act_mems.append(estimate_conv(layer))
            return max(act_mems)
        elif isinstance(module, nn.AvgPool2d):
            return module.input_mem + module.output_mem
        else:
            raise NotImplementedError

    def compute_peak_memory(self, model, dummy_input:torch.Tensor = None, show_details: bool = False):
        model.eval()
        model.apply(add_io_hooks)
        if dummy_input is not None: self.calibrate(model, dummy_input)

        if isinstance(model, ReED):
            model = model.student.model
        else:
            model = model.model

        info_table = {}
        for name, module in model.named_children():
            act_mem = self.estimate_layer(module)
            info_table[name] = {"module": module, "act_mem": act_mem}
        self.print_info_table(info_table, show_details)


class ImageNet_ResNet_PeakMemHook(PeakMemoryHook):
    def __init__(self):
        super(ImageNet_ResNet_PeakMemHook, self).__init__()

    def estimate_layer(self, module):
        if isinstance(module, nn.Sequential):
            layer_mems = []
            for layer in module:
                convs = [layer.conv1, layer.conv2]
                if hasattr(layer, 'conv3'):
                    convs.append(layer.conv3)
                if layer.downsample is not None:
                    res_mem = estimate_conv(layer.downsample[0])
                else:
                    res_mem = module.input_mem
                layer_mems.append(max([estimate_conv(c) for c in convs]) + res_mem)
            print([mem2mb(m) for m in layer_mems])
            return max(layer_mems)
        elif isinstance(module, nn.Conv2d):
            return estimate_conv(module)
        elif isinstance(module, nn.MaxPool2d):
            return module.input_mem + module.output_mem
        else:
            raise NotImplementedError

    def compute_peak_memory(self, model, dummy_input:torch.Tensor = None, show_details: bool = False):
        model.eval()
        model.apply(add_io_hooks)
        if dummy_input is not None: self.calibrate(model, dummy_input)

        if isinstance(model, ReED):
            model = model.student
        else:
            model = model

        info_table = {}
        layers = {"conv1": model.conv1,
                  "maxpool": model.maxpool,
                  "layer1": model.layer1,
                  "layer2": model.layer2,
                  "layer3": model.layer3,
                  "layer4": model.layer4}
        for name, module in layers.items():
            act_mem = self.estimate_layer(module)
            info_table[name] = {"module": module, "act_mem": act_mem}
        self.print_info_table(info_table, show_details)


class STL10_MobileNetV3Small_PeakMemHook(PeakMemoryHook):
    def __init__(self):
        super(STL10_MobileNetV3Small_PeakMemHook, self).__init__()

    def estimate_layer(self, module):
        if isinstance(module, MobileNetV3_InvertedResidual):
            layer_mems = []
            for layer in module.conv:
                if isinstance(layer, nn.Conv2d):
                    layer_mems.append(estimate_conv(layer))
                elif isinstance(layer, MobileNetV3_SELayer):
                    avg_mem = layer.avg_pool.input_mem + layer.avg_pool.output_mem
                    fc1_mem = layer.fc[0].input_mem + layer.fc[0].output_mem
                    fc2_mem = layer.fc[1].input_mem + layer.fc[1].output_mem
                    layer_mems.append(max([avg_mem, fc1_mem, fc2_mem]) + layer.input_mem)
            return max(layer_mems) + module.input_mem if module.identity else max(layer_mems)
        elif isinstance(module, nn.Sequential):
            return estimate_conv(module[0])
        else:
            raise NotImplementedError

    def compute_peak_memory(self, model, dummy_input:torch.Tensor = None, show_details: bool = False):
        model.eval()
        model.apply(add_io_hooks)
        if dummy_input is not None: self.calibrate(model, dummy_input)

        if isinstance(model, ReED):
            model = model.student.net.features
        else:
            model = model.net.features

        info_table = {}
        for name, module in model.named_children():
            act_mem = self.estimate_layer(module)
            info_table[name] = {"module": module, "act_mem": act_mem}
        self.print_info_table(info_table, show_details)

class STL10_MobileNetV2_PeakMemHook(PeakMemoryHook):
    def __init__(self):
        super(STL10_MobileNetV2_PeakMemHook, self).__init__()

    def estimate_layer(self, module):
        if isinstance(module, MobileNetV2_InvertedResidual):
            layer_mems = []
            for layer in module.conv:
                if isinstance(layer, nn.Conv2d):
                    layer_mems.append(estimate_conv(layer))
            return max(layer_mems) + module.input_mem if module.identity else max(layer_mems)
        elif isinstance(module, nn.Sequential):
            return estimate_conv(module[0])
        else:
            raise NotImplementedError

    def compute_peak_memory(self, model, dummy_input:torch.Tensor = None, show_details: bool = False):
        model.eval()
        model.apply(add_io_hooks)
        if dummy_input is not None: self.calibrate(model, dummy_input)

        if isinstance(model, ReED):
            model = model.student.net.features
        else:
            model = model.net.features

        info_table = {}
        for name, module in model.named_children():
            act_mem = self.estimate_layer(module)
            info_table[name] = {"module": module, "act_mem": act_mem}
        self.print_info_table(info_table, show_details)


class STL10_ResNext_PeakMemHook(PeakMemoryHook):
    def __init__(self):
        super(STL10_ResNext_PeakMemHook, self).__init__()

    def estimate_layer(self, module):
        if isinstance(module, nn.Sequential):
            layer_mems = []
            for layer in module:
                convs = [layer.conv1, layer.conv2]
                if hasattr(layer, "conv3"):
                    convs.append(layer.conv3)
                if layer.downsample is not None:
                    res_mem = estimate_conv(layer.downsample[0])
                else:
                    res_mem = module.input_mem
                layer_mems.append(max([estimate_conv(c) for c in convs]) + res_mem)
            print([mem2mb(m) for m in layer_mems])
            return max(layer_mems)
        elif isinstance(module, nn.Conv2d):
            return estimate_conv(module)
        elif isinstance(module, nn.MaxPool2d):
            return module.input_mem + module.output_mem
        else:
            raise NotImplementedError

    def compute_peak_memory(self, model, dummy_input:torch.Tensor = None, show_details: bool = False):
        model.eval()
        model.apply(add_io_hooks)
        if dummy_input is not None: self.calibrate(model, dummy_input)

        if isinstance(model, ReED):
            model = model.student.net
        else:
            model = model.net

        info_table = {}
        layers = {"conv1": model.conv1,
                  "maxpool": model.maxpool,
                  "layer1": model.layer1,
                  "layer2": model.layer2,
                  "layer3": model.layer3,
                  "layer4": model.layer4}
        for name, module in layers.items():
            act_mem = self.estimate_layer(module)
            info_table[name] = {"module": module, "act_mem": act_mem}
        self.print_info_table(info_table, show_details)


class DDPM_PeakMemHook(PeakMemoryHook):
    def __init__(self):
        super(DDPM_PeakMemHook, self).__init__()

    def estimate_layer(self, module):
        if isinstance(module, ResBlock):
            block1, temb_proj, block2 = module.block1, module.temb_proj, module.block2
            return max([
                        estimate_conv(block1[2]) + temb_proj[1].output_mem,
                        estimate_conv(block2[3]),
                    ]) + module.input_mem
        if isinstance(module, (DownSample, UpSample)):
            return module.output_mem

    def compute_peak_memory(self, model, dummy_input:torch.Tensor = None, show_details: bool = False):
        model.eval()
        model.apply(add_io_hooks)
        if dummy_input is not None: self.calibrate(model, dummy_input)

        if isinstance(model, ReED):
            model = model.student
        else:
            model = model

        info_table = {}
        for name, module in model.named_children():
            # print(name, module)
            res_mem = []
            if name == "time_embedding":
                act_mem = max([module.timembedding[1].output_mem, module.timembedding[3].output_mem])
                print(f"{name}-{mem2mb(act_mem)}")
            if name == "head":
                act_mem = estimate_conv(module)
                res_mem.append(module.output_mem)
                print(f"{name}-{mem2mb(act_mem)}")
            if name == "downblocks":
                act_mem = []
                for i in range(len(module)):
                    block_mem = self.estimate_layer(module[i])
                    act_mem.append(block_mem + sum(res_mem))
                    res_mem.append(module[i].output_mem)
                print(f"{name}-{[mem2mb(m) for m in act_mem]}")
                act_mem = max(act_mem)
            if name == "middleblocks":
                act_mem = []
                for i in range(len(module)):
                    act_mem.append(self.estimate_layer(module[i]))
                print(f"{name}-{[mem2mb(m) for m in act_mem]}")
                act_mem = max(act_mem) + sum(res_mem)
            if name == "upblocks":
                act_mem = []
                for i in range(len(module)):
                    res_mem = res_mem[:-1]
                    block_mem = self.estimate_layer(module[i])
                    act_mem.append(block_mem + sum(res_mem))
                print(f"{name}-{[mem2mb(m) for m in act_mem]}")
                act_mem = max(act_mem)
            if name == "tail":
                act_mem = module.output_mem
                print(f"{name}-{mem2mb(act_mem)}")
            info_table[name] = {"module": module, "act_mem": act_mem}
        self.print_info_table(info_table, show_details)

class ReEDPeakMemory(PeakMemoryHook):
    def __init__(self):
        super(ReEDPeakMemory, self).__init__()

    def print_info_table(self, info_table: dict, show_details: bool = False):
        ACT_Mems = []
        for name, value in info_table.items():
            module, act_mem = value["module"], value["act_mem"]
            if show_details and act_mem != 0:
                print("{}, {}, Memory: {} MB".format(name, type(module), mem2mb(act_mem)))
            ACT_Mems.append(act_mem)
        print("ReED blocks' Peak Memory: {} MB".format(mem2mb(max(ACT_Mems))))

    def estimate_layer(self, module):
        logit_mem = estimate_conv(module.logit[0]) + module.input_mem
        res_enc_mem = estimate_conv(module.residual_encoder[0])
        return max([logit_mem, res_enc_mem])

    def compute_peak_memory(self, model, dummy_input:torch.Tensor = None, show_details: bool = False):
        model.eval()
        model.apply(add_io_hooks)
        if dummy_input is not None: self.calibrate(model, dummy_input)

        if isinstance(model, ReED):
            model = model.dist_modules
        else:
            print("Not a ReED model!")
            return

        info_table = {}
        for name, module in model.named_children():
            act_mem = self.estimate_layer(module)
            info_table[name] = {"module": module, "act_mem": act_mem}
        self.print_info_table(info_table, show_details)

HookDict = {
    "torchvision-mobilenetv2": TorchVisionMobileNetV2_PeakMemHook,
    "imagenet-mobilenetv2": ImageNet_MobileNetV2_PeakMemHook,
    "imagenet-resnet": ImageNet_ResNet_PeakMemHook,
    "stl10-mobilenetv3_small": STL10_MobileNetV3Small_PeakMemHook,
    "stl10-mobilenetv2": STL10_MobileNetV2_PeakMemHook,
    "stl10-resnext": STL10_ResNext_PeakMemHook,
    "ddpm": DDPM_PeakMemHook,
}


def PeakMemory(func):
    def decorator(**kwargs):
        distill = kwargs["distill"]
        model = kwargs["model"]
        model_type = kwargs["model_type"]
        dummy_input = kwargs["inputs"][0]
        dtype = kwargs["dtype"]
        global FP32, INT8
        if dtype == "fp32":  FP32 = True
        elif dtype == "int8": INT8 = True
        else:
            raise NotImplementedError
        peak_mem_hook = HookDict[model_type]()
        if distill == None or distill == "reed": model = model
        else: model = model.model
        peak_mem_hook.compute_peak_memory(model, dummy_input, show_details = True)
        ReEDPeakMemory().compute_peak_memory(model, dummy_input, show_details = True)
        return func(**kwargs)
    return decorator

def PeakMemoryDDPM(func):
    def decorator(**kwargs):
        distill = kwargs["distill"]
        model = kwargs["model"]
        model_type = kwargs["model_type"]
        dummy_input = {'x': kwargs["inputs"][0], 't': kwargs["inputs"][1]}
        peak_mem_hook = HookDict[model_type]()
        peak_mem_hook.compute_peak_memory(model, dummy_input, show_details = True)
        if distill == "reed":
            ReEDPeakMemory().compute_peak_memory(model, dummy_input, show_details = True)
        return func(**kwargs)
    return decorator

if __name__ == "__main__":
    import torchvision
    net = torchvision.models.mobilenet_v2()
    sample_input = torch.randn(1, 3, 224, 224)
    peak_memory_hook = TorchVisionMobileNetV2_PeakMemHook()
    peak_memory_hook.compute_peak_memory(net, sample_input, True)