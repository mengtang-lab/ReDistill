import numpy as np
import torch


def measure_latency(**kwargs):
    '''
        latency_repetitions: number of repetitions for inference
        model: torch model
        inputs: torch tensor
        device: cuda | cpu
    '''
    if "latency_repetitions" in kwargs: latency_repetitions = kwargs["latency_repetitions"]
    else: latency_repetitions = 300
    model, inputs, device = kwargs["model"], kwargs["inputs"], kwargs["device"]
    inputs = tuple([input.to(device) for input in inputs])
    model.to(device)
    model.eval()

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((round(latency_repetitions), 1))
    # GPU-WARM-UP
    for _ in range(int(0.1 * latency_repetitions)):
        _ = model(*inputs)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(latency_repetitions):
            starter.record()
            _ = model(*inputs)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / latency_repetitions
    std_syn = np.std(timings)
    print("Latency: {:.4f} +/- {:.4f} ms/{} images".format(mean_syn, std_syn, inputs[0].shape[0]))


def Latency(func):
    def decorator(**kwargs):
        measure_latency(**kwargs)
        return func(**kwargs)
    return decorator