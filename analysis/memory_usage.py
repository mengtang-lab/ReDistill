import torch

def memory_usage(func):
    def decorator(**kwargs):
        device = kwargs["device"]
        print(f"Memory Usage: {torch.cuda.memory_allocated(device=device) / 1024**2} M")
        print(f"Memory Cached: {torch.cuda.memory_reserved(device=device) / 1024**2} M")
        return func(**kwargs)
    return decorator

def max_memory(func):
    def decorator(**kwargs):
        device = kwargs["device"]
        print(f"Max Memory: {torch.cuda.max_memory_allocated(device=device) / 1024**2} M")
        print(f"Max Mmeory Cached: {torch.cuda.max_memory_reserved(device=device) / 1024 **2} M")
        return func(**kwargs)
    return decorator
