import os
import psutil
import socket
import pickle
from collections import abc

import torch
import torch.nn.functional as F
import torch.distributed as dist

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def init_distributed_mode(cfg):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        cfg.RANK = int(os.environ["RANK"])
        cfg.WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        cfg.GPU = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ and 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        cfg.RANK = int(os.environ['SLURM_PROCID'])
        cfg.WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        cfg.GPU = cfg.RANK % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        cfg.GPU = list(range(torch.cuda.device_count()))
        cfg.DIST = False
        return

    cfg.DIST = True

    torch.cuda.set_device(cfg.GPU)
    print(f'{socket.gethostname()} | DISTRIBUTED INIT (RANK {cfg.RANK}): {cfg.DIST_URL}, GPU {cfg.GPU}', flush=True)
    dist.init_process_group(backend=cfg.DIST_BACKEND,
                            init_method=cfg.DIST_URL,
                            world_size=cfg.WORLD_SIZE,
                            rank=cfg.RANK)
    dist.barrier()
    # XXX: disable printing if not in master process
    setup_for_distributed(is_master=cfg.RANK == 0)

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def sync_tensor_list_dist(lst):
    lst = torch.cat(lst).cpu()
    # collect lst from all workers when DDP enabled
    if is_dist_avail_and_initialized():
        dist_lst = [None] * get_world_size()
        dist.barrier()
        dist.all_gather_object(dist_lst, lst)
        lst = torch.cat(dist_lst)

    return lst

def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024 ** 3

def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage, total

def pad_img_batch(batch, mode='constant', value=0):
    """
    Pad images in a batch to max height and max width
    Args:
        batch: list
            Input batch of images to pad, 
            e.g [img1, img2, img3] in C, H, W
        mode: str
        value: int
            value to pad with
    """
    max_h, max_w = 0, 0
    # find max h, w
    for img in batch:
        h, w = img.shape[-2:]
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    padded_batch = []
    for img in batch:
        h, w = img.shape[-2:]
        w_p = max_w - w
        w_s = w_p // 2
        w_e = w_p - w_s

        h_p = max_h - h
        h_s = h_p // 2
        h_e = h_p - h_s

        # pad image equally in both dimensions
        img_padded = F.pad(img, (w_s, w_e, h_s, h_e), mode, value)
        padded_batch.append(img_padded)

    return padded_batch

def flatten_old(seq):
    """
    Flatten lists of lists of lists of....
    """
    for el in seq:
        if isinstance(el, (tuple, list)):
            yield from flatten(el)
        else:
            yield el

def flatten(data):
    if isinstance(data, (list, tuple)):
        return [item for sublist in data for item in flatten(sublist)]
    elif isinstance(data, dict):
        return [item for value in data.values() for item in flatten(value)]
    elif isinstance(data, abc.Iterable) and not isinstance(data, str):
        return [item for item in data]
    else:
        return [data]

def nan_hook(self, inp, out):
    """
    Check for NaN inputs or outputs at each layer in the model
    Usage:
        # forward hook
        for submodule in model.modules():
            submodule.register_forward_hook(nan_hook)
    """

    outputs = isinstance(out, tuple) and out or [out]
    inputs = isinstance(inp, tuple) and inp or [inp]

    outputs = flatten(outputs)
    inputs = flatten(inputs)

    contains_nan = lambda x: torch.isnan(x).any()
    layer = self.__class__.__name__

    for i, inp in enumerate(inputs):
        if inp is not None and contains_nan(inp):
            raise RuntimeError(f'Found NaN input at index: {i} in layer: {layer}')

    for i, out in enumerate(outputs):
        if out is not None and contains_nan(out):
            raise RuntimeError(f'Found NaN output at index: {i} in layer: {layer}')

def inf_hook(self, inp, out):
    """
    Check for inf inputs or outputs at each layer in the model
    Usage:
        # forward hook
        for submodule in model.modules():
            submodule.register_forward_hook(nan_hook)
    """
    outputs = isinstance(out, tuple) and out or [out]
    inputs = isinstance(inp, tuple) and inp or [inp]

    outputs = flatten(outputs)
    inputs = flatten(inputs)

    contains_inf = lambda x: torch.isinf(x).any()
    layer = self.__class__.__name__

    for i, inp in enumerate(inputs):
        if inp is not None and contains_inf(inp):
            raise RuntimeError(f'Found inf input at index: {i} in layer: {layer}')

    for i, out in enumerate(outputs):
        if out is not None and contains_inf(out):
            raise RuntimeError(f'Found inf output at index: {i} in layer: {layer}')