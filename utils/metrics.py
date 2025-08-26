from __future__ import division

import time
import datetime
import warnings
from collections import defaultdict, deque

import numpy as np

import torch
import torch.distributed as dist

from .misc import is_dist_avail_and_initialized

def accuracy_binary(output, target):
    batch_size = output.shape[0]
    preds = (output.reshape(-1) >= 0.5).int()
    targets = target.reshape(-1)
    return torch.count_nonzero(preds == targets).div(batch_size)

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def classification_report(targets, predictions, labels, digits=2, output_dict=False):
    """
    Collect targets and predictions across workers and calculate classification report
    Args:
        targets (list[tensor])
        predictions (list[tensor])
        labels (list[str | int])
        digits (int): report decimal precision
        output_dict(bool): output dict instead of string
    Returns: str | dict
        classification report 
    """
    from sklearn.metrics import classification_report
    
    classes = list(range(len(labels)))
    cr = classification_report(targets.cpu().numpy(), predictions.cpu().numpy(), labels=classes, target_names=labels, digits=digits, output_dict=output_dict)
    return cr

'''
Multi-Label classification stats
'''
def average_precision(pred, target):
    """Calculate the average precision for a single class.
    AP summarizes a precision-recall curve as the weighted mean of maximum
    precisions obtained for any r'>r, where r is the recall:
    ..math::
        \\text{AP} = \\sum_n (R_n - R_{n-1}) P_n
    Note that no approximation is involved since the curve is piecewise
    constant.
    Args:
        pred (np.ndarray): The model prediction with shape (N, ).
        target (np.ndarray): The target of each prediction with shape (N, ).
    Returns:
        float: a single float as average precision value.
    """
    eps = np.finfo(np.float32).eps

    # sort examples
    sort_inds = np.argsort(-pred)
    sort_target = target[sort_inds]

    # count true positive examples
    pos_inds = sort_target == 1
    tp = np.cumsum(pos_inds)
    total_pos = tp[-1]

    # count not difficult examples
    pn_inds = sort_target != -1
    pn = np.cumsum(pn_inds)

    tp[np.logical_not(pos_inds)] = 0
    precision = tp / np.maximum(pn, eps)
    ap = np.sum(precision) / np.maximum(total_pos, eps)
    return ap

def mean_average_precision(pred, target):
    """Calculate the mean average precision with respect of classes.
    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape
            (N, C), where C is the number of classes.
        target (torch.Tensor | np.ndarray): The target of each prediction with
            shape (N, C), where C is the number of classes. 1 stands for
            positive examples, 0 stands for negative examples and -1 stands for
            difficult examples.
    Returns:
        float: A single float as mAP value.
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')

    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'
    num_classes = pred.shape[1]
    ap = np.zeros(num_classes)
    for k in range(num_classes):
        ap[k] = average_precision(pred[:, k], target[:, k])
    mean_ap = ap.mean() * 100.0
    return mean_ap

def average_performance(pred, target, thr=None, k=None):
    """Calculate CP, CR, CF1, OP, OR, OF1, where C stands for per-class
    average, O stands for overall average, P stands for precision, R stands for
    recall and F1 stands for F1-score.
    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape
            (N, C), where C is the number of classes.
        target (torch.Tensor | np.ndarray): The target of each prediction with
            shape (N, C), where C is the number of classes. 1 stands for
            positive examples, 0 stands for negative examples and -1 stands for
            difficult examples.
        thr (float): The confidence threshold. Defaults to None.
        k (int): Top-k performance. Note that if thr and k are both given, k
            will be ignored. Defaults to None.
    Returns:
        tuple: (CP, CR, CF1, OP, OR, OF1)
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')
    if thr is None and k is None:
        thr = 0.5
        warnings.warn('Neither thr nor k is given, set thr as 0.5 by '
                      'default.')
    elif thr is not None and k is not None:
        warnings.warn('Both thr and k are given, use threshold in favor of '
                      'top-k.')

    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'

    eps = np.finfo(np.float32).eps
    target[target == -1] = 0
    if thr is not None:
        # a label is predicted positive if the confidence is no lower than thr
        pos_inds = pred >= thr

    else:
        # top-k labels will be predicted positive for any example
        sort_inds = np.argsort(-pred, axis=1)
        sort_inds_ = sort_inds[:, :k]
        inds = np.indices(sort_inds_.shape)
        pos_inds = np.zeros_like(pred)
        pos_inds[inds[0], sort_inds_] = 1

    tp = (pos_inds * target) == 1
    fp = (pos_inds * (1 - target)) == 1
    fn = ((1 - pos_inds) * target) == 1

    precision_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fp.sum(axis=0), eps)
    recall_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fn.sum(axis=0), eps)
    CP = precision_class.mean() * 100.0
    CR = recall_class.mean() * 100.0
    CF1 = 2 * CP * CR / np.maximum(CP + CR, eps)
    OP = tp.sum() / np.maximum(tp.sum() + fp.sum(), eps) * 100.0
    OR = tp.sum() / np.maximum(tp.sum() + fn.sum(), eps) * 100.0
    OF1 = 2 * OP * OR / np.maximum(OP + OR, eps)
    return CP, CR, CF1, OP, OR, OF1


'''
Distributed Metrics
'''
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def std(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.std().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
    
    def __getitem__(self, item):
        return self.meters[item]

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB), flush=True)
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)), flush=True)
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))