# 导入所需库和包
# Import necessary libraries and packages
import os
import re
import torch
import torchvision
import unicodedata
from sys import stderr
from torch import Tensor
from loguru import logger
from argparse import Action
from collections import deque
from typing import Optional, List
from torch import distributed as dist


__all__ = [
    "PyDecorator", "NestedTensor", "AvgMeter", "DictAction", "sanitize_filename", "parse_devices", 
    "_max_by_axis", "nested_tensor_from_tensor_list", "_onnx_nested_tensor_from_tensor_list", 
    "get_param_groups", "setup_logger", "get_rank", "get_world_size", "synchronize", "reduce_sum", 
    "reduce_mean", "all_gather_object", "is_distributed", "is_available"
]


class PyDecorator:
    """
    PyDecorator类提供Python常见的装饰器

    The PyDecorator class provides common decorators for Python.
    """
    
    @staticmethod
    def overrides(interface_class):
        """
        确保一个方法是在给定接口类中重写的方法

        Ensures that a method is an override from a given interface class.

        参数/Parameters
        ----------
        interface_class : type
            要重写方法的接口类/The interface class to override methods from

        返回/Returns
        -------
        function
            被覆写的方法 / The method being overridden
        """
        def overrider(method):
            assert method.__name__ in dir(interface_class), "{} function not in {}".format(
                method.__name__, interface_class
            )
            return method
        return overrider


class NestedTensor(object):
    """
    NestedTensor类用于包含Tensor及其对应的mask的结构体

    The NestedTensor class is used to encapsulate a Tensor with its corresponding mask.
    """
    
    def __init__(self, tensors, mask: Optional[Tensor]):
        """
        初始化NestedTensor实例

        Initialize a NestedTensor instance.

        参数/Parameters
        ----------
        tensors : torch.Tensor
            实际存储的数据
            The data tensor
        mask : Optional[Tensor]
            与Tensor匹配的可选mask
            An optional mask matching the tensor
        """
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        """
        将NestedTensor转换到指定设备上

        Convert the NestedTensor to a specified device

        参数/Parameters
        ----------
        device : torch.device
            目标设备
            Target device

        返回/Returns
        -------
        NestedTensor
            转换到目标设备后的NestedTensor
            The NestedTensor on the target device
        """
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        """
        返回NestedTensor中包含的tensors和mask

        Return the tensors and mask contained in the NestedTensor.

        返回/Returns
        -------
        Tuple[Tensor, Optional[Tensor]]
            包含的tensors和mask
            The tensors and mask contained
        """
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
    

class AvgMeter(object):
    """
    AvgMeter类用于计算值的平均数，支持有限窗口大小

    The AvgMeter class is used to calculate averages of values, with support for a limited window size.
    """
    
    def __init__(self, window_size=50):
        """
        初始化AvgMeter实例

        Initialize an AvgMeter instance.

        参数/Parameters
        ----------
        window_size : int, optional
            窗口大小，用于计算滑动平均数
            The window size for calculating the moving average
        """
        self.window_size = window_size
        self._value_deque = deque(maxlen=window_size)
        self._total_value = 0.0
        self._wdsum_value = 0.0
        self._count_deque = deque(maxlen=window_size)
        self._total_count = 0.0
        self._wdsum_count = 0.0

    def reset(self):
        """
        重置AvgMeter的所有状态

        Reset all the states of the AvgMeter.
        """
        self._value_deque.clear()
        self._total_value = 0.0
        self._wdsum_value = 0.0
        self._count_deque.clear()
        self._total_count = 0.0
        self._wdsum_count = 0.0

    def update(self, value, n=1):
        """
        更新AvgMeter的值

        Update the values of the AvgMeter.

        参数/Parameters
        ----------
        value : float
            新的值
            New value
        n : int, optional
            值的权重，默认值为1
            Weight of the value, default is 1
        """
        if len(self._value_deque) >= self.window_size:
            self._wdsum_value -= self._value_deque.popleft()
            self._wdsum_count -= self._count_deque.popleft()
        self._value_deque.append(value * n)
        self._total_value += value * n
        self._wdsum_value += value * n
        self._count_deque.append(n)
        self._total_count += n
        self._wdsum_count += n

    @property
    def avg(self):
        return self.global_avg

    @property
    def global_avg(self):
        """
        返回全局平均数

        Return the global average.

        返回/Returns
        -------
        float
            全局平均数
            The global average
        """
        return self._total_value / max(self._total_count, 1e-5)

    @property
    def window_avg(self):
        """
        返回窗口平均数

        Return the window average.

        返回/Returns
        -------
        float
            窗口平均数
            The window average
        """
        return self._wdsum_value / max(self._wdsum_count, 1e-5)


class DictAction(Action):
    """
    argparse动作类，用于将参数拆分为KEY=VALUE形式

    An argparse action to split an argument into KEY=VALUE form and append to a dictionary.    
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ["true", "false"]:
            return True if val.lower() == "true" else False
        return val

    @staticmethod
    def _parse_iterable(val):
        """
        解析字符串中的可迭代值。所有在'()'或'[]'中的元素被视为可迭代值。
        
        Parse iterable values in strings, where all elements inside '()' or '[]' are treated as iterable values.

        参数/Parameters
        ----------
        val : str
            字符串值 / The string value

        返回/Returns
        -------
        list | tuple
            扩展的列表或元组 / The expanded list or tuple

        示例/Examples
        -------
        >>> DictAction._parse_iterable('1,2,3')
        [1, 2, 3]
        >>> DictAction._parse_iterable('[a, b, c]')
        ['a', 'b', 'c']
        >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
        [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """
            在字符串中寻找下一个逗号的位置。如果字符串中没有找到逗号，则返回字符串长度
            (即逗号在'()'和'[]'内会被忽略)

            Find the position of the next comma in the string. If no comma is found, return string length.
            Commas inside '()' and '[]' are ignored.
            """
            assert (string.count("(") == string.count(")")) and (
                string.count("[") == string.count("]")
            ), f"Imbalanced brackets exist in {string}"
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if (char == ",") and (pre.count("(") == pre.count(")")) and (pre.count("[") == pre.count("]")):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip("'\"").replace(" ", "")
        is_tuple = False
        if val.startswith("(") and val.endswith(")"):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith("[") and val.endswith("]"):
            val = val[1:-1]
        elif "," not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1 :]
        if is_tuple:
            values = tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split("=", maxsplit=1)
            options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)


def sanitize_filename(value, allow_unicode=False):
    """
    清理文件名以确保安全性和标准化

    Sanitize the filename to ensure safety and standardization.

    参数/Parameters
    ----------
    value : str
        原文件名/The original filename
    allow_unicode : bool, optional
        是否允许Unicode字符/Whether to allow unicode characters

    返回/Returns
    -------
    str
        清理过的文件名/The sanitized filename
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def parse_devices(gpu_ids):
    """
    解析并返回可用GPU的设备ID

    Parse and return available GPU device IDs.

    参数/Parameters
    ----------
    gpu_ids : str
        描述GPU范围的字符串/A string describing GPU range

    返回/Returns
    -------
    str
        格式化的GPU ID串/Formated GPU IDs string
    """
    if "-" in gpu_ids:
        gpus = gpu_ids.split("-")
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        parsed_ids = ",".join(map(lambda x: str(x), list(range(*gpus))))
        return parsed_ids
    else:
        return gpu_ids


def _max_by_axis(the_list):
    """
    接受一个整数列表的列表并返回每个子列表轴上的最大值。
    
    Takes a list of lists of integers and returns the max of each axis of the sublists.

    参数/Parameters
    ----------
    the_list : List[List[int]]
        需要计算最大值的列表 / The list to calculate max values from

    返回/Returns
    -------
    List[int]
        每个列的最大值/Maximum of each column
    """
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    """
    从Tensor列表创建一个NestedTensor实例

    Create a NestedTensor instance from a list of Tensors.

    参数/Parameters
    ----------
    tensor_list : List[Tensor]
        组成NestedTensor的Tensor列表/The list of Tensors forming NestedTensor

    返回/Returns
    -------
    NestedTensor
        包含Tensors和对应Masks的NestedTensor实例/NestedTensor instance with Tensors and corresponding Masks
    """
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    """
    从Tensor列表创建一个支持ONNX跟踪的NestedTensor实例

    Create a NestedTensor instance from a list of Tensors supporting ONNX tracing.

    参数/Parameters
    ----------
    tensor_list : List[Tensor]
        组成NestedTensor的Tensor列表/The list of Tensors forming NestedTensor

    返回/Returns
    -------
    NestedTensor
        包含Tensors和对应Masks的NestedTensor实例/NestedTensor instance with Tensors and corresponding Masks
    """
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def get_param_groups(model, optimizer_setup):
    """
    根据优化器设置划分模型参数组

    Divide the model parameter groups based on the optimizer setup.

    参数/Parameters
    ----------
    model : torch.nn.Module
        所需划分参数组的模型/Model whose parameters are to be grouped
    optimizer_setup : dict
        优化器的设置/Setup for the optimizer

    返回/Returns
    -------
    List[dict]
        参数组列表/List of parameter groups
    """
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model.named_parameters():
        if match_name_keywords(n, optimizer_setup["freeze_names"]):
            p.requires_grad = False

    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not match_name_keywords(n, optimizer_setup["backb_names"])
                and not match_name_keywords(n, optimizer_setup["extra_names"])
                and p.requires_grad
            ],
            "lr": optimizer_setup["base_lr"],
            "wd": optimizer_setup["wd"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if match_name_keywords(n, optimizer_setup["backb_names"]) and p.requires_grad
            ],
            "lr": optimizer_setup["backb_lr"],
            "wd": optimizer_setup["wd"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if match_name_keywords(n, optimizer_setup["extra_names"]) and p.requires_grad
            ],
            "lr": optimizer_setup["extra_lr"],
            "wd": optimizer_setup["wd"],
        },
    ]

    return param_groups


def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """
    为训练和测试设置日志记录器

    Setup the logger for training and testing.

    参数/Parameters
    ----------
    save_dir : str
        日志文件的保存位置/Location to save the log file
    distributed_rank : int, optional
        多GPU环境下设备顺序/Device rank in multi-GPU environment
    mode : str, optional
        日志文件写入模式，默认为`a`/Log file write mode, default is `a`

    返回/Returns
    -------
    logger instance
        日志实例/Logger instance
    """
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    format = f"[Rank #{distributed_rank}] | " + "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}"
    if distributed_rank > 0:
        logger.remove()
        logger.add(
            stderr,
            format=format,
            level="WARNING",
        )
    logger.add(
        save_file,
        format=format,
        filter="",
        level="INFO" if distributed_rank == 0 else "WARNING",
        enqueue=True,
    )

    return logger


def get_rank() -> int:
    """
    获取当前进程的种子

    Get the rank of the current process.

    返回/Returns
    -------
    int
        当前进程的种子/Rank of the current process
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """
    获取当前环境的世界大小

    Get the world size of the current environment.

    返回/Returns
    -------
    int
        当前环境的世界大小/World size of the current environment
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def synchronize():
    """
    在使用分布式训练时用于同步（障碍）所有进程的帮助函数

    Helper function to synchronize (barrier) among all processes when using distributed training.
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    current_world_size = dist.get_world_size()
    if current_world_size == 1:
        return
    dist.barrier()


def reduce_sum(tensor):
    """
    在所有进程间减少（求和）张量

    Reduce (sum) a tensor across all processes.

    参数/Parameters
    ----------
    tensor : torch.Tensor
        需要聚合的张量/Tensor to be reduced

    返回/Returns
    -------
    Tensor
        聚合后的张量/Reduced tensor
    """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def reduce_mean(tensor):
    """
    在所有进程间减少（平均）张量

    Reduce (mean) a tensor across all processes.

    参数/Parameters
    ----------
    tensor : torch.Tensor
        需要聚合的张量/Tensor to be reduced

    返回/Returns
    -------
    Tensor
        平均后的张量/Mean tensor
    """
    return reduce_sum(tensor) / float(get_world_size())


def all_gather_object(obj):
    """
    从所有进程收集对象

    Gather objects from all processes.

    参数/Parameters
    ----------
    obj : Any
        需要收集的对象/Objects to be gathered

    返回/Returns
    -------
    List
        从所有进程收集的对象/Objects gathered from all processes
    """
    world_size = get_world_size()
    if world_size < 2:
        return [obj]
    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, obj)
    return output


def is_distributed() -> bool:
    """
    检查是否使用了分布式训练

    Check if distributed training is being used.

    返回/Returns
    -------
    bool
        是否使用分布式训练/Whether distributed training is being used
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def is_available() -> bool:
    """
    检查分布式功能是否可用

    Check if distributed capabilities are available.

    返回/Returns
    -------
    bool
        分布式功能是否可用/Whether distributed capabilities are available
    """
    return dist.is_available()
