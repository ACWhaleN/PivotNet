# 导入所需库和包
# Import necessary libraries and packages
import os
import sys
import torch
import functools
import numpy as np
from torch.nn import Module
from tabulate import tabulate
from abc import ABCMeta, abstractmethod
from mapmaster.utils.misc import DictAction


class BaseExp(metaclass=ABCMeta):
    """
    BaseExp类是所有实验的基础类，提供了一组实验所需的接口作为抽象类

    The BaseExp class serves as the base class for all experiments, providing a set of essential interfaces for experiments as an abstract class.

    属性/Attributes
    ----------
    _batch_size_per_device : int
        每个设备的批量大小
        Batch size for each device
    _total_devices : int
        使用的设备总数
        Total number of devices used
    _max_epoch : int
        总训练周期数，学习率调度可能会根据这个值进行调整
        Total number of training epochs, the learning rate scheduler may adapt based on this value
    """


    def __init__(self, batch_size_per_device, total_devices, max_epoch):
        self._batch_size_per_device = batch_size_per_device
        self._max_epoch = max_epoch
        self._total_devices = total_devices
        # ----------------------------------------------- extra configure ------------------------- #
        self.seed = None
        self.exp_name = os.path.splitext(os.path.basename(sys.argv.copy()[0]))[0]  # entrypoint filename as exp_name
        self.print_interval = 100
        self.dump_interval = 10
        self.eval_interval = 10
        self.num_keep_latest_ckpt = 10
        self.ckpt_oss_save_dir = None
        self.enable_tensorboard = False
        self.eval_executor_class = None

    @property
    def train_dataloader(self):
        """
        配置并获取训练数据加载器

        Configure and get the train dataloader.
        
        返回/Returns
        -------
        torch.utils.data.DataLoader
            配置好的训练数据加载器实例
            An instance of the configured train dataloader
        """
        if "_train_dataloader" not in self.__dict__:
            # 如果未配置，将其配置为训练数据加载器
            # Configure it as the train dataloader if not already configured
            self._train_dataloader = self._configure_train_dataloader()
        return self._train_dataloader

    @property
    def val_dataloader(self):
        """
        配置并获取验证数据加载器

        Configure and get the validation dataloader.
        
        返回/Returns
        -------
        torch.utils.data.DataLoader
            配置好的验证数据加载器实例
            An instance of the configured validation dataloader
        """
        if "_val_dataloader" not in self.__dict__:
            # 如果未配置，将其配置为验证数据加载器
            # Configure it as the validation dataloader if not already configured
            self._val_dataloader = self._configure_val_dataloader()
        return self._val_dataloader

    @property
    def test_dataloader(self):
        """
        配置并获取测试数据加载器

        Configure and get the test dataloader.
        
        返回/Returns
        -------
        torch.utils.data.DataLoader
            配置好的测试数据加载器实例
            An instance of the configured test dataloader
        """
        if "_test_dataloader" not in self.__dict__:
            # 如果未配置，将其配置为测试数据加载器
            # Configure it as the test dataloader if not already configured
            self._test_dataloader = self._configure_test_dataloader()
        return self._test_dataloader

    @property
    def model(self):
        """
        配置并获取模型实例

        Configure and get the model instance.
        
        返回/Returns
        -------
        torch.nn.Module
            配置的模型实例
            The configured model instance
        """
        if "_model" not in self.__dict__:
            # 配置并返回模型实例
            # Configure and return the model instance
            self._model = self._configure_model()
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def callbacks(self):
        """
        配置并获取回调函数列表

        Configure and get the list of callback functions.
        
        返回/Returns
        -------
        list
            配置好的回调函数列表
            A list of configured callback functions
        """
        if not hasattr(self, "_callbacks"):
            # 如果回调函数列表不存在，则配置它
            # If the callback list does not exist, configure it
            self._callbacks = self._configure_callbacks()
        return self._callbacks

    @property
    def optimizer(self):
        """
        配置并获取优化器实例

        Configure and get the optimizer instance.
        
        返回/Returns
        -------
        torch.optim.Optimizer
            配置好的优化器实例
            The configured optimizer instance
        """
        if "_optimizer" not in self.__dict__:
            # 如果优化器不存在，则进行配置
            # Configure the optimizer if it does not exist
            self._optimizer = self._configure_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        """
        配置并获取学习率调度器实例

        Configure and get the learning rate scheduler instance.
        
        返回/Returns
        -------
        torch.optim.lr_scheduler
            配置好的学习率调度器实例
            The configured learning rate scheduler instance
        """
        if "_lr_scheduler" not in self.__dict__:
            # 如果学习率调度器不存在，则进行配置
            # Configure the learning rate scheduler if it does not exist
            self._lr_scheduler = self._configure_lr_scheduler()
        return self._lr_scheduler

    @property
    def batch_size_per_device(self):
        return self._batch_size_per_device

    @property
    def max_epoch(self):
        return self._max_epoch

    @property
    def total_devices(self):
        return self._total_devices

    @abstractmethod
    def _configure_model(self) -> Module:
        pass

    @abstractmethod
    def _configure_train_dataloader(self):
        """"""

    def _configure_callbacks(self):
        return []

    @abstractmethod
    def _configure_val_dataloader(self):
        """"""

    @abstractmethod
    def _configure_test_dataloader(self):
        """"""

    def training_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def _configure_optimizer(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def _configure_lr_scheduler(self, **kwargs):
        pass

    def update_attr(self, options: dict) -> str:
        if options is None:
            return ""
        assert isinstance(options, dict)
        msg = ""
        for k, v in options.items():
            if k in self.__dict__:
                old_v = self.__getattribute__(k)
                if not v == old_v:
                    self.__setattr__(k, v)
                    msg = "{}\n'{}' is overriden from '{}' to '{}'".format(msg, k, old_v, v)
            else:
                self.__setattr__(k, v)
                msg = "{}\n'{}' is set to '{}'".format(msg, k, v)

        # update exp_name
        exp_name_suffix = "-".join(sorted([f"{k}-{v}" for k, v in options.items()]))
        self.exp_name = f"{self.exp_name}--{exp_name_suffix}"
        return msg

    def get_cfg_as_str(self) -> str:
        config_table = []
        for c, v in self.__dict__.items():
            if not isinstance(v, (int, float, str, list, tuple, dict, np.ndarray)):
                if hasattr(v, "__name__"):
                    v = v.__name__
                elif hasattr(v, "__class__"):
                    v = v.__class__
                elif type(v) == functools.partial:
                    v = v.func.__name__
            if c[0] == "_":
                c = c[1:]
            config_table.append((str(c), str(v)))

        headers = ["config key", "value"]
        config_table = tabulate(config_table, headers, tablefmt="plain")
        return config_table

    def __str__(self):
        return self.get_cfg_as_str()

    def to_onnx(self):
        pass

    @classmethod
    def add_argparse_args(cls, parser):  # pragma: no-cover
        parser.add_argument(
            "--exp_options",
            nargs="+",
            action=DictAction,
            help="override some settings in the exp, the key-value pair in xxx=yyy format will be merged into exp. "
            'If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b '
            'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
            "Note that the quotation marks are necessary and that no white space is allowed.",
        )
        parser.add_argument("-b", "--batch-size-per-device", type=int, default=None)
        parser.add_argument("-e", "--max-epoch", type=int, default=None)
        return parser
