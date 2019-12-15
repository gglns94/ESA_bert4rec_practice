import torch.nn as nn

from abc import *


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, args, dataset_meta):
        super().__init__()
        self.args = args
        self.dataset_meta = dataset_meta

    @classmethod
    @abstractmethod
    def code(cls):
        pass

