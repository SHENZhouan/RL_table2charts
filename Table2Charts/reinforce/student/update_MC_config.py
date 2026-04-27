# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch.nn import Module
from torch.optim.optimizer import Optimizer


class StudentConfig:
    def __init__(self, optimizer: Optimizer, loss: Module,
                 memory_size: int, min_memory: int, random_train: bool,
                 log_tag: str, log_freq: int, log_dir: str,
                 mc_top_k: int = 3, mc_rollout_depth: int = 2,
                 mc_num_rollouts: int = 2, mc_discount: float = 0.9,
                 mc_rollout_weight: float = 0.5, mc_seed: int = 20260425):
        self.optimizer = optimizer
        self.criterion = loss

        self.scale_start = 0.9
        self.scale_end = 0.001
        self.scale_decay = 0.8

        self.memory_size = memory_size
        self.min_memory = min_memory

        self.random_train = random_train

        self.log_tag = log_tag
        self.log_freq = log_freq
        self.log_dir = log_dir

        self.mc_top_k = mc_top_k
        self.mc_rollout_depth = mc_rollout_depth
        self.mc_num_rollouts = mc_num_rollouts
        self.mc_discount = mc_discount
        self.mc_rollout_weight = mc_rollout_weight
        self.mc_seed = mc_seed
