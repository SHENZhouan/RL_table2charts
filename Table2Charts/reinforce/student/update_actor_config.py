# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch.nn import Module
from torch.optim.optimizer import Optimizer


class StudentConfig:
    def __init__(self, optimizer: Optimizer, loss: Module,
                 memory_size: int, min_memory: int, random_train: bool,
                 log_tag: str, log_freq: int, log_dir: str,
                 actor_loss_weight: float = 0.1, entropy_weight: float = 0.001,
                 critic_score_weight: float = 0.5, use_actor_in_eval: bool = False):
        self.optimizer = optimizer
        self.criterion = loss
        self.actor_loss_weight = actor_loss_weight
        self.entropy_weight = entropy_weight
        self.critic_score_weight = critic_score_weight
        self.use_actor_in_eval = use_actor_in_eval

        self.scale_start = 0.9
        self.scale_end = 0.001
        self.scale_decay = 0.8

        self.memory_size = memory_size
        self.min_memory = min_memory

        self.random_train = random_train

        self.log_tag = log_tag
        self.log_freq = log_freq
        self.log_dir = log_dir
