# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import numpy as np
import os
import torch
import torch.distributed as dist
from copy import copy
from enum import IntEnum
from helper import save_ddp_checkpoint
from sklearn import metrics
from time import perf_counter, process_time
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
from typing import List

try:
    from apex import amp
except ImportError:
    pass

from data import DataConfig, SpecialTokens, QValue, State, Sequence, get_template
from search.agent import ParallelAgents, SearchConfig
from search.agent.update_MC_light_drill_down import UpdateMCLightBeamDrillDownAgent
from util import to_device, scores_from_confusion, time_str
from .update_MC_light_config import StudentConfig
from .replay_memory import ReplayMemory

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class BatchMode(IntEnum):
    Estimate = 0,
    Train = 1,
    Test = 2


class Student:
    def __init__(self, config: StudentConfig, data_config: DataConfig, search_config: SearchConfig,
                 ddp: Module, use_apex: bool, device, local_rank, summary_writer: SummaryWriter):
        self.config = config
        self.data_config = data_config
        self.search_config = search_config
        self.special_tokens = SpecialTokens(data_config)

        self.model = ddp
        self.optimizer = config.optimizer
        self.criterion = config.criterion
        self.apex = use_apex

        self.local_rank = local_rank
        self.device = device
        self.agent_workers = self._resolve_agent_workers_(getattr(config, "agent_workers", 0))
        self.agents = ParallelAgents(max_workers=self.agent_workers)
        self.memory = ReplayMemory(config.memory_size)

        self.current_epoch = -1
        self.is_testing = False
        self.start_perf_time = 0
        self.start_process_time = 0
        self.batch_cnt = 0
        self.loss_sum = 0.0
        self.confusion_sum = np.zeros((2, 2), dtype=int)

        self.mc_top_k = max(1, int(getattr(config, "mc_top_k", 2)))
        self.mc_rollout_depth = max(1, int(getattr(config, "mc_rollout_depth", 1)))
        self.mc_num_rollouts = max(1, int(getattr(config, "mc_num_rollouts", 1)))
        self.mc_discount = float(getattr(config, "mc_discount", 0.9))
        self.mc_rollout_weight = float(getattr(config, "mc_rollout_weight", 0.35))
        self.rollout_rng = np.random.default_rng(getattr(config, "mc_seed", 20260426))

        self.logger = logging.getLogger(f"Student {config.log_tag}")
        self.log_freq = config.log_freq
        self.summary_writer = summary_writer
        self.global_train_step = 0
        self.global_test_step = 0

        fake_state = State.init_state(get_template(data_config.search_types[0],
                                                   data_config.allow_multiple_values,
                                                   data_config.consider_grouping_operations,
                                                   data_config.limit_search_group))
        fake_actions = Sequence([], [])
        samples = [QValue(fake_state, fake_actions, [], [])]
        self.fake_data = self._prepare_data_(samples, only_estimate=True)

    @staticmethod
    def _resolve_agent_workers_(requested_workers: int) -> int:
        if requested_workers and requested_workers > 0:
            return requested_workers
        cpu_count = os.cpu_count() or 8
        return max(8, min(32, cpu_count))

    def add_table(self, tUID: str):
        self.agents.add(UpdateMCLightBeamDrillDownAgent(tUID, self.data_config, self.special_tokens,
                                                        self.search_config))

    def n_tables(self):
        return self.agents.remaining() + self.agents.finished()

    def reset(self, epoch: int, is_testing: bool):
        self.start_perf_time = perf_counter()
        self.start_process_time = process_time()
        self.batch_cnt = 0
        self.loss_sum = 0.0
        self.confusion_sum = np.zeros((2, 2), dtype=int)

        self.current_epoch = epoch
        self.is_testing = is_testing

        self.agents.shutdown()
        self.agents = ParallelAgents(max_workers=self.agent_workers)

    def metrics(self):
        return self.loss_sum, self.batch_cnt, self.confusion_sum, self.start_perf_time

    def save_checkpoint(self, dir_path: str):
        return save_ddp_checkpoint(dir_path, self.current_epoch, self.model, self.optimizer)

    def _prepare_data_(self, samples: List[QValue], only_estimate: bool = False):
        data = QValue.collate(samples, self.data_config, not self.is_testing and self.data_config.field_permutation)
        if only_estimate:
            data.pop("values")
        return to_device(data, self.device)

    def _estimate_scores_(self, samples: List[QValue]) -> np.ndarray:
        if len(samples) == 0:
            return np.empty([0])
        data = self._prepare_data_(samples, only_estimate=True)
        output = self.model(data["state"], data["actions"])
        return output.detach()[:, :, 1].exp().cpu().numpy()

    def _successor_sample_(self, sample: QValue, action_idx: int):
        next_state = copy(sample.state).append(sample.actions[action_idx])
        if next_state.is_complete():
            return None
        valid_actions = next_state.valid_actions(sample.actions, max_rc=self.search_config.max_rc,
                                                 top_freq_func=self.data_config.top_freq_func)
        return QValue(next_state, sample.actions, valid_actions, [0] * len(sample.actions))

    def _top_k_valid_indices_(self, sample: QValue, scores: np.ndarray) -> np.ndarray:
        valid_indices = np.flatnonzero(sample.valid_mask[:len(sample.actions)])
        if len(valid_indices) == 0:
            return valid_indices
        ranked = valid_indices[np.argsort(scores[valid_indices])[::-1]]
        return ranked[:min(self.mc_top_k, len(ranked))]

    def _one_step_rollout_(self, sample: QValue) -> float:
        if sample is None or not sample.has_valid_action:
            return 0.0
        scores = self._estimate_scores_([sample])[0]
        topk_indices = self._top_k_valid_indices_(sample, scores)
        if len(topk_indices) == 0:
            return 0.0
        topk_scores = scores[topk_indices]
        weights = np.clip(topk_scores, 1e-8, None)
        weights = weights / weights.sum()
        chosen_idx = int(self.rollout_rng.choice(topk_indices, p=weights))
        return float(scores[chosen_idx])

    def _rollout_return_(self, sample: QValue, depth: int) -> float:
        if sample is None or depth <= 0 or not sample.has_valid_action:
            return 0.0

        if depth == 1:
            return self._one_step_rollout_(sample)

        scores = self._estimate_scores_([sample])[0]
        topk_indices = self._top_k_valid_indices_(sample, scores)
        if len(topk_indices) == 0:
            return 0.0

        topk_scores = scores[topk_indices]
        weights = np.clip(topk_scores, 1e-8, None)
        weights = weights / weights.sum()
        chosen_idx = int(self.rollout_rng.choice(topk_indices, p=weights))
        immediate = float(scores[chosen_idx])
        successor = self._successor_sample_(sample, chosen_idx)
        if successor is None:
            return immediate
        return immediate + self.mc_discount * self._rollout_return_(successor, depth - 1)

    def _mc_adjust_scores_(self, sample: QValue, base_scores: np.ndarray) -> np.ndarray:
        adjusted = base_scores.copy()
        topk_indices = self._top_k_valid_indices_(sample, adjusted)
        for idx in topk_indices:
            successor = self._successor_sample_(sample, int(idx))
            if successor is None:
                continue
            futures = [self._rollout_return_(successor, self.mc_rollout_depth)
                       for _ in range(self.mc_num_rollouts)]
            avg_future = float(np.mean(futures)) if futures else 0.0
            adjusted[idx] = adjusted[idx] + self.mc_rollout_weight * self.mc_discount * avg_future
        return adjusted

    def _estimate_light_rollout_batch_(self, successors: List[QValue]) -> np.ndarray:
        """
        Fast path for MC-light's default setting:
        - mc_rollout_depth == 1
        - mc_num_rollouts == 1

        In that case each root successor only needs one extra model estimate.
        We batch those successor estimates together instead of running one
        forward pass per successor.
        """
        if len(successors) == 0:
            return np.empty([0], dtype=float)

        scores_batch = self._estimate_scores_(successors)
        rollout_returns = np.zeros(len(successors), dtype=float)
        for i, (sample, scores) in enumerate(zip(successors, scores_batch)):
            topk_indices = self._top_k_valid_indices_(sample, scores)
            if len(topk_indices) == 0:
                continue
            topk_scores = scores[topk_indices]
            weights = np.clip(topk_scores, 1e-8, None)
            weights = weights / weights.sum()
            chosen_idx = int(self.rollout_rng.choice(topk_indices, p=weights))
            rollout_returns[i] = float(scores[chosen_idx])
        return rollout_returns

    def _mc_adjust_scores_batch_light_(self, samples: List[QValue], estimates: np.ndarray) -> np.ndarray:
        adjusted_batch = np.array(estimates, copy=True)
        successor_samples = []
        successor_refs = []

        for sample_idx, (sample, scores) in enumerate(zip(samples, adjusted_batch)):
            topk_indices = self._top_k_valid_indices_(sample, scores)
            for action_idx in topk_indices:
                successor = self._successor_sample_(sample, int(action_idx))
                if successor is None:
                    continue
                successor_refs.append((sample_idx, int(action_idx)))
                successor_samples.append(successor)

        rollout_returns = self._estimate_light_rollout_batch_(successor_samples)
        for (sample_idx, action_idx), future in zip(successor_refs, rollout_returns):
            adjusted_batch[sample_idx, action_idx] += self.mc_rollout_weight * self.mc_discount * float(future)
        return adjusted_batch

    def _feed_batch_nn_(self, mode: BatchMode, samples: List[QValue]) -> np.ndarray:
        self.model.train(mode is BatchMode.Train)

        if len(samples) == 0:
            with torch.no_grad():
                self.model(self.fake_data["state"], self.fake_data["actions"])
            return np.empty([0])

        with torch.set_grad_enabled(mode is BatchMode.Train):
            if mode is BatchMode.Estimate:
                with torch.no_grad():
                    estimates = self._estimate_scores_(samples)
                    if self.mc_rollout_depth == 1 and self.mc_num_rollouts == 1:
                        return self._mc_adjust_scores_batch_light_(samples, estimates)
                    return np.array([self._mc_adjust_scores_(sample, scores)
                                     for sample, scores in zip(samples, estimates)], dtype=float)

            data = self._prepare_data_(samples)
            output = self.model(data["state"], data["actions"])
            target = data["values"]
            loss = self.criterion(output.transpose(-1, -2), target)

            if mode is BatchMode.Train:
                self.optimizer.zero_grad()
                if self.apex:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()

            self.batch_cnt += 1
            self.loss_sum += loss.item()
            y_pred = output.detach().argmax(dim=-1).cpu().numpy().ravel()
            y_true = target.cpu().numpy().ravel()
            valid_b = (y_true != -1)
            y_pred = y_pred[valid_b]
            y_true = y_true[valid_b]
            self.confusion_sum += metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])

            ap, ar, af1 = scores_from_confusion(self.confusion_sum)
            tfnp = self.confusion_sum.ravel()
            if self.batch_cnt % self.log_freq == 0:
                phase_tag = "test/valid" if self.is_testing else "train"
                self.logger.info(f"EP-{self.current_epoch} {phase_tag} B{self.batch_cnt}: loss={loss.item()}" +
                                 "| (tn, fp, fn, tp)=%s all_prf1=(%f, %f, %f) avg_loss=%f all_acc=%f" % (
                                     tfnp, ap, ar, af1,
                                     self.loss_sum / self.batch_cnt,
                                     np.trace(self.confusion_sum) / np.sum(self.confusion_sum)) +
                                 "| process=%.1fs elapsed=%.1fs" % (
                                     process_time() - self.start_process_time,
                                     perf_counter() - self.start_perf_time))

            if self.local_rank == 0:
                if self.is_testing:
                    self.global_test_step += 1
                    global_step = self.global_test_step
                else:
                    self.global_train_step += 1
                    global_step = self.global_train_step
                phase_tag = 'test/valid' if self.is_testing else 'train'
                self.summary_writer.add_scalar("{0}/loss".format(phase_tag), loss.item(), global_step)
                self.summary_writer.add_scalar("{0}/avg_loss".format(phase_tag), self.loss_sum / self.batch_cnt,
                                               global_step)
                self.summary_writer.add_scalar("{0}/all_acc".format(phase_tag),
                                               np.trace(self.confusion_sum) / np.sum(self.confusion_sum),
                                               global_step)
                self.summary_writer.add_scalar("{0}/precision".format(phase_tag), ap, global_step)
                self.summary_writer.add_scalar("{0}/recall".format(phase_tag), ar, global_step)
                self.summary_writer.add_scalar("{0}/f1_score".format(phase_tag), af1, global_step)
                self.summary_writer.flush()

            del loss
            if mode is BatchMode.Test:
                return output.detach()[:, :, 1].exp().cpu().numpy()
        return np.empty([0])

    @staticmethod
    def _feed_batch_random_(samples: List[QValue]) -> List:
        return [np.clip(sample.values + np.random.rand(len(sample.values)), 0, 1) for sample in samples]

    def _act_step_(self, samples: List[QValue]):
        useful_samples = [sample for sample in samples if sample.has_valid_action]
        for sample in useful_samples:
            self.memory.push(sample)
        if not self.is_testing and self.config.random_train:
            return self._feed_batch_random_(samples)
        useful_results = self._feed_batch_nn_(BatchMode.Estimate, useful_samples)
        results = []
        idx = 0
        for sample in samples:
            if sample.has_valid_action:
                results.append(useful_results[idx])
                idx += 1
            else:
                results.append(sample.values)
        return results

    def act_step(self):
        futures = self.agents.step([self._act_step_])
        finished_info = self.agents.update([(lambda: future.result()) for future in futures])
        return finished_info

    def sample_learn(self, rounds: int, batch_size: int):
        for _ in range(rounds):
            self._feed_batch_nn_(BatchMode.Train, self.memory.sample(batch_size))

    def dist_summary(self):
        end_perf_time = perf_counter()
        loss_sum_tensor = torch.tensor(self.loss_sum, device=self.device, dtype=torch.double)
        confusion_sum_tensor = torch.tensor(self.confusion_sum, device=self.device, dtype=torch.int64)

        merged_info = self.agents.summary(divide_total=False)
        log_dir_path = os.path.join(self.config.log_dir, "test-valid" if self.is_testing else "train")
        os.makedirs(log_dir_path, exist_ok=True)
        log_file_path = os.path.join(log_dir_path,
                                     f"[summary-{self.current_epoch:02d}]{time_str()}.rank-{self.local_rank}.log")
        with open(log_file_path, "w") as log_file:
            json.dump(merged_info, log_file, sort_keys=True, indent=4)

        recall01ordered = merged_info['evaluation']['stages']["complete"]['recall']['@01']
        recall03ordered = merged_info['evaluation']['stages']["complete"]['recall']['@03']
        recall05ordered = merged_info['evaluation']['stages']["complete"]['recall']['@05']
        recall10ordered = merged_info['evaluation']['stages']["complete"]['recall']['@10']

        info_tensor = torch.tensor([
            merged_info['expanded_states'], merged_info['reached_states'],
            merged_info["cut_states"], merged_info["dropped_states"], merged_info["complete_states"],
            merged_info['perf_time'], merged_info['process_time'],
            recall01ordered, recall03ordered, recall05ordered, recall10ordered
        ], device=self.device, dtype=torch.double)

        success = merged_info['t_cnt']
        final_stage_cnt = merged_info['evaluation']['stages']["complete"]['t_cnt']
        cnt_tensor = torch.tensor([self.batch_cnt, success, final_stage_cnt], device=self.device, dtype=torch.int64)

        dist.all_reduce(loss_sum_tensor)
        dist.all_reduce(cnt_tensor)
        dist.all_reduce(confusion_sum_tensor)
        dist.all_reduce(info_tensor)
