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
from search.agent.update_MC_drill_down import UpdateMCBeamDrillDownAgent
from util import to_device, scores_from_confusion, time_str
from .update_MC_config import StudentConfig
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
        self.agents = ParallelAgents()
        self.memory = ReplayMemory(config.memory_size)

        self.current_epoch = -1
        self.is_testing = False
        self.start_perf_time = 0
        self.start_process_time = 0
        self.batch_cnt = 0
        self.loss_sum = 0.0
        self.confusion_sum = np.zeros((2, 2), dtype=int)

        self.mc_top_k = max(1, int(getattr(config, "mc_top_k", 3)))
        self.mc_rollout_depth = max(1, int(getattr(config, "mc_rollout_depth", 2)))
        self.mc_num_rollouts = max(1, int(getattr(config, "mc_num_rollouts", 2)))
        self.mc_discount = float(getattr(config, "mc_discount", 0.9))
        self.mc_rollout_weight = float(getattr(config, "mc_rollout_weight", 0.5))
        self.rollout_rng = np.random.default_rng(getattr(config, "mc_seed", 20260425))

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

    def add_table(self, tUID: str):
        self.agents.add(UpdateMCBeamDrillDownAgent(tUID, self.data_config, self.special_tokens, self.search_config))

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
        self.agents = ParallelAgents()

    def metrics(self):
        return self.loss_sum, self.batch_cnt, self.confusion_sum, self.start_perf_time

    def save_checkpoint(self, dir_path: str):
        return save_ddp_checkpoint(dir_path, self.current_epoch, self.model, self.optimizer)

    def _prepare_data_(self, samples: List[QValue], only_estimate: bool = False):
        d_config = self.data_config
        data = QValue.collate(samples, d_config, not self.is_testing and d_config.field_permutation)
        if only_estimate:
            data.pop("values")
        data = to_device(data, self.device)
        return data

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

    def _rollout_return_(self, sample: QValue, depth: int) -> float:
        if sample is None or depth <= 0 or not sample.has_valid_action:
            return 0.0

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
            futures = [self._rollout_return_(successor, self.mc_rollout_depth - 1)
                       for _ in range(self.mc_num_rollouts)]
            avg_future = float(np.mean(futures)) if len(futures) else 0.0
            adjusted[idx] = adjusted[idx] + self.mc_rollout_weight * self.mc_discount * avg_future
        return adjusted

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
                    adjusted = []
                    for sample, scores in zip(samples, estimates):
                        adjusted.append(self._mc_adjust_scores_(sample, scores))
                    return np.array(adjusted, dtype=float)
            else:
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
                matrix = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
                self.confusion_sum += matrix

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
            samples = self.memory.sample(batch_size)
            self._feed_batch_nn_(BatchMode.Train, samples)

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

        if self.local_rank == 0:
            avg_loss = loss_sum_tensor.item() / cnt_tensor[0].item() if cnt_tensor[0].item() > 0 else 0
            confusion_sum = confusion_sum_tensor.cpu().numpy()
            precision, recall, f1 = scores_from_confusion(confusion_sum)
            info_tensor = (info_tensor[:, 0] / info_tensor[:, 1]).cpu().numpy()

            tfnp = confusion_sum.ravel()
            self.logger.info(f"EP-{self.current_epoch} {'test/valid' if self.is_testing else 'train'} SUMMARY: " +
                             "elapsed=%.1fs | avg_loss=%f " % (end_perf_time - self.start_perf_time, avg_loss) +
                             "(tn, fp, fn, tp)=%s " % tfnp +
                             "precision=%f recall=%f f1=%f | " % (precision, recall, f1) +
                             f"total_cnt={cnt_tensor[0].item()}"
                             f"success_cnt={cnt_tensor[1].item()} "
                             f"#states(expanded, reached, cut, dropped, complete)="
                             f"({info_tensor[0]:.2f}, {info_tensor[1]:.2f}, "
                             f"{info_tensor[2]:.2f}, {info_tensor[3]:.2f}, {info_tensor[4]:.2f}) " +
                             f"t(perf, process)=({info_tensor[5]:.2f}s, {info_tensor[6]:.2f}s) " +
                             f"final_stage_cnt={cnt_tensor[2].item()} R@1={info_tensor[7]} " +
                             f"R@3={info_tensor[8]} R@5={info_tensor[9]} R@10={info_tensor[10]}")

            phase_tag = ('test/valid' if self.is_testing else 'train') + "-summary"
            global_step = self.current_epoch
            self.summary_writer.add_scalar("{0}/tn".format(phase_tag), tfnp[0], global_step)
            self.summary_writer.add_scalar("{0}/fp".format(phase_tag), tfnp[1], global_step)
            self.summary_writer.add_scalar("{0}/fn".format(phase_tag), tfnp[2], global_step)
            self.summary_writer.add_scalar("{0}/tp".format(phase_tag), tfnp[3], global_step)
            self.summary_writer.add_scalar("{0}/precision".format(phase_tag), precision, global_step)
            self.summary_writer.add_scalar("{0}/recall".format(phase_tag), recall, global_step)
            self.summary_writer.add_scalar("{0}/f1".format(phase_tag), f1, global_step)
            self.summary_writer.add_scalar("{0}/R@1_ordered".format(phase_tag), info_tensor[7], global_step)
            self.summary_writer.add_scalar("{0}/R@3_ordered".format(phase_tag), info_tensor[8], global_step)
            self.summary_writer.add_scalar("{0}/R@5_ordered".format(phase_tag), info_tensor[9], global_step)
            self.summary_writer.add_scalar("{0}/R@10_ordered".format(phase_tag), info_tensor[10], global_step)
            self.summary_writer.flush()
