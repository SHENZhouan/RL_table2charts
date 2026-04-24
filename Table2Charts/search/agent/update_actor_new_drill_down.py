# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import random
from copy import copy
from data import DataConfig, State, DataTable, TableQValues, QValue, SpecialTokens, Result, \
    determine_action_values, get_template
from search import Recorder
from sortedcontainers import SortedList
from typing import Optional, List, Iterable, Set, Union

from .agent import Agent
from .config import SearchConfig


class UpdateActorNewBeamDrillDownAgent(Agent):
    def __init__(self, info: Union[str, dict], data_config: DataConfig, special_tokens: SpecialTokens,
                 search_config: SearchConfig,
                 selected_field_indices: Optional[Set[int]] = None):
        self.tUID = info if isinstance(info, str) else None
        self.info = info if isinstance(info, dict) else None
        self.load_ground_truth = search_config.load_ground_truth
        self.data_config = data_config
        self.st = special_tokens
        self.table = None
        self.selected_fields = selected_field_indices
        self.max_rc = search_config.max_rc
        self.state_actions = dict()

        self.expand_limit = search_config.expand_limit
        self.time_limit = search_config.time_limit
        self.frontier_size = search_config.frontier_size
        self.beam_size = search_config.beam_size
        self.min_threshold = search_config.min_threshold

        self.frontier = SortedList([], key=lambda x: -x.score)
        self.beam = []
        self.finished = False

        self.recorder = None
        self.search_all_types = search_config.search_all_types
        self.search_single_type = search_config.search_single_type
        self.test_field_selections = search_config.test_field_selections
        self.test_design_choices = search_config.test_design_choices
        self.log_path = search_config.log_path
        self.is_single_inference = False

        self.search_config = search_config
        self.policy_rng = random.Random(getattr(search_config, "actor_policy_seed", None))

    def _policy_param(self, name: str, default):
        return getattr(self.search_config, name, default)

    def _normalize_actor_weights(self, actor_probs: List[float]) -> List[float]:
        temperature = max(float(self._policy_param("actor_sampling_temperature", 1.0)), 1e-6)
        adjusted = [math.pow(max(prob, 0.0), 1.0 / temperature) for prob in actor_probs]
        total = sum(adjusted)
        if total <= 0.0:
            return [1.0 / len(actor_probs)] * len(actor_probs)
        return [value / total for value in adjusted]

    def _choose_continuation_idx(self, new_results: List[Result], actor_probs: List[float]) -> Optional[int]:
        if len(new_results) == 0:
            return None

        if not self._policy_param("actor_use_sampling", True):
            return max(range(len(new_results)), key=lambda idx: actor_probs[idx])

        weights = self._normalize_actor_weights(actor_probs)
        return self.policy_rng.choices(range(len(new_results)), weights=weights, k=1)[0]

    def done(self):
        return self.finished

    def ranked_complete_states(self) -> List[Result]:
        return self.recorder.completed_results()

    def table(self):
        return self.table

    def initialize(self):
        if self.load_ground_truth:
            tqv = TableQValues(self.tUID, self.st, self.data_config, search_sampling=True)
            self.table = tqv.table
            self.state_actions = tqv.get_state_actions()
            if len(self.state_actions) == 0:
                self.finished = True
                raise ValueError("No user created valid PivotTable in {}!".format(self.tUID))
            self.recorder = Recorder(self.table, tqv.get_positive_prefixes(), self.log_path,
                                     test_field_selections=self.test_field_selections,
                                     test_design_choices=self.test_design_choices)
        else:
            self.table = DataTable(self.info if self.info is not None else self.tUID, self.st, self.data_config)
            self.recorder = Recorder(self.table, log_path=self.log_path,
                                     test_field_selections=self.test_field_selections,
                                     test_design_choices=self.test_design_choices)

        search_ana_types = set(self.data_config.search_types)
        if not self.search_all_types:
            search_ana_types.intersection_update(self.table.ana_type_set)
        if self.search_single_type is not None:
            search_ana_types.intersection_update(self.search_single_type)
        for ana_type in search_ana_types:
            begin_state = State.init_state(get_template(ana_type, self.data_config.allow_multiple_values,
                                                        self.data_config.consider_grouping_operations,
                                                        self.data_config.limit_search_group))
            self.beam.append(begin_state)
            self.recorder.record_reached([Result(1., begin_state)])
        self.finished = False
        self.recorder.start()

    def step(self) -> List[QValue]:
        if self.finished:
            raise ValueError("Agent already done searching!")
        if self.table is None:
            self.initialize()

        if len(self.beam) < self.beam_size and len(self.frontier):
            swap_size = min(self.beam_size - len(self.beam), len(self.frontier))
            self.beam.extend(result.state for result in self.frontier[:swap_size])
            del self.frontier[:swap_size]

        chosen = []
        for state in self.beam:
            valid_actions = state.valid_actions(self.table.action_space, selected_field_indices=self.selected_fields,
                                                max_rc=self.max_rc, top_freq_func=self.data_config.top_freq_func)
            positive_actions = self.state_actions[state] if state in self.state_actions else None
            action_values = determine_action_values(self.table.action_space, positive_actions)
            chosen.append(QValue(state, self.table.action_space, valid_actions, action_values))
        return chosen

    def update(self, chosen: List[QValue], predicted_values: Iterable) -> Optional[dict]:
        if self.finished:
            raise ValueError("Agent already done searching!")

        self.recorder.count_expanded(len(chosen))
        beam_idx = 0
        incomplete_results = []
        for state_actions, prediction in zip(chosen, predicted_values):
            state = state_actions.state
            actions = state_actions.actions
            valid_mask = state_actions.valid_mask
            critic_scores = prediction["critic_scores"]
            actor_probs = prediction["actor_probs"]

            new_results = []
            valid_actor_probs = []
            for action, valid, critic_score, actor_prob in zip(actions, valid_mask,
                                                               critic_scores[:len(actions)],
                                                               actor_probs[:len(actions)]):
                if not valid:
                    continue
                new_results.append(Result(float(critic_score), copy(state).append(action)))
                valid_actor_probs.append(float(actor_prob))
            self.recorder.record_reached(new_results)

            best_idx = self._choose_continuation_idx(new_results, valid_actor_probs)
            best_state = new_results[best_idx].state if best_idx is not None else None

            if best_state is None or best_state.is_complete():
                self.beam[beam_idx] = None
            else:
                self.beam[beam_idx] = best_state

            for i, result in enumerate(new_results):
                if i == best_idx:
                    continue
                score, next_state = result
                if not next_state.is_complete():
                    if score < self.min_threshold:
                        self.recorder.count_cut(1)
                    else:
                        incomplete_results.append(result)
            beam_idx += 1

        self.frontier.update(incomplete_results)
        if len(self.frontier) > self.frontier_size:
            drop = len(self.frontier) - self.frontier_size
            del self.frontier[-drop:]
            self.recorder.count_dropped(drop)
        self.beam = [state for state in self.beam if state is not None]

        process_t, perf_t = self.recorder.passed_time()
        end_search = self.recorder.expanded_states >= self.expand_limit or process_t >= self.time_limit or \
                     (len(self.frontier) == 0 and len(self.beam) == 0)
        if end_search:
            self.finished = True
            return self.recorder.end(self.is_single_inference)
