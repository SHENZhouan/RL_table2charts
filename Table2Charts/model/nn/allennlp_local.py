# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Minimal replacements for the few AllenNLP APIs used by CopyNet, so the project
# runs without installing the full `allennlp` package (see dockerfile for upstream stack).

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def tiny_value_of_dtype(dtype: torch.dtype) -> float:
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype in (torch.float32, torch.float64):
        return 1e-13
    if dtype == torch.float16:
        return 1e-4
    raise TypeError(f"Does not support dtype {dtype}")


def masked_softmax(
    vector: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = -1,
    memory_efficient: bool = False,
) -> torch.Tensor:
    if mask is None:
        return torch.nn.functional.softmax(vector, dim=dim)
    while mask.dim() < vector.dim():
        mask = mask.unsqueeze(1)
    if not memory_efficient:
        result = torch.nn.functional.softmax(vector * mask, dim=dim)
        result = result * mask
        result = result / (result.sum(dim=dim, keepdim=True) + tiny_value_of_dtype(result.dtype))
    else:
        masked_vector = vector.masked_fill(~mask.bool(), float("-inf"))
        result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for _i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(1, attention.size(1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


def get_lengths_from_binary_sequence_mask(mask: torch.Tensor) -> torch.LongTensor:
    return mask.sum(-1).long()


def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor):
    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)
    index_range = torch.arange(0, len(sequence_lengths), device=sequence_lengths.device)
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index


def get_final_encoder_states(
    encoder_outputs: torch.Tensor, mask: torch.Tensor, bidirectional: bool = False
) -> torch.Tensor:
    last_word_indices = mask.sum(1) - 1
    batch_size, _, encoder_output_dim = encoder_outputs.size()
    expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch_size, 1, encoder_output_dim)
    final_encoder_output = encoder_outputs.gather(1, expanded_indices).squeeze(1)
    if bidirectional:
        final_forward_output = final_encoder_output[:, : (encoder_output_dim // 2)]
        final_backward_output = encoder_outputs[:, 0, (encoder_output_dim // 2) :]
        final_encoder_output = torch.cat([final_forward_output, final_backward_output], dim=-1)
    return final_encoder_output


class _Util:
    masked_softmax = staticmethod(masked_softmax)
    weighted_sum = staticmethod(weighted_sum)
    get_final_encoder_states = staticmethod(get_final_encoder_states)


util = _Util()


class Activation:
    @staticmethod
    def by_name(name: str):
        mapping = {
            "linear": nn.Identity,
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported activation: {name}")
        return mapping[name]


RnnState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class PytorchSeq2SeqWrapper(nn.Module):
    """Mirrors AllenNLP `PytorchSeq2SeqWrapper` for non-stateful GRU/LSTM encoders (batch_first)."""

    def __init__(self, module: nn.Module, stateful: bool = False) -> None:
        super().__init__()
        if stateful:
            raise NotImplementedError("Stateful encoder not needed for Table2Charts CopyNet path.")
        self._module = module
        self.stateful = False
        try:
            self._is_bidirectional = self._module.bidirectional
        except AttributeError:
            self._is_bidirectional = False
        self._num_directions = 2 if self._is_bidirectional else 1

    def is_bidirectional(self) -> bool:
        """AllenNLP exposes this as a method; CopyNet calls ``encoder.is_bidirectional()``."""
        return self._is_bidirectional

    def sort_and_run_forward(self, module, inputs, mask, hidden_state=None):
        batch_size = mask.size(0)
        num_valid = int(torch.sum(mask[:, 0]).item())
        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        sorted_inputs, sorted_sequence_lengths, restoration_indices, _sorting_indices = sort_batch_by_length(
            inputs, sequence_lengths
        )
        packed_sequence_input = pack_padded_sequence(
            sorted_inputs[:num_valid, :, :],
            sorted_sequence_lengths[:num_valid].cpu().tolist(),
            batch_first=True,
            enforce_sorted=False,
        )
        module_output, final_states = module(packed_sequence_input, hidden_state)
        return module_output, final_states, restoration_indices

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor, hidden_state: Optional[RnnState] = None
    ) -> torch.Tensor:
        batch_size, total_sequence_length = mask.size()
        packed_sequence_output, final_states, restoration_indices = self.sort_and_run_forward(
            self._module, inputs, mask, hidden_state
        )
        unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)
        num_valid = unpacked_sequence_tensor.size(0)
        if num_valid < batch_size:
            _, length, output_dim = unpacked_sequence_tensor.size()
            zeros = unpacked_sequence_tensor.new_zeros(batch_size - num_valid, length, output_dim)
            unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 0)
        sequence_length_difference = total_sequence_length - unpacked_sequence_tensor.size(1)
        if sequence_length_difference > 0:
            zeros = unpacked_sequence_tensor.new_zeros(
                batch_size, sequence_length_difference, unpacked_sequence_tensor.size(-1)
            )
            unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 1)
        return unpacked_sequence_tensor.index_select(0, restoration_indices)


class LinearAttention(nn.Module):
    """Linear attention with combination ``x,y`` (same as AllenNLP default for this call site)."""

    def __init__(
        self,
        tensor_1_dim: int,
        tensor_2_dim: int,
        combination: str = "x,y",
        activation: Optional[nn.Module] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if combination != "x,y":
            raise NotImplementedError(f"Only x,y combination is supported, got {combination}")
        if tensor_1_dim != tensor_2_dim:
            raise NotImplementedError("Mismatched tensor dims not implemented for local LinearAttention.")
        self._d = tensor_1_dim
        self._weight_vector = nn.Parameter(torch.Tensor(2 * self._d))
        self._bias = nn.Parameter(torch.Tensor(1))
        self._activation = activation if activation is not None else nn.Identity()
        self._normalize = normalize
        std = math.sqrt(6 / (self._weight_vector.numel() + 1))
        nn.init.uniform_(self._weight_vector, -std, std)
        nn.init.zeros_(self._bias)

    def forward(
        self,
        vector: torch.Tensor,
        matrix: torch.Tensor,
        matrix_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        w1, w2 = self._weight_vector[: self._d], self._weight_vector[self._d :]
        s1 = torch.matmul(vector.unsqueeze(1), w1)
        s2 = torch.matmul(matrix, w2)
        logits = self._activation(s1 + s2 + self._bias)
        if self._normalize:
            return masked_softmax(logits, matrix_mask)
        return logits
