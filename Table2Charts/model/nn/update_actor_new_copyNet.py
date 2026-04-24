# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .update_actor_copyNet import UpdateActorCopyNetSeq2Seq
from .config import CopyNetConfig
from .embedding import InputEmbedding


class UpdateActorNewCopyNet(UpdateActorCopyNetSeq2Seq):
    def __init__(self, config: CopyNetConfig):
        input_embed = InputEmbedding(config)
        super().__init__(input_embed, config)
