# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import torch
import torch.distributed as dist
from datetime import datetime, timezone, timedelta
from torch.nn import NLLLoss, Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim.optimizer import Optimizer

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    pass

from data import get_data_config, DataConfig
from model import get_cp_config
from model.nn.update_actor_copyNet import UpdateActorCopyNet
from util import log_params, load_optimizer, set_no_calculate_gradient, load_extra, save_checkpoint, load_states


def _dist_rank_or_default(default="single"):
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return default


def load_update_actor_checkpoint(file_path: str, module: Module, device="cpu"):
    states = load_states(file_path)
    incompatible = module.load_state_dict(states["module_state"], strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    logger = logging.getLogger(f"load_update_actor_checkpoint({_dist_rank_or_default()})")
    if missing:
        logger.info("Missing keys while loading checkpoint into update_actor model: %s", missing)
    if unexpected:
        logger.info("Unexpected keys while loading checkpoint into update_actor model: %s", unexpected)
    module.to(device)
    return states["epoch"], module, states.get("extra")


def construct_data_config(args, is_train=True) -> DataConfig:
    if is_train:
        data_config = get_data_config(args.corpus_path, args.features,
                                      args.search_type, args.previous_type, args.input_type,
                                      args.unified_ana_token, args.num_train_analysis, args.field_permutation,
                                      lang=args.lang, mode=args.mode)
    else:
        data_config = get_data_config(args.corpus_path, args.features,
                                      args.search_type, args.previous_type, args.input_type,
                                      args.unified_ana_token, field_permutation=args.field_permutation,
                                      lang=args.lang, mode=args.mode)

    if args.model_name == "cp":
        data_config.need_field_indices = True

    return data_config


def create_model(args):
    """
    Create CopyNet DQN based on configuration arguments
    :param args:
    :return: Raw model, Experiment name
    """
    rank = _dist_rank_or_default()
    logger = logging.getLogger(f"create_model({rank})")
    data_config = construct_data_config(args)

    if args.model_name == "cp":
        model_config = get_cp_config(data_config, args.model_size)
        model = UpdateActorCopyNet(model_config)
        if rank == 0:
            logger.info("UpdateActorCopyNetConfig: {}".format(vars(model_config)))
    else:
        raise NotImplementedError(f"{args.model_name} not yet implemented.")

    experiment_name = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d%H%M%S-update_actor-") + \
                      str(model_config) + "-" + str(args.search_type)
    log_params(args, data_config, model_config, experiment_name)

    return model, experiment_name


def prepare_model(model: Module, device, args):
    """
    Freeze parameters, reinitialize parts, and wrap with DDP.
    :param model: the raw model
    :param device:
    :param args:
    :return: ddp, optimizer, loss
    """
    logger = logging.getLogger(f"prepare_model({_dist_rank_or_default()})")

    if args.pre_model_file:
        logger.info(f"Loading pre-trained / previous model from {args.pre_model_file}...")
        load_update_actor_checkpoint(args.pre_model_file, model, device)
    else:
        args.restart_epoch = -1

    # TODO: check freeze and re-initialize parameters before DDP wrapping works or not.
    # Freeze parameters
    if args.freeze_embed:
        for module in model.get_embed_modules():
            set_no_calculate_gradient(module)
    if args.freeze_encoder:
        for module in model.get_encoder_modules():
            set_no_calculate_gradient(module)

    # Re-initialize layers in decoder
    if args.fresh_decoder:
        if isinstance(model, UpdateActorCopyNet):
            model._attention.reset_parameters()
            model._input_projection_layer.reset_parameters()
            model._decoder_cell.reset_parameters()
            model._output_generation_layer_1.reset_parameters()
            model._output_generation_layer_2.reset_parameters()
            model._output_copying_layer_1.reset_parameters()
            model._output_copying_layer_2.reset_parameters()
            model._actor_generation_layer.reset_parameters()
            model._actor_copying_layer.reset_parameters()

    model.to(device)

    if args.apex:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        ddp = DDP(model)
        if args.restart_epoch >= 0:  # Also load optimizer if it's a restart
            load_optimizer(args.pre_model_file, optimizer)
            extra = load_extra(args.pre_model_file)
            amp.load_state_dict(extra['amp'])
    else:
        ddp = DistributedDataParallel(model, device_ids=[device], output_device=device)
        # Setting the Adam optimizer with hyper-param
        optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
        if args.restart_epoch >= 0:  # Also load optimizer if it's a restart
            load_optimizer(args.pre_model_file, optimizer)

    # Using Negative Log Likelihood Loss function for predicting the action-value
    criterion = NLLLoss(weight=torch.tensor((args.negative_weight, 1)).to(device), ignore_index=-1, reduction="sum")

    return ddp, optimizer, criterion


def save_ddp_checkpoint(save_dir: str, epoch: int, ddp: Module, optimizer: Optimizer = None, apex: bool = False):
    if apex:
        extra = {'amp': amp.state_dict()}
        return save_checkpoint(save_dir, epoch, ddp.module, optimizer, extra)
    else:
        return save_checkpoint(save_dir, epoch, ddp.module, optimizer)
