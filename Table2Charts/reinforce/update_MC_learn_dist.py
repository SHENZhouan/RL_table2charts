# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Launch this script following https://pytorch.org/docs/stable/distributed.html#launch-utility
Or find examples in learn_dist.sh
"""
import argparse
import logging
import multiprocessing as mp
import numpy as np
import os
import pika
import sys
import torch
import torch.distributed as dist
import traceback
from data import Index, DEFAULT_LANGUAGES, DEFAULT_FEATURE_CHOICES, DEFAULT_ANALYSIS_TYPES
from enum import IntEnum
from helper import construct_data_config, create_model, prepare_model
from model import DEFAULT_MODEL_SIZES, DEFAULT_MODEL_NAMES
from os import path, getpid
from reinforce.student.update_MC_student import Student
from reinforce.student.update_MC_config import StudentConfig
from search.agent import get_search_config, DEFAULT_SEARCH_LIMITS
from time import perf_counter
from torch.utils.tensorboard import SummaryWriter
from util import num_params

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.getLogger("pika").setLevel(logging.WARNING)


def _slice_tuids(tuids, limit=None, offset=0, stride=1):
    if stride <= 0:
        raise ValueError("table stride must be >= 1")
    if offset < 0:
        raise ValueError("table offset must be >= 0")

    selected = tuids[offset::stride]
    if limit is not None and limit >= 0:
        selected = selected[:limit]
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description="RL Training")

    parser.add_argument('-p', '--pre_model_file', type=str, metavar='PATH',
                        help='file path to the pre-trained model (as the starting point)')
    parser.add_argument('-m', "--model_save_path", default="/storage/models/", type=str)
    parser.add_argument('-l', '--log_save_path', default="evaluations", type=str, metavar='PATH',
                        help='subdir path of model_save_path to log the evaluation metrics during validation/testing')

    parser.add_argument("--corpus_path", type=str, required=True, help="The corpus path for metadata task.")
    parser.add_argument("--lang", "--specify_lang", choices=DEFAULT_LANGUAGES, default='en', type=str,
                        help="Specify the header language(s) to load tables.")

    parser.add_argument("--model_name", choices=DEFAULT_MODEL_NAMES, default="tf", type=str)

    parser.add_argument('--model_size', choices=DEFAULT_MODEL_SIZES, required=True, type=str)
    parser.add_argument('--features', choices=DEFAULT_FEATURE_CHOICES, default="all-mul_bert", type=str,
                        help="Limit the data loading and control the feature ablation.")
    parser.add_argument('--log_freq_batch', default=100, type=int, metavar='N',
                        help='number of batches to print dqn evaluation metrics')
    parser.add_argument('--log_freq_agent', default=100, type=int, metavar='N',
                        help='number of tables to print agent evaluation metrics')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total training epochs to run')
    parser.add_argument('--restart_epoch', default=-1, type=int, metavar='N',
                        help='if the pretrain model is from an interrupted model saved by this script, '
                             'reload and restart from next epoch')
    parser.add_argument('--summary_path', default="/storage/summaries/", type=str,
                        help='tensorboard summary path')
    parser.add_argument('-s', '--search_type', choices=DEFAULT_ANALYSIS_TYPES, type=str, required=True,
                        help="Determine which data to load and what types of analysis to search.")
    parser.add_argument('--input_type', choices=DEFAULT_ANALYSIS_TYPES, type=str, default=None,
                        help="Determine which data to load. This parameter is prior to --search_type.")
    parser.add_argument('--previous_type', choices=DEFAULT_ANALYSIS_TYPES, type=str,
                        help="Tell the action space information of pre_model_file/model_file."
                             "Bar grouping should be the same as in data_constraint.")
    parser.add_argument('--field_permutation', default=False, dest='field_permutation', action='store_true',
                        help="Whether to randomly permutate table fields when training.")
    parser.add_argument('--unified_ana_token', default=False, dest='unified_ana_token', action='store_true',
                        help="Whether to use unified analysis token [ANA] instead of concrete type tokens.")
    parser.add_argument("--freeze_embed", default=False, dest='freeze_embed', action='store_true',
                        help="Whether to freeze params in embedding layer."
                             "Only take effect when --pre_model_file is available.")
    parser.add_argument("--freeze_encoder", default=False, dest='freeze_encoder', action='store_true',
                        help="Whether to freeze params in encoder layers."
                             "Only take effect when --pre_model_file is available.")
    parser.add_argument("--fresh_decoder", default=False, dest='fresh_decoder', action='store_true',
                        help="Whether to re-initialize params in decoding layers (attention layer, copy layer, etc.)"
                             "Only take effect when --pre_model_file is available.")

    parser.add_argument("--negative_weight", default=0.02, type=float, help="Negative class weight for NLLLoss.")
    parser.add_argument('--memory_size', default=150000, type=int, metavar='N',
                        help='the max capacity of experience replay memory')
    parser.add_argument('--min_memory', default=5000, type=int, metavar='N',
                        help='min number of experiences to start learning')
    parser.add_argument('--memory_sample_size', default=128, type=int, metavar='N',
                        help='the number of experiences in each memory sampling, '
                             'in other words, training batch size for each distributed process')
    parser.add_argument('--memory_sample_rounds', default=4, type=int, metavar='N',
                        help='how many times to do the sampling after each expansion of agents')
    parser.add_argument("--num_train_analysis", type=int, help="Number of Analysis each ana_type for training.")

    parser.add_argument('--random_train', action="store_true",
                        help="Set the flag if training samples will be generated randomly.")
    parser.add_argument('--max_tables', default=64, type=int, metavar='N',
                        help='number of tables to handle at the same time')
    parser.add_argument('--train_table_limit', default=None, type=int, metavar='N',
                        help='optionally limit training to the first N tables after offset/stride slicing')
    parser.add_argument('--valid_table_limit', default=None, type=int, metavar='N',
                        help='optionally limit validation to the first N tables after offset/stride slicing')
    parser.add_argument('--train_table_offset', default=0, type=int, metavar='N',
                        help='skip the first N training tables before subset selection')
    parser.add_argument('--valid_table_offset', default=0, type=int, metavar='N',
                        help='skip the first N validation tables before subset selection')
    parser.add_argument('--train_table_stride', default=1, type=int, metavar='N',
                        help='keep every Nth training table after the offset')
    parser.add_argument('--valid_table_stride', default=1, type=int, metavar='N',
                        help='keep every Nth validation table after the offset')
    parser.add_argument('--search_limits', choices=DEFAULT_SEARCH_LIMITS, default="e200-b8-r4c2", type=str,
                        help="Search config option")
    parser.add_argument('--mc_top_k', default=3, type=int,
                        help='root and rollout candidate width for top-k MC evaluation')
    parser.add_argument('--mc_rollout_depth', default=2, type=int,
                        help='how many action steps to evaluate in each shallow rollout')
    parser.add_argument('--mc_num_rollouts', default=2, type=int,
                        help='how many Monte Carlo rollouts to run per root action')
    parser.add_argument('--mc_discount', default=0.9, type=float,
                        help='discount factor applied to future rollout returns')
    parser.add_argument('--mc_rollout_weight', default=0.5, type=float,
                        help='how much rollout return influences the root action score')
    parser.add_argument('--mc_seed', default=20260425, type=int,
                        help='random seed for shallow MC rollout sampling')

    parser.add_argument('--apex', default=False, dest='apex', action='store_true',
                        help="Use NVIDIA Apex DistributedDataParallel instead of the PyTorch one.")
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int, metavar='N',
                        help="local rank to guide use which GPU device, given by the launch script")
    parser.add_argument('--amqp_addr', default="127.0.0.1", type=str,
                        help="Last node (rank world_size - 1)'s address, should be either "
                             "the IP address or the hostname of the node, for "
                             "single node multi-proc training, the --master_addr can simply be 127.0.0.1")
    parser.add_argument('--amqp_port', default=5672, type=str)
    parser.add_argument('--amqp_routing_key', default='TUID_QUEUE', type=str)
    parser.add_argument('--amqp_user', default='dist', type=str)
    parser.add_argument('--amqp_pwd', default='CommonAnalysis', type=str)
    parser.add_argument('--queue_mode', choices=['amqp', 'local'], default='amqp', type=str,
                        help="Task queue mode. Use 'local' for single-process smoke tests without RabbitMQ.")

    return parser.parse_args()


class SenderTask(IntEnum):
    Train = 0
    Valid = 1
    Stop = 2


def get_conn_channel(args, purge_queue: bool = False):
    credentials = pika.PlainCredentials(username=args.amqp_user, password=args.amqp_pwd)
    parameters = pika.ConnectionParameters(host=args.amqp_addr, port=args.amqp_port, credentials=credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.queue_declare(args.amqp_routing_key, durable=True)
    if purge_queue:
        channel.queue_purge(args.amqp_routing_key)
    return connection, channel


def task_queue(receiver, world_size: int, index: Index, args):
    logger = logging.getLogger("pika sender")

    def send_msg(channel, message: str):
        channel.basic_publish(exchange='', routing_key=args.amqp_routing_key, body=message.encode("ascii"),
                              properties=pika.BasicProperties(delivery_mode=2))

    logger.info("Wait for commands...")
    while True:
        task = receiver.recv()
        if task == SenderTask.Train:
            logger.info("Train Command Recv.")
            train_tuids = index.train_tUIDs()
            connection, channel = get_conn_channel(args)
            for i in np.random.permutation(len(train_tuids)):
                send_msg(channel, train_tuids[i])
            for _ in range(world_size):
                send_msg(channel, "")
            logger.info(f"Train {len(train_tuids)} + {world_size} sent.")
            channel.close()
            connection.close()
        elif task == SenderTask.Valid:
            logger.info("Valid Command Recv.")
            valid_tuids = index.valid_tUIDs()
            connection, channel = get_conn_channel(args)
            for tUID in valid_tuids:
                send_msg(channel, tUID)
            for _ in range(world_size):
                send_msg(channel, "")
            logger.info(f"Valid {len(valid_tuids)} + {world_size} sent.")
            channel.close()
            connection.close()
        else:
            logger.info("Stop")
            return


def dist_sum(device, value):
    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor)
    return tensor.item()


def dist_min(device, value):
    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return tensor.item()


def iteration(epoch: int, student: Student, is_testing: bool, n_tables: int, args, local_tuids=None):
    student.reset(epoch, is_testing)

    start_perf_t = perf_counter()
    queue_empty = False
    enough_memory = False
    logged_finished = 0
    if args.queue_mode == 'amqp':
        connection, channel = get_conn_channel(args)
    else:
        connection, channel = None, None
        local_tuids = [] if local_tuids is None else local_tuids
        local_idx = 0
    cnt = 0
    while dist_sum(student.device, student.agents.finished()) < n_tables:
        while not queue_empty and student.agents.remaining() < args.max_tables:
            if args.queue_mode == 'local':
                if local_idx >= len(local_tuids):
                    queue_empty = True
                    break
                student.add_table(local_tuids[local_idx])
                local_idx += 1
                cnt += 1
                continue
            try:
                method, properties, body = channel.basic_get(args.amqp_routing_key, auto_ack=True)
            except (ConnectionResetError, pika.exceptions.StreamLostError):
                traceback.print_exc(file=sys.stdout)
                student.logger.info("Setting up connection again...")
                connection, channel = get_conn_channel(args)
                method, properties, body = channel.basic_get(args.amqp_routing_key, auto_ack=True)

            if body is None:
                continue
            cnt += 1
            tUID = body.decode("ascii")

            if len(tUID) == 0:
                queue_empty = True
            else:
                student.add_table(tUID)

        student.act_step()
        if student.agents.finished() - logged_finished >= args.log_freq_agent:
            student.logger.info(f"Agents finished={student.agents.finished()} remaining={student.agents.remaining()}"
                                f" error={student.agents.error_cnt}! " +
                                f"EP-{epoch} ({'test/valid' if is_testing else 'train'})"
                                " elapsed time: %.1fs" % (perf_counter() - start_perf_t))
            logged_finished = student.agents.finished()

        if not enough_memory:
            min_memory = dist_min(student.device, len(student.memory))
            if min_memory >= student.config.min_memory:
                enough_memory = True

        if not is_testing and enough_memory:
            student.sample_learn(args.memory_sample_rounds, args.memory_sample_size)

    student.dist_summary()
    if channel is not None:
        channel.close()
    if connection is not None:
        connection.close()


def main(args):
    args.mode = None
    logger = logging.getLogger("Rank {}({})|main".format(args.local_rank, getpid()))

    if args.local_rank == 0:
        logger.info("Started: {}".format(args))

    args.device = args.local_rank % torch.cuda.device_count()
    torch.cuda.set_device(args.device)
    if "NCCL_SOCKET_IFNAME" not in os.environ and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(backend=dist.Backend.NCCL, init_method='env://')

    data_config = construct_data_config(args)
    if args.local_rank == 0:
        logger.info("DataConfig: {}".format(vars(data_config)))

    if args.queue_mode == 'amqp':
        get_conn_channel(args, True)

    logger.info("Loading index...")
    index = Index(data_config)
    train_tuids = index.train_tUIDs()
    valid_tuids = index.valid_tUIDs()
    original_train_size = len(train_tuids)
    original_valid_size = len(valid_tuids)
    train_tuids = _slice_tuids(train_tuids, args.train_table_limit,
                               args.train_table_offset, args.train_table_stride)
    valid_tuids = _slice_tuids(valid_tuids, args.valid_table_limit,
                               args.valid_table_offset, args.valid_table_stride)
    train_size = len(train_tuids)
    valid_size = len(valid_tuids)
    logger.info("Training subset: %d / %d tables (offset=%d, stride=%d, limit=%s)",
                train_size, original_train_size,
                args.train_table_offset, args.train_table_stride, args.train_table_limit)
    logger.info("Validation subset: %d / %d tables (offset=%d, stride=%d, limit=%s)",
                valid_size, original_valid_size,
                args.valid_table_offset, args.valid_table_stride, args.valid_table_limit)

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    if args.queue_mode == 'amqp' and global_rank == world_size - 1:
        logger.info("Setting up ventilator...")
        recv_queue, send_queue = mp.Pipe(duplex=False)
        ventilator = mp.Process(target=task_queue, args=(recv_queue, world_size, index, args))
        ventilator.start()

    logger.info("Preparing DDP model...")
    model, experiment_name = create_model(args)
    experiment_name += '-MC-RL'
    logger.info(f"{args.model_name} #parameters = {num_params(model)}")
    ddp, optimizer, criterion = prepare_model(model, args.device, args)
    args.model_save_path = path.join(args.model_save_path, experiment_name)

    student_config = StudentConfig(optimizer, criterion,
                                   memory_size=args.memory_size, min_memory=args.min_memory,
                                   random_train=args.random_train,
                                   log_tag=f"{args.local_rank}({getpid()})", log_freq=args.log_freq_batch,
                                   log_dir=path.join(args.model_save_path, args.log_save_path),
                                   mc_top_k=args.mc_top_k,
                                   mc_rollout_depth=args.mc_rollout_depth,
                                   mc_num_rollouts=args.mc_num_rollouts,
                                   mc_discount=args.mc_discount,
                                   mc_rollout_weight=args.mc_rollout_weight,
                                   mc_seed=args.mc_seed)
    search_config = get_search_config(True, args.search_limits, data_config.search_all_types)
    summary_writer = SummaryWriter(log_dir=path.join(args.summary_path, experiment_name + f"-R{args.local_rank}"))
    if args.local_rank == 0:
        logger.info("StudentConfig: {}".format(vars(student_config)))
        logger.info("SearchConfig: {}".format(vars(search_config)))
    student = Student(student_config, data_config, search_config, ddp, args.apex,
                      args.device, args.local_rank, summary_writer)

    for epoch in range(args.restart_epoch + 1, args.epochs):
        logger.info("Starting epoch %d" % epoch)

        if args.queue_mode == 'amqp':
            if global_rank == world_size - 1:
                send_queue.send(SenderTask.Train)
            iteration(epoch, student, False, train_size, args)
        else:
            iteration(epoch, student, False, train_size, args, local_tuids=train_tuids[global_rank::world_size])

        if args.local_rank == 0:
            output_path = student.save_checkpoint(args.model_save_path)
            logger.info("EP-%d model saved at: %s" % (epoch, output_path))

        if args.queue_mode == 'amqp':
            if global_rank == world_size - 1:
                send_queue.send(SenderTask.Valid)
            iteration(epoch, student, True, valid_size, args)
        else:
            iteration(epoch, student, True, valid_size, args, local_tuids=valid_tuids[global_rank::world_size])

    if args.queue_mode == 'amqp' and global_rank == world_size - 1:
        logger.info("Stopping ventilator...")
        send_queue.send(SenderTask.Stop)
        ventilator.join()
    logger.info("Finished!")


if __name__ == '__main__':
    main(parse_args())
