# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Launch this script following https://pytorch.org/docs/stable/distributed.html#launch-utility
Or find examples in learn_dist.sh
"""
import argparse
import glob
import logging
import multiprocessing as mp
import numpy as np
import os
import pika
import random
import sys
import torch
import torch.distributed as dist
import traceback
from data import Index, DEFAULT_LANGUAGES, DEFAULT_FEATURE_CHOICES, DEFAULT_ANALYSIS_TYPES
from enum import IntEnum
from helper import construct_data_config, create_model, prepare_model
from model import DEFAULT_MODEL_SIZES, DEFAULT_MODEL_NAMES
from os import path, getpid
from reinforce.student.config import StudentConfig
from reinforce.student.update_teacher_student import UpdateTeacherStudent
from search.agent import get_search_config, DEFAULT_SEARCH_LIMITS
from time import perf_counter
from torch.utils.tensorboard import SummaryWriter
from util import num_params

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.getLogger("pika").setLevel(logging.WARNING)


def parse_args():
    """
    Helper function parsing the command line options
    """
    parser = argparse.ArgumentParser(description="RL Training")

    # Model and evaluation paths
    parser.add_argument('-p', '--pre_model_file', type=str, metavar='PATH',
                        help='file path to the pre-trained model (as the starting point)')
    parser.add_argument('-m', "--model_save_path", default="/storage/models/", type=str)
    parser.add_argument('-l', '--log_save_path', default="evaluations", type=str, metavar='PATH',
                        help='subdir path of model_save_path to log the evaluation metrics during validation/testing')

    parser.add_argument("--corpus_path", type=str, required=True, help="The corpus path for metadata task.")
    parser.add_argument("--lang", "--specify_lang", choices=DEFAULT_LANGUAGES, default='en', type=str,
                        help="Specify the header language(s) to load tables.")

    # Model choose configurations
    parser.add_argument("--model_name", choices=DEFAULT_MODEL_NAMES, default="tf", type=str)

    # Experiment settings
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

    # Reinforcement learning parameters
    parser.add_argument("--negative_weight", default=0.02, type=float, help="Negative class weight for NLLLoss.")
    # Replay memory and its sampling
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

    # Learning agent/student
    parser.add_argument('--random_train', action="store_true",
                        help="Set the flag if training samples will be generated randomly.")
    parser.add_argument('--max_tables', default=64, type=int, metavar='N',
                        help='number of tables to handle at the same time')
    parser.add_argument('--search_limits', choices=DEFAULT_SEARCH_LIMITS, default="e200-b8-r4c2", type=str,
                        help="Search config option")
    parser.add_argument('--update_teacher_exact', default=0.95, type=float,
                        help='teacher dense reward for an exact ground-truth next action')
    parser.add_argument('--update_teacher_default', default=0.05, type=float,
                        help='teacher dense reward for ordinary non-target actions')
    parser.add_argument('--update_teacher_same_token', default=0.10, type=float,
                        help='teacher dense reward for non-exact actions with the same token type as a target action')
    parser.add_argument('--update_teacher_field', default=0.15, type=float,
                        help='teacher dense reward for non-exact field actions when the target action is a field')
    parser.add_argument('--update_teacher_same_field_type', default=0.35, type=float,
                        help='teacher dense reward for non-exact field actions matching a target field type')
    parser.add_argument('--update_teacher_positive_threshold', default=0.5, type=float,
                        help='threshold used only for binary training metrics from teacher dense rewards')
    parser.add_argument('--teacher_policy_epsilon_start', default=0.2, type=float,
                        help='initial epsilon for teacher-policy exploration')
    parser.add_argument('--teacher_policy_epsilon_end', default=0.02, type=float,
                        help='minimum epsilon for teacher-policy exploration')
    parser.add_argument('--teacher_policy_epsilon_decay', default=0.8, type=float,
                        help='epoch-level epsilon decay for teacher-policy exploration')
    parser.add_argument('--teacher_policy_explore_top_m', default=5, type=int,
                        help='sample among the top-M scored valid actions when teacher epsilon exploration is selected')
    parser.add_argument('--teacher_policy_seed', default=None, type=int,
                        help='optional random seed for teacher-policy exploration')
    parser.add_argument('--teacher_stage', choices=['collect', 'train'], default='collect', type=str,
                        help='collect teacher preference data or train a student from collected teacher data')
    parser.add_argument('--teacher_data_path', default="../Results/teacher_data", type=str,
                        help='directory used to save/load teacher-generated preference samples')
    parser.add_argument('--teacher_collect_ratio', default=1.0, type=float,
                        help='fraction of training tables to use when collecting teacher data')
    parser.add_argument('--teacher_train_steps_per_epoch', default=1000, type=int,
                        help='number of supervised student updates per epoch in teacher train stage')

    # Distributed computing
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
    Train = 0,
    Valid = 1,
    # TODO: add Test
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
    """
    Ventilator: Send tUIDs into RabbitMQ for GPU processes
    :param receiver: SenderTask command queue
    :param world_size: number of GPUs/processes
    :param index: dataset index
    :param args: arguments
    """
    logger = logging.getLogger("pika sender")

    def send_msg(channel, message: str):
        channel.basic_publish(exchange='', routing_key=args.amqp_routing_key, body=message.encode("ascii"),
                              properties=pika.BasicProperties(delivery_mode=2))

    logger.info("Wait for commands...")
    while True:
        task = receiver.recv()
        if task == SenderTask.Train:
            logger.info("Train Command Recv.")
            train_tUIDS = index.train_tUIDs()
            connection, channel = get_conn_channel(args)
            for i in np.random.permutation(len(train_tUIDS)):
                send_msg(channel, train_tUIDS[i])
            for _ in range(world_size):
                send_msg(channel, "")  # Empty string to notify the queue is empty.
            logger.info(f"Train {len(train_tUIDS)} + {world_size} sent.")
            channel.close()
            connection.close()
        elif task == SenderTask.Valid:
            logger.info("Valid Command Recv.")
            valid_tUIDs = index.valid_tUIDs()
            connection, channel = get_conn_channel(args)
            for tUID in valid_tUIDs:
                send_msg(channel, tUID)
            for _ in range(world_size):
                send_msg(channel, "")  # Empty string to notify the queue is empty.
            logger.info(f"Valid {len(valid_tUIDs)} + {world_size} sent.")
            channel.close()
            connection.close()
        else:
            logger.info("Stop")
            return


# TODO: implement a info_queue to collect evaluation metrics of each tUID searching
# See summary() in test_agent_mp.py


def dist_sum(device, value):
    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor)
    return tensor.item()


def dist_min(device, value):
    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return tensor.item()


def iteration(epoch: int, student: UpdateTeacherStudent, is_testing: bool, n_tables: int, args, local_tuids=None):
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
    # Keep looping while there are tUIDs not fully searched by an agent.
    while dist_sum(student.device, student.agents.finished()) < n_tables:  # Sync point!
        # Take in and start new agent if the RabbitMQ is not empty, and the student still has room for more agents.
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
                # print(f"({student.local_rank}) Waiting to receive.")
                method, properties, body = channel.basic_get(args.amqp_routing_key, auto_ack=True)
            except (ConnectionResetError, pika.exceptions.StreamLostError):
                traceback.print_exc(file=sys.stdout)
                student.logger.info("Setting up connection again...")  # In case when previous connection was lost.
                connection, channel = get_conn_channel(args)
                method, properties, body = channel.basic_get(args.amqp_routing_key, auto_ack=True)

            if body is None:
                # student.logger.info(f"Got NONE ({cnt})!")
                continue
            cnt += 1
            tUID = body.decode("ascii")

            if len(tUID) == 0:  # Empty string means end of the iteration.
                # student.logger.info(f"Got EMPTY ({cnt})!")
                queue_empty = True
            else:
                # student.logger.info(f"Got {tUID} [{len(tUID)}] ({cnt})")
                student.add_table(tUID)

        # print(f"({student.local_rank}) Step with {student.agents.remaining()} agents!")
        student.act_step()  # Sync point!
        if student.agents.finished() - logged_finished >= args.log_freq_agent:
            student.logger.info(f"Agents finished={student.agents.finished()} remaining={student.agents.remaining()}"
                                f" error={student.agents.error_cnt}! " +
                                f"EP-{epoch} ({'test/valid' if is_testing else 'train'})"
                                " elapsed time: %.1fs" % (perf_counter() - start_perf_t))
            logged_finished = student.agents.finished()

        # print(f"({student.local_rank}) Memory size {len(student.memory)}")
        if not enough_memory:  # Check if everyone have enough replay experiences to start sampling
            min_memory = dist_min(student.device, len(student.memory))  # Sync point!
            if min_memory >= student.config.min_memory:
                enough_memory = True

        if not is_testing and enough_memory:
            # If number of experiences is large enough, then start learning.
            # print(f"({student.local_rank}) Sample {args.memory_sample_size} to learn")
            student.sample_learn(args.memory_sample_rounds, args.memory_sample_size)  # Sync point!

    student.dist_summary()  # Sync point! Reduce and log overall metrics.
    if channel is not None:
        channel.close()
    if connection is not None:
        connection.close()


def collect_iteration(epoch: int, student: UpdateTeacherStudent, n_tables: int, args, local_tuids=None):
    student.reset(epoch, False)
    queue_empty = False
    logged_finished = 0
    local_tuids = [] if local_tuids is None else local_tuids
    local_idx = 0

    while dist_sum(student.device, student.agents.finished()) < n_tables:
        while not queue_empty and student.agents.remaining() < args.max_tables:
            if local_idx >= len(local_tuids):
                queue_empty = True
                break
            student.add_table(local_tuids[local_idx])
            local_idx += 1

        student.act_step()
        if student.agents.finished() - logged_finished >= args.log_freq_agent:
            student.logger.info("Teacher collect finished=%d remaining=%d error=%d! EP-%d elapsed time: %.1fs" % (
                student.agents.finished(), student.agents.remaining(), student.agents.error_cnt,
                epoch, perf_counter() - student.start_perf_time))
            logged_finished = student.agents.finished()


def save_teacher_memory_samples(student: UpdateTeacherStudent, output_dir: str, rank: int):
    os.makedirs(output_dir, exist_ok=True)
    output_path = path.join(output_dir, f"teacher_samples.rank-{rank}.pt")
    torch.save(student.teacher_memory_samples(), output_path)
    return output_path


def load_teacher_memory_samples(input_dir: str):
    sample_paths = sorted(glob.glob(path.join(input_dir, "teacher_samples.rank-*.pt")))
    samples = []
    for sample_path in sample_paths:
        samples.extend(torch.load(sample_path))
    return samples, sample_paths


def teacher_collect_main(args, student: UpdateTeacherStudent, train_tuids, global_rank: int, world_size: int):
    local_tuids = train_tuids[global_rank::world_size]
    if args.teacher_collect_ratio < 1.0:
        keep = max(1, int(len(local_tuids) * args.teacher_collect_ratio))
        local_tuids = local_tuids[:keep]
    n_tables = int(dist_sum(student.device, len(local_tuids)))

    for epoch in range(args.restart_epoch + 1, args.epochs):
        logging.getLogger(f"Rank {args.local_rank}({getpid()})|collect").info("Starting teacher collect epoch %d", epoch)
        collect_iteration(epoch, student, n_tables, args, local_tuids=local_tuids)

    output_path = save_teacher_memory_samples(student, args.teacher_data_path, global_rank)
    logging.getLogger(f"Rank {args.local_rank}({getpid()})|collect").info("Saved teacher samples to: %s", output_path)


def teacher_train_main(args, student: UpdateTeacherStudent, global_rank: int, world_size: int):
    all_samples, sample_paths = load_teacher_memory_samples(args.teacher_data_path)
    if not all_samples:
        raise FileNotFoundError(f"No teacher samples found under {args.teacher_data_path}")

    local_samples = all_samples[global_rank::world_size]
    if not local_samples:
        raise RuntimeError(f"Rank {global_rank} received no teacher samples from {sample_paths}")

    logger = logging.getLogger(f"Rank {args.local_rank}({getpid()})|teacher-train")
    logger.info("Loaded %d total teacher samples from %d shard(s); local shard size=%d",
                len(all_samples), len(sample_paths), len(local_samples))

    for epoch in range(args.restart_epoch + 1, args.epochs):
        student.reset(epoch, False)
        for step in range(args.teacher_train_steps_per_epoch):
            batch_size = min(args.memory_sample_size, len(local_samples))
            batch = random.sample(local_samples, batch_size)
            student.train_supervised_batch(batch)
            if (step + 1) % max(1, args.log_freq_batch) == 0:
                logger.info("Teacher student epoch=%d step=%d/%d avg_loss=%.6f",
                            epoch, step + 1, args.teacher_train_steps_per_epoch,
                            student.loss_sum / max(1, student.batch_cnt))

        if args.local_rank == 0:
            output_path = student.save_checkpoint(args.model_save_path)
            logger.info("Teacher student EP-%d model saved at: %s", epoch, output_path)


# TODO: load all tables before searching?
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
    data_config.update_teacher_exact = args.update_teacher_exact
    data_config.update_teacher_default = args.update_teacher_default
    data_config.update_teacher_same_token = args.update_teacher_same_token
    data_config.update_teacher_field = args.update_teacher_field
    data_config.update_teacher_same_field_type = args.update_teacher_same_field_type
    if args.local_rank == 0:
        logger.info("DataConfig: {}".format(vars(data_config)))

    if args.queue_mode == 'amqp':
        get_conn_channel(args, True)  # To Clean Up the Queue!

    logger.info("Loading index...")
    index = Index(data_config)
    train_tuids = index.train_tUIDs()
    valid_tuids = index.valid_tUIDs()
    train_size = len(train_tuids)
    valid_size = len(valid_tuids)

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    if args.queue_mode == 'amqp' and global_rank == world_size - 1:
        logger.info("Setting up ventilator...")
        recv_queue, send_queue = mp.Pipe(duplex=False)
        ventilator = mp.Process(target=task_queue, args=(recv_queue, world_size, index, args))
        ventilator.start()

    logger.info("Preparing DDP model...")
    model, experiment_name = create_model(args)
    experiment_name += '-update_teacher-RL'
    logger.info(f"{args.model_name} #parameters = {num_params(model)}")
    ddp, optimizer, criterion = prepare_model(model, args.device, args)
    args.model_save_path = path.join(args.model_save_path, experiment_name)

    student_config = StudentConfig(optimizer, criterion,
                                   memory_size=args.memory_size, min_memory=args.min_memory,
                                   random_train=args.random_train,
                                   log_tag=f"{args.local_rank}({getpid()})", log_freq=args.log_freq_batch,
                                   log_dir=path.join(args.model_save_path, args.log_save_path))
    search_config = get_search_config(True, args.search_limits, data_config.search_all_types)
    summary_writer = SummaryWriter(log_dir=path.join(args.summary_path, experiment_name + f"-R{args.local_rank}"))
    if args.local_rank == 0:
        logger.info("StudentConfig: {}".format(vars(student_config)))
        logger.info("SearchConfig: {}".format(vars(search_config)))
    setattr(student_config, "update_teacher_negative_weight", args.negative_weight)
    setattr(student_config, "update_teacher_positive_threshold", args.update_teacher_positive_threshold)
    setattr(student_config, "teacher_policy_epsilon_start", args.teacher_policy_epsilon_start)
    setattr(student_config, "teacher_policy_epsilon_end", args.teacher_policy_epsilon_end)
    setattr(student_config, "teacher_policy_epsilon_decay", args.teacher_policy_epsilon_decay)
    setattr(student_config, "teacher_policy_explore_top_m", args.teacher_policy_explore_top_m)
    setattr(search_config, "teacher_policy_seed", args.teacher_policy_seed)
    student = UpdateTeacherStudent(student_config, data_config, search_config, ddp, args.apex,
                                   args.device, args.local_rank, summary_writer)

    if args.teacher_stage == 'collect':
        if args.queue_mode != 'local':
            raise ValueError("update_teacher collect stage currently supports queue_mode=local only.")
        teacher_collect_main(args, student, train_tuids, global_rank, world_size)
    elif args.teacher_stage == 'train':
        teacher_train_main(args, student, global_rank, world_size)
    else:
        raise ValueError(f"Unsupported teacher stage: {args.teacher_stage}")

    if args.queue_mode == 'amqp' and global_rank == world_size - 1:
        logger.info("Stopping ventilator...")
        send_queue.send(SenderTask.Stop)
        ventilator.join()
    logger.info("Finished!")


if __name__ == '__main__':
    # mp.set_start_method("spawn", force=True)
    main(parse_args())
